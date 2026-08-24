// SPDX-License-Identifier: Apache-2.0
// test_ane_tail_fused.cpp - FULL GDN tail as ONE fp16 K-tiled ANE program.
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>

#include <IOSurface/IOSurface.h>

#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"

static inline float hf(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0) u = m ? 0 : s;
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e - 15u + 127u) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint16_t fh(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000, re = (x >> 23) & 0xff, mt = x & 0x7fffff;
    if (re == 0xff) return (uint16_t)(s | 0x7c00);
    int ex = (int)re - 127 + 15;
    if (ex >= 31) return (uint16_t)(s | 0x7c00);
    if (ex <= 0) { if (ex < -10) return (uint16_t)s;
        uint32_t mm = mt | 0x800000; int sh = 14 - ex;
        uint32_t half = mm >> sh, rem = mm & ((1u << sh) - 1), mid = 1u << (sh - 1);
        return (uint16_t)(s | (half + ((rem > mid || (rem == mid && (half & 1))) ? 1 : 0))); }
    uint32_t h = ((uint32_t)ex << 10) | (mt >> 13);
    uint32_t rem = mt & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return (uint16_t)(s | h);
}
static const std::string Q(const std::string& s) { return "\"" + s + "\""; }
static const std::string Q(const char* s) { return Q(std::string(s)); }

static const char* BI =
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

int main(int argc, char** argv) {
    const size_t H = 5120, C = 6144, I = 17408, S = 32;
    size_t KC = 2048;
    if (argc > 1) KC = (size_t)std::atoi(argv[1]);

    auto lcg = [s = 777u](float scale) mutable {
        s = s * 1664525u + 1013904223u;
        return ((s >> 8) & 0xffff) / 65535.0f * 2.0f * scale - scale;
    };

    std::vector<std::string> wnames;
    std::vector<const void*> wptrs;
    std::vector<size_t> wsizes;
    std::vector<std::vector<uint16_t>> keep;

    auto add_blob = [&](const std::string& tag, const std::vector<uint16_t>& v) {
        keep.push_back(v);
        wnames.push_back(tag + ".bin");
        wptrs.push_back(keep.back().data());
        wsizes.push_back(v.size() * sizeof(uint16_t));
    };
    auto make_mat = [&](size_t rows, size_t cols) {
        std::vector<uint16_t> m(rows * cols);
        for (auto& v : m) v = fh(lcg(0.02f));
        return m;
    };
    auto mat_o = make_mat(H, C);
    auto mat_gu = make_mat(2 * I, H);
    auto mat_dn = make_mat(H, I);

    std::ostringstream mil;
    mil << "program(1.3)\n" << BI << "\n{\n";
    mil << "  func main<ios18>(tensor<fp16, [1, " << (C + H) << ", 1, " << S << "]> xin) {\n";

    // K-tiled conv emitter
    auto emit_conv = [&](const char* base, size_t rows,
                         const std::vector<uint16_t>& src, size_t ic,
                         const std::string& xexpr, const std::string& out_name) {
        int T = (int)((ic + KC - 1) / KC);
        std::vector<size_t> ks(T, ic / T);
        for (int r = 0; r < (int)(ic % T); ++r) ks[r] += 1;
        size_t off = 0;
        for (int t = 0; t < T; ++t) {
            const size_t kc = ks[t];
            const std::string tag = std::string(base) + "_k" + std::to_string(t);
            std::vector<uint16_t> chunk(rows * kc);
            for (size_t r = 0; r < rows; ++r)
                for (size_t k = 0; k < kc; ++k)
                    chunk[r * kc + k] = src[r * ic + off + k];
            add_blob(tag + ".bin", chunk);
            mil << "    tensor<fp16, [" << rows << ", " << kc << ", 1, 1]> " << tag << "W"
                << " = const()[name=" << Q(tag + "W") << ", val=tensor<fp16, [" << rows
                << ", " << kc << ", 1, 1]>(BLOBFILE(path=string(" << Q("@model_path/weights/" + tag + ".bin")
                << "), offset=uint64(64)))];\n"
                << "    tensor<int32, [4]> " << tag << "b = const()[name=" << Q(tag + "b")
                << ", val=tensor<int32, [4]>([0," << off << ",0,0])];\n"
                << "    tensor<int32, [4]> " << tag << "e = const()[name=" << Q(tag + "e")
                << ", val=tensor<int32, [4]>([1," << (off + kc) << ",1," << S << ")];\n"
                << "    tensor<fp16, [1, " << kc << ", 1, " << S << "]> " << tag << "x"
                << " = slice_by_index(begin=" << tag << "b, end=" << tag << "e, x="
                << xexpr << ")[name=" << Q(tag + "x") << "];\n";
            if (t == 0) {
                mil << "    tensor<fp16, [1, " << rows << ", 1, " << S << "]> " << out_name
                    << " = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight="
                    << tag << "W, x=" << tag << "x)[name=" << Q(out_name) << "];\n";
            } else {
                mil << "    tensor<fp16, [1, " << rows << ", 1, " << S << "]> " << tag << "p"
                    << " = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight="
                    << tag << "W, x=" << tag << "x)[name=" << Q(tag + "p") << "];\n"
                    << "    tensor<fp16, [1, " << rows << ", 1, " << S << "]> " << out_name
                    << " = add(x=" << out_name << ", y=" << tag << "p)[name="
                    << Q(out_name + "_a" + std::to_string(t)) << "];\n";
            }
            off += kc;
        }
    };

    mil << "    string pt = const()[name=" << Q("pt") << ", val=string(" << Q("valid") << ")];\n"
        << "    tensor<int32, [2]> st = const()[name=" << Q("st") << ", val=tensor<int32, [2]>([1,1])];\n"
        << "    tensor<int32, [4]> pd = const()[name=" << Q("pd") << ", val=tensor<int32, [4]>([0,0,0,0])];\n"
        << "    tensor<int32, [2]> dl = const()[name=" << Q("dl") << ", val=tensor<int32, [2]>([1,1])];\n"
        << "    int32 gr = const()[name=" << Q("gr") << ", val=int32(1)];\n"
        << "    tensor<int32, [4]> cb = const()[name=" << Q("cb") << ", val=tensor<int32, [4]>([0,0,0,0])];\n"
        << "    tensor<int32, [4]> ce = const()[name=" << Q("ce") << ", val=tensor<int32, [4]>([1," << C << ",1," << S << "])];\n"
        << "    tensor<int32, [4]> rb = const()[name=" << Q("rb") << ", val=tensor<int32, [4]>([0," << C << ",0,0])];\n"
        << "    tensor<int32, [4]> re = const()[name=" << Q("re") << ", val=tensor<int32, [4]>([1," << (C + H) << ",1," << S << "])];\n"
        << "    tensor<fp16, [1, " << C << ", 1, " << S << "]> core = slice_by_index(begin=cb, end=ce, x=xin)[name=" << Q("core") << "];\n"
        << "    tensor<fp16, [1, " << H << ", 1, " << S << "]> res = slice_by_index(begin=rb, end=re, x=xin)[name=" << Q("res") << "];\n";

    emit_conv("o", H, mat_o, C, "core", "outp");
    mil << "    tensor<fp16, [1, " << H << ", 1, " << S << "]> h = add(x=res, y=outp)[name=" << Q("h") << "];\n";
    mil << "    tensor<fp16, [1, " << H << ", 1, " << S << "]> sq = mul(x=h, y=h)[name=" << Q("sq") << "];\n";

    // RMS mean via ones-kernel conv
    {
        std::vector<uint16_t> one(H);
        for (auto& v : one) v = fh(1.0f / float(H));
        add_blob("on.bin", one);
    }
    mil << "    tensor<fp16, [1, 1, 1, 1]> onw = const()[name=" << Q("onw")
        << ", val=tensor<fp16, [1, 1, 1, 1]>(BLOBFILE(path=string(" << Q("@model_path/weights/on.bin")
        << "), offset=uint64(64)))];\n"
        << "    tensor<fp16, [1, 1, 1, " << S << "]> ms = conv(dilations=dl, groups=gr, pad=pd, "
        << "pad_type=pt, strides=st, weight=onw, x=sq)[name=" << Q("ms") << "];\n"
        << "    fp16 ep = const()[name=" << Q("ep") << ", val=fp16(0x1.0p-6)];\n"
        << "    tensor<fp16, [1, 1, 1, " << S << "]> msa = add(x=ms, y=ep)[name=" << Q("msa") << "];\n"
        << "    tensor<fp16, [1, 1, 1, " << S << "]> sd = sqrt(x=msa)[name=" << Q("sd") << "];\n"
        << "    tensor<fp16, [1, " << H << ", 1, " << S << "]> nx = real_div(x=h, y=sd)[name=" << Q("nx") << "];\n";

    emit_conv("gu", 2 * I, mat_gu, H, "nx", "guc");
    mil << "    tensor<fp16, [1, " << I << ", 1, " << S << "]> g0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), "
        << "end=tensor<int32, [4]>([1," << I << ",1," << S << "]), x=guc)[name=" << Q("g0") << "];\n"
        << "    tensor<fp16, [1, " << I << ", 1, " << S << "]> u0 = slice_by_index(begin=tensor<int32, [4]>([0," << I
        << ",0,0]), end=tensor<int32, [4]>([1," << (2 * I) << ",1," << S << "]), x=guc)[name=" << Q("u0") << "];\n"
        << "    tensor<fp16, [1, " << I << ", 1, " << S << "]> sg = sigmoid(x=g0)[name=" << Q("sg") << "];\n"
        << "    tensor<fp16, [1, " << I << ", 1, " << S << "]> si = mul(x=g0, y=sg)[name=" << Q("si") << "];\n"
        << "    tensor<fp16, [1, " << I << ", 1, " << S << "]> ac = mul(x=si, y=u0)[name=" << Q("ac") << "];\n";

    emit_conv("dn", H, mat_dn, I, "ac", "dnp");
    mil << "    tensor<fp16, [1, " << H << ", 1, " << S << "]> y = add(x=h, y=dnp)[name=" << Q("y") << "];\n";
    mil << "  } -> (y);\n}\n";

    ANEContext* ctx = ane_context_create();
    if (!ctx) { std::cerr << "no ctx\n"; return 1; }
    std::vector<const char*> np_;
    std::vector<const void*> dp;
    std::vector<size_t> szv;
    for (size_t i = 0; i < wnames.size(); ++i) {
        np_.push_back(wnames[i].c_str());
        dp.push_back(wptrs[i]);
        szv.push_back(wsizes[i]);
    }
    const auto tc = std::chrono::high_resolution_clock::now();
    ANEModel* model = ane_model_compile_mil(ctx, mil.str().c_str(), np_.data(),
                                            dp.data(), szv.data(), np_.size(), 0, 21);
    double compile_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tc).count();
    if (!model) { std::cout << "COMPILE FAILED\n"; return 1; }
    std::cout << "compiled (" << np_.size() << " blobs, KC=" << KC << ") in "
              << compile_ms << " ms\n";

    IOSurfaceRef in_s = metal_create_iosurface((C + H) * S * 2);
    IOSurfaceRef y_s = metal_create_iosurface(H * S * 2);
    std::vector<uint16_t> xin((size_t)(C + H) * S);
    for (size_t i = 0; i < xin.size(); ++i) xin[i] = fh(lcg(0.1f));
    IOSurfaceLock(in_s, 0, nullptr);
    std::memcpy(IOSurfaceGetBaseAddress(in_s), xin.data(), xin.size() * 2);
    IOSurfaceUnlock(in_s, 0, nullptr);
    ANERequest* req = ane_request_create(ctx, model, in_s, y_s, 0);
    if (!req) { std::cout << "REQ FAIL\n"; return 1; }

    for (int i = 0; i < 3; ++i)
        ane_request_evaluate(ctx, model, req, nullptr, 0, nullptr, 0);
    const int N = 30;
    auto tb = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        ane_request_evaluate(ctx, model, req, nullptr, 0, nullptr, 0);
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tb).count() / N;
    double flops = 2.0 * (H*C + 2*I*H + H*I) * S;
    std::printf("FULL FUSED TAIL S=%zu KC=%zu: %.3f ms/eval | %.2f TFLOPS | x64 = %.1f ms/pass\n",
                S, KC, ms, flops/ms/1e9, ms*64);

    ane_request_release(req);
    CFRelease(in_s); CFRelease(y_s);
    ane_model_release(model);
    ane_context_destroy(ctx);
    return 0;
}
