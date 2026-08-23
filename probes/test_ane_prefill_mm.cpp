// SPDX-License-Identifier: Apache-2.0
/*
 * test_ane_prefill_mm.cpp - ANE dynamic-matmul throughput at PREFILL shapes.
 *
 * Question: can a weight-as-input ANE matmul (proven exact in ane-as test
 * [10]) sustain enough TFLOPS at large spatial sizes to beat the hybrid
 * prefill backend (~71 tok/s ~= 5 TFLOPS effective on this model)?
 *
 * Sweep: ic=oc=5120, S in {32,128,512,1024,2048}. Reports ms, TFLOPS,
 * and correctness spot-checks against a CPU reference.
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

#include <IOSurface/IOSurface.h>

#include "../runtime/ane_c_bridge.h"
#include "../runtime/metal_engine.h"

static inline float hf(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1f, m = h & 0x03ff;
    uint32_t u;
    if (e == 0) { u = m ? 0 : sign; }
    else if (e == 31) { u = sign | 0x7f800000u | (m << 13); }
    else { u = sign | ((e - 15 + 127) << 23) | (m << 13); }
    float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint16_t fh(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000, re = (x >> 23) & 0xff, mt = x & 0x7fffff;
    if (re == 0xff) return (uint16_t)(s | 0x7c00);
    const int ex = (int)re - 127 + 15;
    if (ex >= 31) return (uint16_t)(s | 0x7c00);
    if (ex <= 0) { if (ex < -10) return (uint16_t)s;
        const uint32_t mm = mt | 0x800000; const int sh = 14 - ex;
        const uint32_t half = mm >> sh, rem = mm & ((1u<<sh)-1), mid = 1u<<(sh-1);
        return (uint16_t)(s | (half + ((rem > mid || (rem == mid && (half&1))) ? 1:0))); }
    uint32_t h = (ex << 10) | (mt >> 13);
    const uint32_t rem = mt & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
    return (uint16_t)(s | h);
}

static std::string fmt4(const char* name, const char* type_and_vals) {
    return std::string(name) + type_and_vals;
}

static const char* kBuildInfo =
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

// Build each tensor line independently - no cross-line positional args.
static std::string gen_dyn_mm_mil(int ic, int oc, int sq) {
    char L[24][512];
    std::snprintf(L[0], 512, "program(1.3)\n%s\n{\n", kBuildInfo);
    std::snprintf(L[1], 512, "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sq + oc);
    std::snprintf(L[2], 512, "    tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    std::snprintf(L[3], 512, "    tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, sq);
    std::snprintf(L[4], 512, "    tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, sq);
    std::snprintf(L[5], 512, "    tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", sq);
    std::snprintf(L[6], 512, "    tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc);
    std::snprintf(L[7], 512, "    tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc);
    std::snprintf(L[8], 512, "    tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, sq);
    std::snprintf(L[9], 512, "    tensor<fp16, [1,1,%d,%d]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", ic, sq);
    std::snprintf(L[10], 512, "    tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    std::snprintf(L[11], 512, "    tensor<fp16, [1,1,%d,%d]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", sq, ic);
    std::snprintf(L[12], 512, "    tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc);
    std::snprintf(L[13], 512, "    tensor<fp16, [1,1,%d,%d]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", ic, oc);
    std::snprintf(L[14], 512, "    bool bF = const()[name=string(\"bF\"), val=bool(false)];\n");
    std::snprintf(L[15], 512, "    tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", sq, oc);
    std::snprintf(L[16], 512, "    tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", sq, oc);
    std::snprintf(L[17], 512, "    tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, sq);
    std::snprintf(L[18], 512, "    tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, sq);
    std::snprintf(L[19], 512, "  } -> (y);\n}\n");
    std::string out;
    for (int i = 0; i <= 19; ++i) out += L[i];
    return out;
}

// Same GEMM expressed as grouped=1 conv: y[oc,1,S] = W[oc,ic,1,1] * x[ic,1,S].
// Weights still stream from the input surface tail.
static std::string gen_dyn_conv_mil(int ic, int oc, int sq) {
    char L[26][512];
    std::snprintf(L[0], 512, "program(1.3)\n%s\n{\n", kBuildInfo);
    std::snprintf(L[1], 512, "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sq + oc);
    std::snprintf(L[2], 512, "    tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    std::snprintf(L[3], 512, "    tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, sq);
    std::snprintf(L[4], 512, "    tensor<fp16, [1,%d,1,%d]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", ic, sq);
    std::snprintf(L[5], 512, "    tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", sq);
    std::snprintf(L[6], 512, "    tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", ic, oc);
    std::snprintf(L[7], 512, "    tensor<fp16, [1,%d,1,%d]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", ic, oc);
    // wt [1,ic,1,oc] -> [1,1,ic,oc] -> transpose -> [1,1,oc,ic] -> weight [oc,ic,1,1]
    std::snprintf(L[8], 512, "    tensor<int32, [4]> rw1 = const()[name=string(\"rw1\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", ic, oc);
    std::snprintf(L[9], 512, "    tensor<fp16, [1,1,%d,%d]> w2 = reshape(shape=rw1,x=wt)[name=string(\"w2\")];\n", ic, oc);
    std::snprintf(L[10], 512, "    tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n");
    std::snprintf(L[11], 512, "    tensor<fp16, [1,1,%d,%d]> w3 = transpose(perm=pm,x=w2)[name=string(\"w3\")];\n", oc, ic);
    std::snprintf(L[12], 512, "    tensor<int32, [4]> rw2 = const()[name=string(\"rw2\"), val=tensor<int32, [4]>([%d,%d,1,1])];\n", oc, ic);
    std::snprintf(L[13], 512, "    tensor<fp16, [%d,%d,1,1]> W = reshape(shape=rw2,x=w3)[name=string(\"W\")];\n", oc, ic);
    std::snprintf(L[14], 512, "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n");
    std::snprintf(L[15], 512, "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n");
    std::snprintf(L[16], 512, "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n");
    std::snprintf(L[17], 512, "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n");
    std::snprintf(L[18], 512, "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n");
    std::snprintf(L[19], 512, "    tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=act)[name=string(\"y\")];\n", oc, sq);
    std::snprintf(L[20], 512, "  } -> (y);\n}\n");
    std::string out;
    for (int i = 0; i <= 20; ++i) out += L[i];
    return out;
}

// K-tiled dynamic matmul in ONE program: split ic into T chunks, one matmul
// per chunk, sum partials. Every matmul sees K<=chunk -> fast-rate tiling.
static std::string gen_dyn_mm_ktiled_mil(int ic, int oc, int sq, int T) {
    std::string out;
    char L[768];
    std::snprintf(L, 768, "program(1.3)\n%s\n{\n", kBuildInfo); out += L;
    std::snprintf(L, 768, "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", ic, sq + oc); out += L;
    // shared consts first
    out += "    tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n";
    out += "    bool bF = const()[name=string(\"bF\"), val=bool(false)];\n";
    std::vector<int> ks(T, ic / T);
    for (int r = 0; r < ic % T; ++r) ks[r] += 1;
    int off = 0;
    for (int t = 0; t < T; ++t) {
        const int kc = ks[t];
        char n[16]; std::snprintf(n, 16, "t%d", t);
        std::snprintf(L, 768, "    tensor<int32, [4]> %s_ba = const()[name=string(\"%s_ba\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", n, n, off); out += L;
        std::snprintf(L, 768, "    tensor<int32, [4]> %s_sa = const()[name=string(\"%s_sa\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", n, n, kc, sq); out += L;
        std::snprintf(L, 768, "    tensor<fp16, [1,%d,1,%d]> %s_act = slice_by_size(x=x,begin=%s_ba,size=%s_sa)[name=string(\"%s_act\")];\n", kc, sq, n, n, n, n, n); out += L;
        std::snprintf(L, 768, "    tensor<int32, [4]> %s_bw = const()[name=string(\"%s_bw\"), val=tensor<int32, [4]>([0,%d,0,%d])];\n", n, n, off, sq); out += L;
        std::snprintf(L, 768, "    tensor<int32, [4]> %s_sw = const()[name=string(\"%s_sw\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", n, n, kc, oc); out += L;
        std::snprintf(L, 768, "    tensor<fp16, [1,%d,1,%d]> %s_wt = slice_by_size(x=x,begin=%s_bw,size=%s_sw)[name=string(\"%s_wt\")];\n", kc, oc, n, n, n, n, n); out += L;
        std::snprintf(L, 768, "    tensor<int32, [4]> %s_ra = const()[name=string(\"%s_ra\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", n, n, kc, sq); out += L;
        std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> %s_a2 = reshape(shape=%s_ra,x=%s_act)[name=string(\"%s_a2\")];\n", kc, sq, n, n, n, n, n); out += L;
        std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> %s_a3 = transpose(perm=pm,x=%s_a2)[name=string(\"%s_a3\")];\n", sq, kc, n, n, n); out += L;
        std::snprintf(L, 768, "    tensor<int32, [4]> %s_rw = const()[name=string(\"%s_rw\"), val=tensor<int32, [4]>([1,1,%d,%d])];\n", n, n, kc, oc); out += L;
        std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> %s_W = reshape(shape=%s_rw,x=%s_wt)[name=string(\"%s_W\")];\n", kc, oc, n, n, n, n, n); out += L;
        std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> %s_p = matmul(transpose_x=bF,transpose_y=bF,x=%s_a3,y=%s_W)[name=string(\"%s_p\")];\n", sq, oc, n, n, n, n, n); out += L;
        if (t == 0) {
            std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> acc = matmul(transpose_x=bF,transpose_y=bF,x=%s_a3,y=%s_W)[name=string(\"acc\")];\n", sq, oc, n, n); out += L;
        } else {
            char pp[16];
            if (t == 1) std::snprintf(pp, 16, "acc"); else std::snprintf(pp, 16, "acc%d", t - 1);
            std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> acc%d = add(x=%s,y=%s_p)[name=string(\"acc%d\")];\n", sq, oc, t, pp, n, t); out += L;
        }
        off += kc;
    }
    std::snprintf(L, 768, "    tensor<fp16, [1,1,%d,%d]> yt = transpose(perm=pm,x=acc%d)[name=string(\"yt\")];\n", sq, oc, T - 1); out += L;
    std::snprintf(L, 768, "    tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", oc, sq); out += L;
    std::snprintf(L, 768, "    tensor<fp16, [1,%d,1,%d]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", oc, sq); out += L;
    out += "  } -> (y);\n}\n";
    return out;
}

class Surf {
public:
    Surf(size_t bytes) : s_(metal_create_iosurface(bytes)) {}
    ~Surf() { if (s_) CFRelease(s_); }
    IOSurfaceRef get() const { return s_; }
private:
    IOSurfaceRef s_;
};

static void fill(IOSurfaceRef s, const std::vector<uint16_t>& v) {
    IOSurfaceLock(s, 0, nullptr);
    std::memcpy(IOSurfaceGetBaseAddress(s), v.data(), v.size() * 2);
    IOSurfaceUnlock(s, 0, nullptr);
}
static std::vector<uint16_t> readv(IOSurfaceRef s, size_t n) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, nullptr);
    std::vector<uint16_t> v(n);
    std::memcpy(v.data(), IOSurfaceGetBaseAddress(s), n * 2);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, nullptr);
    return v;
}

int main(int argc, char** argv) {
    const int ic = getenv("IC") ? std::atoi(getenv("IC")) : 5120;
    const int oc = getenv("OC") ? std::atoi(getenv("OC")) : ic;
    int Ss[] = {getenv("S0") ? std::atoi(getenv("S0")) : 32,
                getenv("S0") ? std::atoi(getenv("S0")) : 128,
                getenv("S0") ? std::atoi(getenv("S0")) : 512};
    if (!getenv("S0")) { Ss[1] = 128; Ss[2] = 512; }
    if (argc > 1) Ss[0] = std::atoi(argv[1]); // allow single-shape runs

    ANEContext* ane = ane_context_create();
    if (!ane) { std::cerr << "no context\n"; return 1; }

    auto lcg = [s = 20260823u](float sc) mutable {
        s = s * 1664525u + 1013904223u;
        return ((s >> 8) & 0xffff) / 65535.0f * 2.f * sc - sc;
    };
    // shared deterministic weight matrix (oc-major rows of ic)
    std::vector<uint16_t> W((size_t)ic * oc);
    for (auto& v : W) v = fh(lcg(0.05f));

    std::cout << "dyn matmul ic=oc=" << ic << " (weight-from-surface)\n";
    std::cout << std::fixed;
    for (int S : Ss) {
        std::vector<uint16_t> X((size_t)ic * S);
        for (auto& v : X) v = fh(lcg(0.3f));
        std::vector<uint16_t> surf((size_t)ic * (S + oc));
        size_t p = 0;
        for (int i = 0; i < ic; ++i) {
            for (int t = 0; t < S; ++t) surf[p++] = X[(size_t)i * S + t];
            for (int o = 0; o < oc; ++o) surf[p++] = W[(size_t)i * oc + o];
        }
        const std::string mil = gen_dyn_mm_mil(ic, oc, S);
        const bool use_conv = getenv("MM_CONV") != nullptr;
        static int ktiles = getenv("K_TILES") ? std::atoi(getenv("K_TILES")) : 1;
        std::string mil2 = mil;
        if (ktiles > 1) mil2 = gen_dyn_mm_ktiled_mil(ic, oc, S, ktiles);
        {
            char pf[128]; std::snprintf(pf, sizeof(pf), "/tmp/mm_%d_%d_%d.mil", ic, oc, S);
            FILE* f = fopen(pf, "w"); fwrite(mil2.data(), 1, mil2.size(), f); fclose(f);
            if (use_conv) std::cout << "  (conv form)\n";
        }
        const char* no_w[] = {nullptr};
        ANEModel* m = ane_model_compile_mil(ane, mil2.c_str(), no_w, nullptr, nullptr, 0, 0, 21);
        if (!m) { std::cout << "  S=" << S << ": COMPILE FAILED\n"; continue; }
        Surf in(surf.size() * 2), out((size_t)oc * S * 2);
        fill(in.get(), surf);
        ANERequest* req = ane_request_create(ane, m, in.get(), out.get(), 0);
        if (!req) { std::cout << "  S=" << S << ": REQ FAIL\n"; ane_model_release(m); continue; }

        for (int i = 0; i < 3; ++i) ane_request_evaluate(ane, m, req, nullptr, 0, nullptr, 0);
        const int iters = S >= 1024 ? 15 : 30;
        std::vector<double> ms;
        for (int i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            ane_request_evaluate(ane, m, req, nullptr, 0, nullptr, 0);
            ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        const double p50 = ms[ms.size() / 2];

        const auto got = readv(out.get(), (size_t)oc * S);
        // spot-check 32 random (o,t) cells vs CPU reference
        double worst = 0; int wo = 0, wt = 0;
        for (int c = 0; c < 32; ++c) {
            const int o = (c * 1601) % oc, t = (c * 2749) % S;
            float acc = 0.f;
            for (int i = 0; i < ic; ++i)
                acc += hf(X[(size_t)i * S + t]) * hf(W[(size_t)i * oc + o]);
            const float g = hf(got[(size_t)o * S + t]);
            const double d = std::fabs((double)g - acc) / (std::fabs((double)acc) + 1e-3);
            if (d > worst) { worst = d; wo = o; wt = t; }
        }
        const double tflops = 2.0 * ic * oc * S / (p50 * 1e-3) / 1e12;
        std::cout << std::setprecision(1)
                  << "  S=" << std::setw(4) << S << " p50=" << std::setprecision(3) << p50
                  << " ms  " << std::setprecision(2) << tflops << " TFLOPS"
                  << "  worst_relerr=" << std::setprecision(1) << worst
                  << " @" << wo << "," << wt << "\n";
        ane_request_release(req);
        ane_model_release(m);
    }
    ane_context_destroy(ane);
    return 0;
}
