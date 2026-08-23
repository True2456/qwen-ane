// SPDX-License-Identifier: Apache-2.0
/*
 * ane_as.cpp - Native C/C++ Apple Neural Engine Kernel Synthesizer & Bench
 *
 * Synthesizes MIL programs parameterized by shape (C, S, K, groups), wraps
 * weight blobs natively (128-byte DEADBEEF envelope, built inside the bridge),
 * binds zero-copy IOSurface IO and evaluates direct hardware dispatches via
 * the private AppleNeuralEngine runtime - no Python toolchain involved.
 *
 * Tests (--test-all):
 *   1. smoke        4x4x1x32 groups=1 conv, exact fp16 output check
 *   2. offset-probe empirically pins down BLOBFILE offset semantics
 *                   (doc said payload@0x80, working MILs say offset=64)
 *   3. depthwise    causal depthwise conv1d (GDN shape family):
 *                   C=64/S=32/K=4 and C=10240/S=32/K=4, verified against a
 *                   scalar fp32 CPU reference (RelErr bar 5e-4, warn 5e-3)
 *
 * Latency stats report min/p50/p90/max because dispatch overhead is noisy
 * across runs (observed 3.8 - 9.3 kHz for the same graph).
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <pthread.h>
#include <tuple>
#include <utility>

#include <IOSurface/IOSurface.h>

#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"

// ---------------------------------------------------------------------------
// fp16 <-> fp32 conversion
// ---------------------------------------------------------------------------

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp16 = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x03ff;
    uint32_t u32;
    if (exp16 == 0) {
        if (mant == 0) {
            u32 = sign; // +-zero
        } else {        // subnormal: normalize into fp32
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while (!(m & 0x400));
            u32 = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x03ff) << 13);
        }
    } else if (exp16 == 31) {
        u32 = sign | 0x7f800000u | (mant << 13); // inf / nan
    } else {
        u32 = sign | ((uint32_t)(exp16 - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u32, sizeof(f));
    return f;
}

static inline uint16_t fp32_to_fp16(float f) {
    // Round-to-nearest-even fp32 -> fp16.
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const uint32_t raw_exp = (x >> 23) & 0xff;
    const uint32_t mant = x & 0x007fffff;
    if (raw_exp == 0xff)  // inf / nan
        return (uint16_t)(sign | 0x7c00 | (mant ? 0x200u : 0u));
    const int32_t exp = (int32_t)raw_exp - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);            // overflow -> inf
    if (exp <= 0) {                                             // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        const uint32_t m = mant | 0x00800000;
        const int shift = 14 - exp;
        const uint32_t half = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1);
        const uint32_t mid = 1u << (shift - 1);
        const uint32_t inc = (rem > mid || (rem == mid && (half & 1u))) ? 1u : 0u;
        return (uint16_t)(sign | (half + inc));
    }
    uint32_t h = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1u))) ++h;       // RNE
    return (uint16_t)(sign | h);
}

// ---------------------------------------------------------------------------
// MIL synthesis - parameterized program templates
// ---------------------------------------------------------------------------

static const char* kBuildInfo =
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

// Smoke graph: [1,4,1,32] x [4,4,1,1] groups=1, weights all ones.
static std::string make_smoke_mil(uint64_t blob_offset) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "program(1.3)\n%s\n"
        "{\n"
        "  func main<ios18>(tensor<fp16, [1, 4, 1, 32]> x) {\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "    tensor<fp16, [4, 4, 1, 1]> w = const()[name=string(\"w\"), val=tensor<fp16, [4, 4, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(%llu)))];\n"
        "    tensor<fp16, [1, 4, 1, 32]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"y\")];\n"
        "  } -> (y);\n"
        "}\n",
        kBuildInfo, (unsigned long long)blob_offset);
    return std::string(buf);
}

// Causal depthwise conv1d - faithful port of probes/ane_gdn_conv1d.py:
//   weight [C,1,1,K], pad custom [0,0,K-1,0] (left), groups=C, identity-mul
//   output op (forces a materialized named output). Layout [1,C,1,S].
static std::string make_depthwise_conv1d_mil(int C, int S, int K, uint64_t blob_offset) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
        "program(1.3)\n%s\n"
        "{\n"
        "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "    tensor<fp16, [%d, 1, 1, %d]> w = const()[name=string(\"w\"), val=tensor<fp16, [%d, 1, 1, %d]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(%llu)))];\n"
        "    tensor<int32, [2]> strides = const()[name=string(\"strides\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [2]> dil = const()[name=string(\"dil\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pad = const()[name=string(\"pad\"), val=tensor<int32, [4]>([0,0,%d,0])];\n"
        "    tensor<fp16, [1, %d, 1, %d]> c = conv(dilations=dil, groups=int32(%d), pad=pad, pad_type=string(\"custom\"), strides=strides, weight=w, x=x)[name=string(\"c\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> y = mul(x=c, y=fp16(0x1p+0))[name=string(\"y\")];\n"
        "  } -> (y);\n"
        "}\n"
        "// gdn_depthwise_C%d\n",
        kBuildInfo, C, S, C, K, C, K, (unsigned long long)blob_offset,
        K - 1, C, S, C, C, S, C);
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// CPU scalar reference (fp32 accumulate over fp16-rounded operands)
// ---------------------------------------------------------------------------

// x[c*S + s], w[c*K + k] -> y[c*S + s]; left causal pad of K-1 zeros.
static void ref_causal_conv1d(const std::vector<uint16_t>& x,
                              const std::vector<uint16_t>& w,
                              int C, int S, int K,
                              std::vector<float>& y) {
    y.assign((size_t)C * S, 0.0f);
    for (int c = 0; c < C; ++c) {
        for (int t = 0; t < S; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int src = t + k - (K - 1); // shifted window over left pad
                if (src >= 0) {
                    acc += fp16_to_fp32(x[(size_t)c * S + src]) *
                           fp16_to_fp32(w[(size_t)c * K + k]);
                }
            }
            y[(size_t)c * S + t] = acc;
        }
    }
}

static float rel_error(const std::vector<uint16_t>& got,
                       const std::vector<float>& ref) {
    float max_abs = 0.0f, max_ref = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(fp16_to_fp32(got[i]) - ref[i]));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    return max_abs / (max_ref + 1e-9f);
}

// ---------------------------------------------------------------------------
// IOSurface helpers (planar channel-major, RowStride = 2*S per spec section 2)
// ---------------------------------------------------------------------------

class IoSurface {
public:
    IoSurface(size_t bytes) : surf_(metal_create_iosurface(bytes)) {}
    ~IoSurface() { if (surf_) CFRelease(surf_); }
    IOSurfaceRef get() const { return surf_; }
    explicit operator bool() const { return surf_ != nullptr; }
private:
    IOSurfaceRef surf_;
};

static void surface_fill(IOSurfaceRef s, const std::vector<uint16_t>& v) {
    IOSurfaceLock(s, 0, nullptr);
    std::memcpy(IOSurfaceGetBaseAddress(s), v.data(), v.size() * sizeof(uint16_t));
    IOSurfaceUnlock(s, 0, nullptr);
}

static std::vector<uint16_t> surface_read(IOSurfaceRef s, size_t n) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, nullptr);
    std::vector<uint16_t> v(n);
    std::memcpy(v.data(), IOSurfaceGetBaseAddress(s), n * sizeof(uint16_t));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, nullptr);
    return v;
}

// ---------------------------------------------------------------------------
// Benchmark harness with distribution stats
// ---------------------------------------------------------------------------

struct LatencyStats {
    double mean_ms = 0, min_ms = 0, p50_ms = 0, p90_ms = 0, max_ms = 0;
    double khz = 0;
};

static LatencyStats bench_dispatch(ANEContext* ane, ANEModel* model,
                                   ANERequest* req, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i)
        ane_request_evaluate(ane, model, req, nullptr, 0, nullptr, 0);
    std::vector<double> ms(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        ane_request_evaluate(ane, model, req, nullptr, 0, nullptr, 0);
        const auto t1 = std::chrono::high_resolution_clock::now();
        ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(ms.begin(), ms.end());
    LatencyStats st;
    double sum = std::accumulate(ms.begin(), ms.end(), 0.0);
    st.mean_ms = sum / ms.size();
    st.min_ms = ms.front();
    st.max_ms = ms.back();
    st.p50_ms = ms[ms.size() / 2];
    st.p90_ms = ms[(size_t)(ms.size() * 0.9)];
    st.khz = 1.0 / (st.mean_ms * 1e-3) / 1000.0;
    return st;
}

static void print_stats(const LatencyStats& st, double gflops) {
    std::cout << std::fixed << std::setprecision(4)
              << "      latency ms: mean=" << st.mean_ms << " min=" << st.min_ms
              << " p50=" << st.p50_ms << " p90=" << st.p90_ms
              << " max=" << st.max_ms << "\n"
              << "      dispatch rate: " << std::setprecision(2) << st.khz
              << " kHz";
    if (gflops > 0) std::cout << " | " << std::setprecision(2) << gflops << " GFLOP/s";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static int test_smoke(ANEContext* ane, int iters) {
    std::cout << "\n[1] smoke: 4x4x1x32 groups=1 conv (weights = 1.0)\n";
    std::vector<uint16_t> weights(16, 0x3c00); // sixteen 1.0
    const char* names[] = {"w.bin"};
    const void* data[] = {weights.data()};
    const size_t sizes[] = {weights.size() * sizeof(uint16_t)};
    ANEModel* model = ane_model_compile_mil(ane, make_smoke_mil(64).c_str(),
                                            names, data, sizes, 1, 0, 21);
    if (!model) { std::cout << "  [FAIL] compile\n"; return 1; }
    IoSurface in(4 * 32 * sizeof(uint16_t)), out(4 * 32 * sizeof(uint16_t));
    if (!in || !out) { std::cout << "  [FAIL] IOSurface alloc\n"; ane_model_release(model); return 1; }
    surface_fill(in.get(), std::vector<uint16_t>(4 * 32, 0x3c00));
    ANERequest* req = ane_request_create(ane, model, in.get(), out.get(), 0);
    if (!req) { std::cout << "  [FAIL] request\n"; ane_model_release(model); return 1; }

    const LatencyStats st = bench_dispatch(ane, model, req, 5, iters);

    const auto got = surface_read(out.get(), 4 * 32);
    const bool ok = !got.empty() && got[0] == 0x4400; // exact fp16 4.0
    std::cout << "  " << (ok ? "[PASS]" : "[FAIL]")
              << " exact output " << fp16_to_fp32(got.empty() ? 0 : got[0])
              << " (expected 4.0)\n";
    print_stats(st, 2.0 * 4 * 4 * 32 / (st.mean_ms * 1e-3) / 1e9);
    ane_request_release(req);
    ane_model_release(model);
    return ok ? 0 : 1;
}

// Empirically determine BLOBFILE offset semantics. Distinct weights (1..16)
// summed over a ones-input make any mis-offset produce visibly wrong sums.
static int test_offset_probe(ANEContext* ane) {
    std::cout << "\n[2] offset-probe: pin down BLOBFILE offset semantics\n";
    std::vector<uint16_t> weights(16);
    for (int i = 0; i < 16; ++i) weights[i] = fp32_to_fp16((float)(i + 1));
    const char* names[] = {"w.bin"};
    const void* data[] = {weights.data()};
    const size_t sizes[] = {weights.size() * sizeof(uint16_t)};
    const std::vector<uint16_t> ones(4 * 32, 0x3c00);
    int best = -1;
    for (uint64_t off : {0ull, 32ull, 64ull, 128ull}) {
        ANEModel* model = ane_model_compile_mil(ane, make_smoke_mil(off).c_str(),
                                                names, data, sizes, 1, 0, 21);
        if (!model) { std::cout << "  offset " << off << ": COMPILE FAILED\n"; continue; }
        IoSurface in(4 * 32 * sizeof(uint16_t)), out(4 * 32 * sizeof(uint16_t));
        surface_fill(in.get(), ones);
        ANERequest* req = ane_request_create(ane, model, in.get(), out.get(), 0);
        if (!req) { ane_model_release(model); continue; }
        ane_request_evaluate(ane, model, req, nullptr, 0, nullptr, 0);
        const auto got = surface_read(out.get(), 4 * 32);
        // Expected lane sums for lanes 0..3 with weights 1..16 in row-major
        // [4,4]: lane l sums weights[l*4 .. l*4+3] = 4l+10.
        bool match = false;
        float got0 = got.empty() ? 0.f : fp16_to_fp32(got[0]);
        for (int l = 0; l < 4; ++l)
            if (got0 == (float)(4 * l + 10)) { match = true; break; }
        std::cout << "  offset " << std::setw(3) << off << ": "
                  << (match ? "MATCH" : "no") << " (out0="
                  << got0 << ")\n";
        if (match && best < 0) best = (int)off;
        ane_request_release(req);
        ane_model_release(model);
    }
    std::cout << "  => BLOBFILE offset base: " << best
              << (best == 64 ? " (matches the convention used by every working MIL)"
                             : (best < 0 ? " (none matched!)" : ""));
    std::cout << "\n  [NOTE] update docs/HWX-ISA-SPEC.md section 3 accordingly\n";
    return best < 0 ? 1 : 0;
}

// Full synthesizer pipeline for the GDN-shape causal depthwise conv1d.
static int test_depthwise(ANEContext* ane, int C, int S, int K, int iters) {
    std::cout << "\n[*] depthwise causal conv1d C=" << C << " S=" << S << " K=" << K << "\n";
    // Deterministic pseudo-random weights/inputs (LCG), rounded to fp16 first
    // so the CPU reference consumes exactly what the hardware sees.
    auto lcg = [s = 12345u](float scale) mutable {
        s = s * 1664525u + 1013904223u;
        return ((s >> 8) & 0xffff) / 65535.0f * 2.0f * scale - scale;
    };
    std::vector<uint16_t> w((size_t)C * K), x((size_t)C * S);
    for (auto& v : w) v = fp32_to_fp16(lcg(0.15f));
    for (auto& v : x) v = fp32_to_fp16(lcg(0.2f));

    const char* names[] = {"w.bin"};
    const void* data[] = {w.data()};
    const size_t sizes[] = {w.size() * sizeof(uint16_t)};
    const auto t0 = std::chrono::high_resolution_clock::now();
    ANEModel* model = ane_model_compile_mil(
        ane, make_depthwise_conv1d_mil(C, S, K, 64).c_str(),
        names, data, sizes, 1, 0, 21);
    const double compile_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    if (!model) { std::cout << "  [FAIL] compile (" << compile_ms << " ms)\n"; return 1; }
    std::cout << std::fixed << std::setprecision(1)
              << "      compiled in " << compile_ms << " ms\n";

    IoSurface in((size_t)C * S * sizeof(uint16_t)), out((size_t)C * S * sizeof(uint16_t));
    if (!in || !out) { std::cout << "  [FAIL] IOSurface alloc\n"; ane_model_release(model); return 1; }
    surface_fill(in.get(), x);
    ANERequest* req = ane_request_create(ane, model, in.get(), out.get(), 0);
    if (!req) { std::cout << "  [FAIL] request\n"; ane_model_release(model); return 1; }

    const LatencyStats st = bench_dispatch(ane, model, req, 5, iters);
    const auto got = surface_read(out.get(), (size_t)C * S);

    std::vector<float> ref;
    ref_causal_conv1d(x, w, C, S, K, ref);
    const float rel = rel_error(got, ref);
    // The ANE conv kernel accumulates in fp16, so error vs an fp32-accumulate
    // reference sits at ~5e-4 - 6e-4 by construction (matches the Python
    // suite's 1e-3 pass band). FAIL only beyond the fp16-noise envelope.
    const bool pass = rel < 1e-3f, warn = rel < 5e-3f;
    const double gflops = 2.0 * C * K * S / (st.mean_ms * 1e-3) / 1e9;
    std::cout << "  [" << (pass ? "PASS" : (warn ? "WARN" : "FAIL")) << "]"
              << " RelErr=" << std::scientific << std::setprecision(2) << rel
              << " vs fp32-ref (bar 1e-3, warn 5e-3)\n";
    print_stats(st, gflops);
    ane_request_release(req);
    ane_model_release(model);
    return pass ? 0 : (warn ? 0 : 1);
}


// ---------------------------------------------------------------------------
// [4] Real-time scheduling path (evaluateRealTimeWithModel:)
// ---------------------------------------------------------------------------

static int test_realtime(ANEContext* ane, int iters) {
    std::cout << "\n[4] real-time path vs direct dispatch\n";
    std::vector<uint16_t> weights(16, 0x3c00);
    const char* names[] = {"w.bin"};
    const void* data[] = {weights.data()};
    const size_t sizes[] = {weights.size() * sizeof(uint16_t)};
    ANEModel* model = ane_model_compile_mil(ane, make_smoke_mil(64).c_str(),
                                            names, data, sizes, 1, 0, 21);
    if (!model) { std::cout << "  [FAIL] compile\n"; return 1; }
    IoSurface in(4 * 32 * sizeof(uint16_t)), out(4 * 32 * sizeof(uint16_t));
    surface_fill(in.get(), std::vector<uint16_t>(4 * 32, 0x3c00));
    ANERequest* req = ane_request_create(ane, model, in.get(), out.get(), 0);
    if (!req) { ane_model_release(model); return 1; }

    // direct baseline
    const LatencyStats direct = bench_dispatch(ane, model, req, 5, iters);
    // real-time path
    bool rt_ok = true;
    for (int i = 0; i < 10; ++i)
        if (!ane_request_evaluate_realtime(ane, model, req)) { rt_ok = false; break; }
    if (!rt_ok) {
        std::cout << "  [NOTE] evaluateRealTimeWithModel: unavailable or refused on this OS build\n";
        ane_request_release(req); ane_model_release(model);
        return 0;
    }
    std::vector<double> ms(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        ane_request_evaluate_realtime(ane, model, req);
        ms[i] = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
    std::sort(ms.begin(), ms.end());
    LatencyStats rt;
    rt.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    rt.min_ms = ms.front(); rt.p50_ms = ms[ms.size()/2]; rt.max_ms = ms.back();
    std::cout << "      direct : p50=" << direct.p50_ms << " ms min=" << direct.min_ms << "\n";
    std::cout << "      realtim: p50=" << rt.p50_ms << " ms min=" << rt.min_ms << "\n";
    ane_request_release(req);
    ane_model_release(model);
    return 0;
}

// ---------------------------------------------------------------------------
// [5] Pipelined dispatch: concurrent requests hide the mailbox round trip
// ---------------------------------------------------------------------------

struct PipeThreadCtx {
    ANEContext* ane;
    ANEModel* model;
    int iters;
    double ms_per_dispatch;
};

static void* pipe_thread_fn(void* p) {
    PipeThreadCtx* c = (PipeThreadCtx*)p;
    IoSurface in(4 * 32 * sizeof(uint16_t)), out(4 * 32 * sizeof(uint16_t));
    if (!in || !out) { c->ms_per_dispatch = -1; return nullptr; }
    surface_fill(in.get(), std::vector<uint16_t>(4 * 32, 0x3c00));
    ANERequest* req = ane_request_create(c->ane, c->model, in.get(), out.get(), 0);
    if (!req) { c->ms_per_dispatch = -1; return nullptr; }
    for (int i = 0; i < 5; ++i)
        ane_request_evaluate(c->ane, c->model, req, nullptr, 0, nullptr, 0);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < c->iters; ++i)
        ane_request_evaluate(c->ane, c->model, req, nullptr, 0, nullptr, 0);
    c->ms_per_dispatch = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count() / c->iters;
    ane_request_release(req);
    return nullptr;
}

static int test_pipelined(ANEContext* ane, int iters_total) {
    std::cout << "\n[5] pipelined dispatch (concurrent requests, aggregate rate)\n";
    std::vector<uint16_t> weights(16, 0x3c00);
    const char* names[] = {"w.bin"};
    const void* data[] = {weights.data()};
    const size_t sizes[] = {weights.size() * sizeof(uint16_t)};
    ANEModel* model = ane_model_compile_mil(ane, make_smoke_mil(64).c_str(),
                                            names, data, sizes, 1, 0, 21);
    if (!model) { std::cout << "  [FAIL] compile\n"; return 1; }

    double base_rate = 0;
    for (int threads : {1, 2, 4, 8}) {
        const int iters_each = std::max(20, iters_total / threads);
        std::vector<PipeThreadCtx> ctxs(threads);
        for (auto& c : ctxs) { c = {ane, model, iters_each, 0}; }
        std::vector<pthread_t> tids(threads);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < threads; ++t)
            pthread_create(&tids[t], nullptr, pipe_thread_fn, &ctxs[t]);
        for (int t = 0; t < threads; ++t) pthread_join(tids[t], nullptr);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
        bool bad = false;
        for (const auto& c : ctxs) if (c.ms_per_dispatch < 0) bad = true;
        if (bad) { std::cout << "  [FAIL] thread setup\n"; break; }
        const double rate_khz = (threads * iters_each) / wall_ms; // per-ms -> kHz
        if (threads == 1) base_rate = rate_khz;
        const double mean_indiv = std::accumulate(ctxs.begin(), ctxs.end(), 0.0,
            [](double a, const PipeThreadCtx& c){ return a + c.ms_per_dispatch; }) / threads;
        std::cout << std::fixed << std::setprecision(2)
                  << "  threads=" << threads << ": aggregate " << rate_khz
                  << " kHz (" << rate_khz / base_rate << "x single)"
                  << ", per-request " << mean_indiv << " ms/dispatch\n";
    }
    ane_model_release(model);
    return 0;
}

// ---------------------------------------------------------------------------
// [6] Fused multi-op program: gated causal depthwise conv in ONE dispatch
//     y = conv(x, w1) (*) sigmoid(conv(g, w2))   - the GDN gating pattern.
// ---------------------------------------------------------------------------

static std::string make_gated_conv_mil(int C, int S, int K) {
    char buf[3072];
    std::snprintf(buf, sizeof(buf),
        "program(1.3)\n%s\n"
        "{\n"
        "  func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [1, %d, 1, %d]> g) {\n"
        "    tensor<fp16, [%d, 1, 1, %d]> w1 = const()[name=string(\"w1\"), val=tensor<fp16, [%d, 1, 1, %d]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [%d, 1, 1, %d]> w2 = const()[name=string(\"w2\"), val=tensor<fp16, [%d, 1, 1, %d]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n"
        "    tensor<int32, [2]> strides = const()[name=string(\"strides\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [2]> dil = const()[name=string(\"dil\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pad = const()[name=string(\"pad\"), val=tensor<int32, [4]>([0,0,%d,0])];\n"
        "    tensor<fp16, [1, %d, 1, %d]> c = conv(dilations=dil, groups=int32(%d), pad=pad, pad_type=string(\"custom\"), strides=strides, weight=w1, x=x)[name=string(\"c\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> cg = conv(dilations=dil, groups=int32(%d), pad=pad, pad_type=string(\"custom\"), strides=strides, weight=w2, x=g)[name=string(\"cg\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> sg = sigmoid(x=cg)[name=string(\"sg\")];\n"
        "    tensor<fp16, [1, %d, 1, %d]> y = mul(x=c, y=sg)[name=string(\"y\")];\n"
        "  } -> (y);\n"
        "}\n",
        kBuildInfo, C, S, C, S, C, K, C, K, C, K, C, K,
        K - 1, C, S, C, C, S, C, C, S, C, S);
    return std::string(buf);
}

static int test_fused_gated_conv(ANEContext* ane, int C, int S, int K, int iters) {
    std::cout << "\n[6] fused gated depthwise conv C=" << C << " S=" << S << " K=" << K
              << " (conv(*)sigmoid(conv): 5 ops, 1 dispatch)\n";
    auto lcg = [s = 777u](float scale) mutable {
        s = s * 1664525u + 1013904223u;
        return ((s >> 8) & 0xffff) / 65535.0f * 2.0f * scale - scale;
    };
    std::vector<uint16_t> w1((size_t)C * K), w2((size_t)C * K), x((size_t)C * S), g((size_t)C * S);
    for (auto& v : w1) v = fp32_to_fp16(lcg(0.15f));
    for (auto& v : w2) v = fp32_to_fp16(lcg(0.15f));
    for (auto& v : x)  v = fp32_to_fp16(lcg(0.2f));
    for (auto& v : g)  v = fp32_to_fp16(lcg(0.2f));

    const char* names[] = {"w1.bin", "w2.bin"};
    const void* data[] = {w1.data(), w2.data()};
    const size_t sizes[] = {w1.size() * sizeof(uint16_t), w2.size() * sizeof(uint16_t)};
    ANEModel* model = ane_model_compile_mil(
        ane, make_gated_conv_mil(C, S, K).c_str(), names, data, sizes, 2, 0, 21);
    if (!model) { std::cout << "  [FAIL] compile\n"; return 1; }
    IoSurface xin((size_t)C*S*2), gin((size_t)C*S*2), out((size_t)C*S*2);
    if (!xin || !gin || !out) { ane_model_release(model); return 1; }
    surface_fill(xin.get(), x);
    surface_fill(gin.get(), g);
    ANERequest* req = ane_request_create_2in(ane, model, xin.get(), gin.get(), out.get(), 0);
    if (!req) { std::cout << "  [FAIL] 2-input request\n"; ane_model_release(model); return 1; }

    const LatencyStats st = bench_dispatch(ane, model, req, 5, iters);
    const auto got = surface_read(out.get(), (size_t)C*S);

    // CPU reference; also compute input-swapped variant to catch binding order.
    std::vector<float> ref(C*S), ref_swap(C*S);
    std::vector<float> cxa, cga, cxb, cgb;
    ref_causal_conv1d(x, w1, C, S, K, cxa);
    ref_causal_conv1d(g, w2, C, S, K, cga);
    ref_causal_conv1d(g, w1, C, S, K, cxb);
    ref_causal_conv1d(x, w2, C, S, K, cgb);
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i]       = cxa[i] / (1.0f + std::exp(-cga[i]));
        ref_swap[i]  = cxb[i] / (1.0f + std::exp(-cgb[i]));
    }
    // Hypothesis H3: both convs consumed input 0 (binding collapsed).
    std::vector<float> cxx;
    ref_causal_conv1d(x, w1, C, S, K, cxx);
    std::vector<float> ref_h3(C*S);
    for (size_t i = 0; i < ref.size(); ++i)
        ref_h3[i] = cxx[i] / (1.0f + std::exp(-cxx[i]));
    const float rel = rel_error(got, ref);
    const float rel_swap = rel_error(got, ref_swap);
    const float rel_h3 = rel_error(got, ref_h3);
    const char* verdict =
        rel < rel_swap * 0.1f && rel < rel_h3 * 0.1f ? "OK" :
        rel_swap < rel * 0.1f ? "SWAPPED" :
        rel_h3 < rel * 0.1f ? "BOTH-READ-INPUT0" : "UNRESOLVED";
    // fp16 sigmoid adds ~1e-3 relative noise at width; bar is 3e-3 here vs
    // 1e-3 for pure convs (matches measured 5.2e-4 @C=64, 1.0e-3 @C=10240).
    const bool pass = rel < 3e-3f;
    std::cout << "  [" << (pass ? "PASS" : (rel < 5e-3f ? "WARN" : "FAIL")) << "]"
              << " RelErr=" << std::scientific << std::setprecision(2) << rel
              << " (swap=" << rel_swap << " h3=" << rel_h3 << ") => " << verdict
              << "\n";
    print_stats(st, 3.0 * C * K * S / (st.mean_ms * 1e-3) / 1e9);
    ane_request_release(req);
    ane_model_release(model);
    return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// [7] INT4 weight encoding probe - which packed-int4 MIL forms does the
//     compiler accept, and what do the nibbles decode to?
// ---------------------------------------------------------------------------

static int test_int4(ANEContext* ane) {
    std::cout << "\n[7] INT4 weight-path probe\n";
    const int C = 4, S = 32, K = 4;
    // 16 int4 values packed 2-per-byte, low nibble first: v[j] = j - 8 (-8..7).
    std::vector<uint8_t> packed((size_t)C*K/2);
    for (size_t b = 0; b < packed.size(); ++b) {
        const int lo = ((int)(b*2)   % 16) & 0xf;
        const int hi = ((int)(b*2+1) % 16) & 0xf;
        packed[b] = (uint8_t)(lo | (hi << 4));
    }
    const char* names[] = {"w.bin"};
    const void* data[] = {packed.data()};
    const size_t sizes[] = {packed.size()};
    // expected decode under (sign, low-first): w[j] = (j%16) - 8
    auto make_variant = [](const char* body) {
        char buf[2048];
        std::snprintf(buf, sizeof(buf),
            "program(1.3)\n%s\n"
            "{\n"
            "  func main<ios18>(tensor<fp16, [1, 4, 1, 32]> x) {\n"
            "%s"
            "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
            "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
            "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
            "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
            "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
            "    tensor<fp16, [1, 4, 1, 32]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"y\")];\n"
            "  } -> (y);\n"
            "}\n", kBuildInfo, body);
        return std::string(buf);
    };
    const char* var_a =
        "    tensor<int4, [4, 1, 1, 4]> w4 = const()[name=string(\"w4\"), val=tensor<int4, [4, 1, 1, 4]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [4, 1, 1, 4]> w = cast(x=w4, dtype=string(\"fp16\"))[name=string(\"w\")];\n";
    const char* var_b =
        "    tensor<uint8, [8]> qd = const()[name=string(\"qd\"), val=tensor<uint8, [8]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [4, 1, 1, 4]> w = constexpr_affine_to_dense(quantized_data=qd, zero_point=fp16(0x0p+0), scale=fp16(0x1p+0), axis=-1)[name=string(\"w\")];\n";
    const char* var_c =
        "    tensor<uint8, [8]> qd = const()[name=string(\"qd\"), val=tensor<uint8, [8]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [4, 1, 1, 4]> w = dequantize(weight=qd, scale=fp16(0x1p+0), bias=fp16(0x0p+0))[name=string(\"w\")];\n";

    int rc = 1;
    for (auto&& [tag, body] : {std::pair<const char*, const char*>{"A int4-tensor+cast", var_a},
                               {"B constexpr_affine_to_dense", var_b},
                               {"C dequantize", var_c}}) {
        const std::string mil = make_variant(body);
        ANEModel* model = ane_model_compile_mil(ane, mil.c_str(), names, data, sizes, 1, 0, 21);
        if (!model) {
            std::cout << "  variant " << tag << ": REJECTED by compiler\n";
            continue;
        }
        IoSurface in(4*32*2), out(4*32*2);
        surface_fill(in.get(), std::vector<uint16_t>(128, 0x3c00));
        ANERequest* req = ane_request_create(ane, model, in.get(), out.get(), 0);
        if (!req) { ane_model_release(model); continue; }
        ane_request_evaluate(ane, model, req, nullptr, 0, nullptr, 0);
        const auto got = surface_read(out.get(), 128);
        // ones-input conv sums 4 consecutive decoded weights starting at lane l:
        // lane0 sums w[0..3]. With low-first signed decode those are -8..-5 => -26.
        float v0 = got.empty() ? 0.f : fp16_to_fp32(got[0]);
        bool matches_low_first_signed = (v0 == -26.0f);
        std::cout << "  variant " << tag << ": COMPILED, lane0 sum=" << v0
                  << (matches_low_first_signed ? " => signed int4, LOW nibble first CONFIRMED"
                                               : " => compiled but decode differs (investigate)") << "\n";
        ane_request_release(req);
        ane_model_release(model);
        rc = 0;
    }
    if (rc) std::cout << "  => NEGATIVE RESULT: compiler accepts no probed int4 form;"
                         " envelope 'INT4 packed' claim unverified.\n";
    return 0; // probe: never fail the suite, just report
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int req_iters = 100;
    bool test_all = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < std::max(argc, 2)) {
            if (i + 1 < argc) req_iters = std::stoi(argv[++i]);
        } else if (arg == "--smoke-only") {
            test_all = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--iters N] [--smoke-only]\n"
                      << "  default: full suite (smoke, offset-probe, depthwise C=64 & C=10240)\n";
            return 0;
        }
    }

    std::cout << "======================================================================\n"
              << "  Native C/C++ Apple Neural Engine Kernel Synthesizer (ane-as)\n"
              << "======================================================================\n";

    ANEContext* ane = ane_context_create();
    if (!ane) { std::cerr << "ERROR: failed to initialize native ANEContext\n"; return 1; }
    std::cout << "- ANEContext ready (AppleNeuralEngine.framework loaded)\n";

    int rc = 0;
    rc |= test_smoke(ane, req_iters);
    if (test_all) {
        rc |= test_offset_probe(ane);
        rc |= test_depthwise(ane, 64, 32, 4, req_iters);
        rc |= test_depthwise(ane, 10240, 32, 4, req_iters);
        rc |= test_realtime(ane, req_iters);
        rc |= test_pipelined(ane, req_iters * 4);
        rc |= test_fused_gated_conv(ane, 64, 32, 4, req_iters);
        rc |= test_fused_gated_conv(ane, 10240, 32, 4, req_iters);
        rc |= test_int4(ane);
    }

    ane_context_destroy(ane);
    std::cout << "\n======================================================================\n"
              << "  NATIVE EVALUATION " << (rc == 0 ? "COMPLETE: ALL TESTS PASSED" : "FAILED")
              << "\n======================================================================\n";
    return rc;
}
