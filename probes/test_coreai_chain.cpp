// test_coreai_chain.cpp — validates the RindiNativeChain CoreAI backend
// end-to-end: compile_layer() loads the exported bundle via the Swift shim,
// evaluate_tail_batch() does the channel-major<->token-major conversion, and
// results are checked against golden vectors dumped at export time.
//
//   usage: test_coreai_chain <layer> [lanes]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cstring>
#include "rindi_native_chain.h"

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t man  = (h & 0x03FF);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; }
               man &= 0x3FF; bits = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out; memcpy(&out, &bits, 4); return out;
}

static std::vector<uint16_t> load_f16(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint16_t> v(sz / 2);
    if (fread(v.data(), 2, v.size(), f) != v.size()) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    setenv("RINDI_TAIL_COREAI", "1", 1);   // route compile_layer to bundles
    const int layer = argc > 1 ? atoi(argv[1]) : 3;
    const size_t lanes = argc > 2 ? (size_t)atol(argv[2]) : 32;
    const size_t H = 5120, C = 6144;

    if (getenv("RINDI_DIRECT_FIRST")) {
        auto xg = load_f16("/tmp/golden_L3_x.f16");
        auto yg = load_f16("/tmp/golden_L3_y.f16");
        std::string home = getenv("HOME");
        void* h0 = rindi_ane_load((home +
            "/.rindi/aimodels/qwen38_27b_tail_L3_ip1_int4_g32_tm_s32.aimodel").c_str(), 1);
        std::vector<uint16_t> dout(32 * (5120 + 16480));
        rindi_ane_run(h0, xg.data(), 32, 11264, dout.data(), (long)dout.size());
        double md = 0;
        for (size_t i = 0; i < yg.size(); ++i)
            md = std::max(md, (double)std::abs((float)dout[i] - (float)yg[i]));
        printf("[pre-ctor-direct] max|dy|=%.4f rel=%.5f\n", md, md / 64.5);
        rindi_ane_free(h0);
    }

    RindiNativeChain chain(H, 32);
    if (!chain.ane_context()) { fprintf(stderr, "no ANE ctx\n"); return 1; }
    SafeTensorsLoader dummy;   // unused by coreai path
    if (!chain.compile_layer(layer, "/nonexistent", dummy)) {
        fprintf(stderr, "compile_layer(%d) failed\n", layer);
        return 2;
    }
    printf("[chain] L%d bundle loaded\n", layer);

    auto x = load_f16("/tmp/golden_L" + std::to_string(layer) + "_x.f16");
    auto yr = load_f16("/tmp/golden_L" + std::to_string(layer) + "_y.f16");
    auto y2r = load_f16("/tmp/golden_L" + std::to_string(layer) + "_y2.f16");

    // split golden token-major x into engine-layout channel-major buffers:
    // xin row s = [core[0..C), res[0..H)] -> core[c*lanes+s], res[c*lanes+s]
    std::vector<uint16_t> core(C * lanes), res(H * lanes);
    for (size_t c = 0; c < C; ++c)
        for (size_t s = 0; s < lanes; ++s) core[c * lanes + s] = x[s * (C + H) + c];
    for (size_t c = 0; c < H; ++c)
        for (size_t s = 0; s < lanes; ++s) res[c * lanes + s] = x[s * (C + H) + C + c];

    if (std::getenv("RINDI_DEBUG_COREAI_STAGING"))
        std::fprintf(stderr, "[probe-golden]   x[0..3]=%04x %04x %04x %04x "
                             "x[C..C+3]=%04x %04x %04x %04x\n",
                     x[0], x[1], x[2], x[3], x[C], x[C+1], x[C+2], x[C+3]);
    // A: DIRECT shim call on raw golden x through a fresh load
    {
        std::string home = getenv("HOME");
        void* h2 = rindi_ane_load((home +
            "/.rindi/aimodels/qwen38_27b_tail_L3_ip1_int4_g32_tm_s32.aimodel").c_str(), 1);
        if (!h2) { fprintf(stderr, "direct load failed\n"); return 7; }
        std::vector<uint16_t> dout(32 * (5120 + 16480));
        long n = rindi_ane_run(h2, x.data(), 32, 11264, dout.data(),
                               (long)dout.size());
        double md = 0;
        for (size_t i = 0; i < yr.size(); ++i)
            md = std::max(md, (double)std::abs((float)dout[i] - (float)yr[i]));
        printf("[chain-direct] n=%ld max|dy|=%.4f (ref|max|~64 => rel=%.5f)\n",
               n, md, md / 64.5);
        rindi_ane_free(h2);
    }
    std::vector<uint16_t> out, np;
    if (!chain.evaluate_tail_batch(layer, core.data(), C, res.data(), lanes,
                                   out, &np)) {
        fprintf(stderr, "evaluate_tail_batch failed\n");
        return 3;
    }
    if (out.size() != H * lanes || np.size() != y2r.size()) {
        fprintf(stderr, "size mismatch out=%zu np=%zu want y2=%zu\n",
                out.size(), np.size(), y2r.size());
        return 4;
    }

    double max_y = 0, max_y2 = 0, ref_y = 0, ref_y2 = 0;
    const size_t P = np.size() / lanes;
    for (size_t i = 0; i < out.size(); ++i) {
        // engine output is channel-major [c*lanes+s]; golden is token-major
        size_t c = i / lanes, s2 = i % lanes;
        max_y = std::max(max_y, (double)std::abs(f16_to_f32(out[c * lanes + s2]) -
                                                 f16_to_f32(yr[s2 * H + c])));
        ref_y = std::max(ref_y, (double)std::abs(f16_to_f32(yr[s2 * H + c])));
    }
    for (size_t i = 0; i < np.size(); ++i) {
        size_t c = i / lanes, s2 = i % lanes;
        max_y2 = std::max(max_y2, (double)std::abs(f16_to_f32(np[c * lanes + s2]) -
                                                   f16_to_f32(y2r[s2 * P + c])));
        ref_y2 = std::max(ref_y2, (double)std::abs(f16_to_f32(y2r[s2 * P + c])));
    }
    double rel_y = max_y / ref_y, rel_y2 = max_y2 / ref_y2;
    printf("[chain] out[0..4]=%04x %04x %04x %04x %04x gold=%04x %04x %04x %04x %04x\n",
           out[0], out[1], out[2], out[3], out[4],
           yr[0], yr[1], yr[2], yr[3], yr[4]);
    printf("[chain] rel(y)=%.5f rel(y2)=%.5f (%s)\n", rel_y, rel_y2,
           (rel_y < 0.05 && rel_y2 < 0.05) ? "MATCH" : "MISMATCH");

    // timing
    using clk = std::chrono::steady_clock;
    for (int w = 0; w < 10; ++w)
        chain.evaluate_tail_batch(layer, core.data(), C, res.data(), lanes, out, &np);
    auto t0 = clk::now();
    for (int i = 0; i < 100; ++i)
        chain.evaluate_tail_batch(layer, core.data(), C, res.data(), lanes, out, &np);
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / 100;
    printf("[chain] %.3f ms/eval incl. layout conversion (%.0f tok/s @S=%zu)\n",
           ms, lanes / ms * 1000, lanes);
    return (rel_y < 0.05 && rel_y2 < 0.05) ? 0 : 5;
}
