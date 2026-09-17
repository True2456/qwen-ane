// Determinative probe: is the GDN layer state COMPOSITIONAL across calls?
// (i.e., processing token-chunk T=[t0..t3] in ONE wide call leaves the same
// conv history + recurrence state as processing it as several narrower calls
// with the same prefix in order). If state depends on batch width, MTP's
// batched-verify can never be exact against sequential decode, and P6's
// per-layer rebuild is also unsound. If it IS compositional, P6 just has a
// different bug worth chasing.
//
// Compares recurrence state + conv history bytes.
#include "runtime/rindi_gdn_layer.h"
#include "runtime/safetensors_loader.h"
#include "runtime/ane_c_bridge.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

// deterministic pseudo-random
static uint32_t rngstate = 0x1234;
static uint16_t hrnd(float amp = 0.1f) {
    rngstate = rngstate * 1664525u + 1013904223u;
    float v = amp * 2.0f * (float)(rngstate >> 8) / (float)0xffffff - amp;
    uint32_t sgn = (rngstate & 0xffff0000) & 0x80000000u;
    uint32_t man = (rngstate >> 13) & 0x3ffu;
    uint32_t exp = 112 + (rngstate & 0xfu);
    uint32_t bits = sgn | (exp << 23) | man;
    return (uint16_t)(bits >> 16);
}

static unsigned long long hashstate(const std::vector<uint16_t>& a,
                                    const std::vector<uint16_t>& b) {
    unsigned long long h = 1469598103934665603ull;
    for (auto x : a) { h ^= x; h *= 1099511628211ull; }
    for (auto x : b) { h ^= x; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1]
        : "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    SafeTensorsLoader loader;
    if (!loader.open_file(std::string(model) + "/gpu_backbone.safetensors")) {
        std::fprintf(stderr, "open failed\n"); return 2;
    }
    ANEContext* ane = ane_context_create();
    if (!ane) { std::fprintf(stderr, "no ane\n"); return 2; }

    constexpr size_t H = 48, D = 128, V = 128, HK = H * D;
    constexpr size_t QKV = 10240, Z = 6144, G = 48;
    constexpr size_t NPC = QKV + Z + 2 * G;

    RindiGdnLayer layer;
    // layer 1 is a GDN (linear) layer in Qwen3.8.
    if (!layer.compile_core(ane, loader, 1, 64)) {
        std::fprintf(stderr, "compile_core failed\n"); return 2;
    }

    // Build deterministic projected inputs for lanes 0..3.
    const size_t T = 4;
    std::vector<uint16_t> np(NPC * T);
    for (size_t c = 0; c < NPC; ++c)
        for (size_t t = 0; t < T; ++t)
            np[c * T + t] = hrnd();


    // Helper: projected view over the first `lanes` lanes of a contig base.
    auto run = [&](size_t lanes, const uint16_t* base,
                   std::vector<uint16_t>& conv, std::vector<uint16_t>& rec) {
        RindiGdnProjectionView pv{base, base + QKV * lanes,
                                  base + (QKV + Z) * lanes,
                                  base + (QKV + Z + G) * lanes};
        std::vector<uint16_t> core, z;
        if (!layer.core_from_projected_view(pv, lanes, core, z)) return false;
        layer.snapshot_state(conv, rec);
        return true;
    };

    // Build lane-major helper: create a compact [NPC x lanes] for chosen lanes.
    // We'll make lane-layout where base[j] = np column j for that lane.
    std::vector<uint16_t> base4(NPC * 4), base3(NPC * 3);
    // np is already [c][t] with t=0..3; baseX lane stride = X channels.
    // Fill base4[c*4 + t] = np[c*4 + t] (same layout). For base3 take t in {0,1,2}.
    for (size_t c = 0; c < NPC; ++c)
        for (size_t t = 0; t < 4; ++t) base4[c * 4 + t] = np[c * 4 + t];
    for (size_t c = 0; c < NPC; ++c)
        for (size_t t = 0; t < 3; ++t) base3[c * 3 + t] = np[c * 4 + t];

    // --- A: one wide call, lanes=4 ---
    layer.reset();
    std::vector<uint16_t> a_conv, a_rec;
    run(4, base4.data(), a_conv, a_rec);

    // --- B: chunked 3+1 ---
    layer.reset();
    std::vector<uint16_t> b_conv, b_rec;
    // feed lanes 0..2, then lane 3 (from base4 layout, lane 3 = column 3)
    RindiGdnProjectionView pv3{base4.data(), base4.data() + QKV * 4,
                               base4.data() + (QKV + Z) * 4,
                               base4.data() + (QKV + Z + G) * 4};
    // NOTE: for lanes=3 we must pack a 3-wide base, not index the 4-wide as 3.
    RindiGdnProjectionView pv3p{base3.data(), base3.data() + QKV * 3,
                                base3.data() + (QKV + Z) * 3,
                                base3.data() + (QKV + Z + G) * 3};
    std::vector<uint16_t> core, z;
    if (!layer.core_from_projected_view(pv3p, 3, core, z)) { std::fprintf(stderr,"3-call fail\n"); return 2; }
    // now feed lane 3 alone from base4 column 3
    std::vector<uint16_t> lone(NPC);
    for (size_t c = 0; c < NPC; ++c) lone[c] = base4[c * 4 + 3];
    RindiGdnProjectionView pv1{lone.data(), lone.data() + QKV,
                               lone.data() + (QKV + Z), lone.data() + (QKV + Z + G)};
    if (!layer.core_from_projected_view(pv1, 1, core, z)) { std::fprintf(stderr,"1-call fail\n"); return 2; }
    layer.snapshot_state(b_conv, b_rec);

    // --- C: chunked 2+2 ---
    layer.reset();
    std::vector<uint16_t> c_conv, c_rec;
    std::vector<uint16_t> two0(NPC*2), two1(NPC*2);
    for (size_t c = 0; c < NPC; ++c) { two0[c*2+0]=base4[c*4+0]; two0[c*2+1]=base4[c*4+1]; two1[c*2+0]=base4[c*4+2]; two1[c*2+1]=base4[c*4+3]; }
    run(2, two0.data(), c_conv, c_rec);
    run(2, two1.data(), c_conv, c_rec);

    std::printf("BASE conicone one-wide(lanes=4)  conv_hash=%llx rec_hash=%llx\n",
                hashstate(a_conv,{}), hashstate(a_rec,{}));
    std::printf("CHUNK (3+1)                     conv_hash=%llx rec_hash=%llx\n",
                hashstate(b_conv,{}), hashstate(b_rec,{}));
    std::printf("CHUNK (2+2)                     conv_hash=%llx rec_hash=%llx\n",
                hashstate(c_conv,{}), hashstate(c_rec,{}));
    bool conv_ok = (hashstate(a_conv,{}) == hashstate(b_conv,{})) && (hashstate(a_conv,{}) == hashstate(c_conv,{}))
        && (hashstate(a_rec,{}) == hashstate(b_rec,{})) && (hashstate(a_rec,{}) == hashstate(c_rec,{}));
    std::printf("COMPOSITIONAL (one-wide == chunked) : %s\n", conv_ok ? "YES" : "NO");
    return 0;
}