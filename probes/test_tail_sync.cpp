// Isolated experiment: compile ONE fused tail, feed fixed inputs, and probe
// (a) evaluate determinism, (b) sync-vs-async completion, (c) lane-count
// invariance. No server may be running (ANE program budget).
#include "runtime/rindi_native_chain.h"
#include "runtime/safetensors_loader.h"
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using clock_ = std::chrono::high_resolution_clock;

static unsigned long long hash_range(const uint16_t* p, size_t n, int cols, int seq) {
    unsigned long long h = 1469598103934665603ull;
    for (size_t c = 0; c < n; ++c)
        for (int s = 0; s < cols; ++s) { h ^= p[c * seq + s]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1]
        : "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    SafeTensorsLoader loader;
    if (!loader.open_file(std::string(model) + "/gpu_backbone.safetensors")) return 2;
    RindiNativeChain chain(5120, 32);
    ANEContext* ane = ane_context_create();
    if (!chain.compile_layer(3, std::string(model) + "/ane_layers", loader)) {
        std::fprintf(stderr, "compile failed\n");
        return 2;
    }
    // Drive through the public batch API with fixed pseudo-random core/residual.
    const size_t core_dim = 6144, H = 5120;
    auto fill = [](std::vector<uint16_t>& v, unsigned seed) {
        for (size_t i = 0; i < v.size(); ++i) {
            seed = seed * 1103515245u + 12345u;
            v[i] = static_cast<uint16_t>((seed >> 16) | 0x3800);
        }
    };
    std::vector<std::vector<uint16_t>> lane0_of;
    const size_t lane_widths[] = {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 17u};
    for (size_t lanes : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 17u}) {
        std::vector<uint16_t> core(core_dim * lanes), res(H * lanes), out;
        fill(core, 7u); fill(res, 9u);
        chain.evaluate_tail_batch(3, core.data(), core_dim, res.data(), lanes, out, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::vector<uint16_t> out2;
        chain.evaluate_tail_batch(3, core.data(), core_dim, res.data(), lanes, out2, nullptr);
        size_t diff01 = 0;
        for (size_t i = 0; i < out.size(); ++i) if (out[i] != out2[i]) ++diff01;
        // lane-0 column extraction: channel-major [C, lanes]
        std::vector<uint16_t> lane0(out.size() / lanes);
        for (size_t c = 0; c < lane0.size(); ++c) lane0[c] = out[c * lanes];
        lane0_of.push_back(lane0);
        printf("L%zu det=%zu", lanes, diff01);
        std::printf("lanes=%2zu det_diff(0v1)=%zu elems=%zu\n", lanes, diff01, out.size());
    }
    const size_t widths[] = {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 17u};
    for (size_t i = 1; i < lane0_of.size(); ++i) {
        size_t bad = 0;
        for (size_t c = 0; c < lane0_of[i].size(); ++c)
            if (lane0_of[i][c] != lane0_of[0][c]) ++bad;
        std::printf("lanes=%zu vs lanes=1: %zu/%zu differ\n",
                    widths[i], bad, lane0_of[i].size());
    }
    ane_context_destroy(ane);
    return 0;
}
