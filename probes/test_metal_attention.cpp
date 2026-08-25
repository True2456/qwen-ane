// SPDX-License-Identifier: Apache-2.0
// probes/test_metal_attention.cpp - Metal attention core vs scalar CPU oracle.
//
// Loads a real attention layer from the packaged safetensors backbone, feeds
// identical random projection outputs through both execution paths, and
// verifies the Metal scores/softmax/PV chain matches the CPU reference within
// fp16 tolerance. Runs no ANE programs, so it can execute while the server
// holds the ANE program budget.
#include "runtime/rindi_attention.h"
#include "runtime/safetensors_loader.h"
#include "runtime/ane_c_bridge.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr size_t kHidden = 5120;

std::vector<uint16_t> random_halfs(size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        // Convert through float to fp16 by truncation-free rounding via float->half.
        const float v = dist(rng);
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        const uint32_t sign = (bits >> 16) & 0x8000u;
        int e = int((bits >> 23) & 255) - 127 + 15;
        uint32_t m = (bits >> 13) & 1023u;
        if (e <= 0) out[i] = uint16_t(sign);
        else if (e >= 31) out[i] = uint16_t(sign | 0x7c00u);
        else out[i] = uint16_t(sign | (uint32_t(e) << 10) | m);
    }
    return out;
}

double diff_stats(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                  double& mean_abs) {
    double max_abs = 0.0, sum = 0.0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        auto cvt = [](uint16_t x) {
            uint32_t s = (uint32_t(x) & 0x8000u) << 16, e = (x >> 10) & 31u, m = x & 1023u, v = s;
            if (e == 0) v = s | (m << 13); else if (e == 31) v = s | 0x7f800000u | (m << 13);
            else v = s | ((e + 112u) << 23) | (m << 13);
            float f; std::memcpy(&f, &v, 4); return f;
        };
        const double d = std::abs(static_cast<double>(cvt(a[i])) - static_cast<double>(cvt(b[i])));
        max_abs = std::max(max_abs, d);
        sum += d;
    }
    mean_abs = n ? sum / n : 0.0;
    return max_abs;
}

struct PathResult {
    bool ok;
    size_t position;
    std::vector<std::vector<uint16_t>> attended;  // one entry per call
};

// Runs `calls` batched chunks through the layer with the given path flag set.
// The RNG is re-seeded per run so CPU and Metal see identical inputs.
PathResult run_path(RindiAttention& attn, const std::vector<uint16_t>& hidden,
                    const std::vector<size_t>& chunk_lanes, bool disable_metal,
                    std::mt19937& rng) {
    rng.seed(0x523138u);
    if (disable_metal) setenv("RINDI_DISABLE_METAL_ATTENTION", "1", 1);
    else unsetenv("RINDI_DISABLE_METAL_ATTENTION");
    attn.reset();
    PathResult r{true, 0, {}};
    for (size_t lanes : chunk_lanes) {
        // Fresh channel-major projections for this chunk.
        auto h = random_halfs(kHidden * lanes, rng);
        std::vector<uint16_t> q, k, v;
        if (!attn.project(h.data(), lanes, q, k, v)) { r.ok = false; break; }
        std::vector<uint16_t> attended;
        if (!attn.core_step_batch(q.data(), k.data(), v.data(), lanes, attended)) {
            r.ok = false; break;
        }
        r.attended.push_back(std::move(attended));
    }
    r.position = attn.position();
    return r;
}

} // namespace

int main(int argc, char** argv) {
    const char* model_dir = argc > 1 ? argv[1]
        : "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    SafeTensorsLoader loader;
    if (!loader.open_file(std::string(model_dir) + "/gpu_backbone.safetensors")) {
        std::fprintf(stderr, "FAIL open backbone\n");
        return 2;
    }

    // First attention layer in the checkpoint.
    int layer = -1;
    for (int i = 0; i < 64; ++i) {
        if (loader.has_tensor("layers." + std::to_string(i) + ".self_attn.q_proj.weight")) {
            layer = i; break;
        }
    }
    if (layer < 0) { std::fprintf(stderr, "FAIL no attention layer found\n"); return 2; }

    RindiAttention attn;
    // A bare context opens the driver without loading programs; compile_core
    // requires a non-null context even though this path compiles host-side
    // int4 projections only.
    ANEContext* ane = ane_context_create();
    if (!ane) { std::fprintf(stderr, "FAIL ane_context_create\n"); return 2; }
    size_t context = 4096;
    size_t width = 32;
    if (const char* reserve = std::getenv("RINDI_TEST_RESERVE_CONTEXT")) {
        context = static_cast<size_t>(std::strtoull(reserve, nullptr, 10));
        width = 128;
    }
    if (!attn.compile_core(ane, loader, layer, context, width)) {
        std::fprintf(stderr, "FAIL compile_core layer %d\n", layer);
        return 2;
    }
    std::printf("layer=%d context=%zu width=%zu\n", layer, context, width);
    if (std::getenv("RINDI_TEST_RESERVE_CONTEXT")) {
        const bool pass = attn.reserve_context(context);
        std::printf("ATTENTION_RESERVE=%s rows=%zu width=%zu\n",
                    pass ? "PASS" : "FAIL", context, width);
        ane_context_destroy(ane);
        return pass ? 0 : 1;
    }

    std::mt19937 rng(0x523138u);
    const std::vector<size_t> decode_calls{1, 1, 1};
    const std::vector<size_t> batch_calls{8, 8};

    // Decode-shaped: three single-token steps.
    PathResult cpu_d = run_path(attn, {}, decode_calls, true, rng);
    PathResult metal_d = run_path(attn, {}, decode_calls, false, rng);
    if (!cpu_d.ok || !metal_d.ok || cpu_d.attended.size() != metal_d.attended.size()) {
        std::fprintf(stderr, "FAIL decode run (cpu=%d metal=%d)\n", (int)cpu_d.ok, (int)metal_d.ok);
        return 3;
    }

    double worst_max = 0.0, worst_mean = 0.0;
    for (size_t c = 0; c < cpu_d.attended.size(); ++c) {
        double mean = 0.0;
        const double mx = diff_stats(cpu_d.attended[c], metal_d.attended[c], mean);
        worst_max = std::max(worst_max, mx);
        worst_mean = std::max(worst_mean, mean);
        std::printf("decode step %zu: max_abs=%.6f mean_abs=%.6f\n", c, mx, mean);
    }
    // Positions must advance identically.
    if (cpu_d.position != metal_d.position) {
        std::printf("FAIL position mismatch %zu vs %zu\n", cpu_d.position, metal_d.position);
        return 4;
    }

    // Prefill-shaped: two 8-lane chunks (causal masking inside the chunk).
    PathResult cpu_b = run_path(attn, {}, batch_calls, true, rng);
    PathResult metal_b = run_path(attn, {}, batch_calls, false, rng);
    if (!cpu_b.ok || !metal_b.ok) {
        std::fprintf(stderr, "FAIL batch run\n");
        return 3;
    }
    for (size_t c = 0; c < cpu_b.attended.size(); ++c) {
        double mean = 0.0;
        const double mx = diff_stats(cpu_b.attended[c], metal_b.attended[c], mean);
        worst_max = std::max(worst_max, mx);
        worst_mean = std::max(worst_mean, mean);
        std::printf("batch chunk %zu (lanes=8): max_abs=%.6f mean_abs=%.6f\n", c, mx, mean);
    }

    const bool pass = worst_max < 5e-3 && worst_mean < 5e-4 &&
                      cpu_d.position == 3 && cpu_b.position == 16;
    std::printf("%s max_abs=%.6f mean_abs=%.6f decode_pos=%zu batch_pos=%zu\n",
                pass ? "METAL_ATTENTION=PASS" : "METAL_ATTENTION=FAIL",
                worst_max, worst_mean, cpu_d.position, cpu_b.position);
    unsetenv("RINDI_DISABLE_METAL_ATTENTION");
    ane_context_destroy(ane);
    return pass ? 0 : 1;
}
