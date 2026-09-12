// SPDX-License-Identifier: Apache-2.0
// Concurrent GPU + SME2 row-split benchmark for production rowwise Q4 GEMV.

#include "runtime/metal_engine.h"
#include "runtime/rindi_sme_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint16_t fp16(float value) {
    const _Float16 half = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float fp32(uint16_t bits) {
    _Float16 half;
    std::memcpy(&half, &bits, sizeof(bits));
    return static_cast<float>(half);
}

template <class Function>
double measure(Function&& function, int iterations) {
    function();
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) function();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count() /
           static_cast<double>(iterations);
}

struct Shape {
    const char* name;
    size_t rows;
    size_t cols;
};

double relative_rms(const uint16_t* output, const std::vector<float>& reference) {
    double error = 0.0, signal = 0.0;
    for (size_t row = 0; row < reference.size(); ++row) {
        const double ref = reference[row];
        const double diff = static_cast<double>(fp32(output[row])) - ref;
        error += diff * diff;
        signal += ref * ref;
    }
    return std::sqrt(error / std::max(signal, 1.0e-30));
}

int run(const Shape& shape, size_t workers, int iterations) {
    const size_t packed_cols = shape.cols / 2;
    const size_t packed_bytes = shape.rows * packed_cols;
    std::mt19937 rng(static_cast<uint32_t>(shape.rows ^ (shape.cols << 7)));
    std::uniform_real_distribution<float> activation(-1.5f, 1.5f);
    std::uniform_int_distribution<int> q4(-8, 7);

    std::vector<uint16_t> x(shape.cols), scales(shape.rows);
    std::vector<uint8_t> weights(packed_bytes);
    std::vector<float> reference(shape.rows);
    for (auto& value : x) value = fp16(activation(rng));
    for (auto& scale : scales) scale = fp16(0.02f);
    for (auto& packed : weights) {
        const int lo = q4(rng), hi = q4(rng);
        packed = static_cast<uint8_t>((lo & 15) | ((hi & 15) << 4));
    }
    if (!RindiSmeEngine::gemv_q4_rowwise_full_reference(
            x.data(), weights.data(), scales.data(), reference.data(),
            shape.rows, shape.cols)) return 2;

    RindiSmeEngine sme(workers);
    MetalContext* metal = metal_context_create();
    if (!sme.is_available() || !metal) return 2;
    MetalBufferHandle xb = metal_buffer_create(metal, x.size() * sizeof(uint16_t));
    MetalBufferHandle wb = metal_buffer_create(metal, weights.size());
    MetalBufferHandle sb = metal_buffer_create(metal, scales.size() * sizeof(uint16_t));
    MetalBufferHandle yb = metal_buffer_create(metal, shape.rows * sizeof(uint16_t));
    MetalBufferHandle pb = metal_buffer_create(metal, shape.rows * 4 * sizeof(float));
    if (!xb || !wb || !sb || !yb || !pb) return 2;
    std::memcpy(metal_buffer_get_contents(xb), x.data(), x.size() * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(wb), weights.data(), weights.size());
    std::memcpy(metal_buffer_get_contents(sb), scales.data(), scales.size() * sizeof(uint16_t));
    auto* output = static_cast<uint16_t*>(metal_buffer_get_contents(yb));

    const auto gpu_run = [&](bool simd) {
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal);
        if (simd) {
            metal_dispatch_gemv_int4_rowwise_simd_offset(
                metal, cmd, xb, 0, wb, sb, yb, 0,
                static_cast<int>(shape.rows), static_cast<int>(shape.cols),
                static_cast<int>(packed_cols), 0, static_cast<int>(shape.rows));
        } else {
            metal_dispatch_gemm_int4_rowwise_offset(
                metal, cmd, xb, 0, wb, sb, yb, 0,
                static_cast<int>(shape.rows), static_cast<int>(shape.cols),
                static_cast<int>(packed_cols), 1);
        }
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
    };

    const double legacy_ms = measure([&] { gpu_run(false); }, iterations);
    const double simd_ms = measure([&] { gpu_run(true); }, iterations);
    gpu_run(true);
    const double simd_error = relative_rms(output, reference);
    const auto splitk_run = [&] {
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal);
        metal_dispatch_gemv_int4_rowwise_simd_splitk_offset(
            metal, cmd, xb, 0, wb, sb, pb, yb, 0,
            static_cast<int>(shape.rows), static_cast<int>(shape.cols),
            static_cast<int>(packed_cols), 0, static_cast<int>(shape.rows), 4);
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
    };
    const double splitk_ms = measure(splitk_run, iterations);
    splitk_run();
    const double splitk_error = relative_rms(output, reference);

    double best_ms = 1.0e30;
    size_t best_gpu_rows = 0;
    double best_error = 0.0;
    const int percentages[] = {10, 20, 30, 40, 50, 60, 70, 75, 80,
                               85, 88, 90, 92, 94, 96, 98};
    for (const int percent : percentages) {
        size_t gpu_rows = shape.rows * static_cast<size_t>(percent) / 100;
        gpu_rows = std::max<size_t>(32, std::min(shape.rows - 32,
                                                 (gpu_rows / 32) * 32));
        const auto stacked = [&] {
            MetalCommandBufferHandle cmd = metal_command_buffer_create(metal);
            metal_dispatch_gemv_int4_rowwise_simd_offset(
                metal, cmd, xb, 0, wb, sb, yb, 0,
                static_cast<int>(shape.rows), static_cast<int>(shape.cols),
                static_cast<int>(packed_cols), 0, static_cast<int>(gpu_rows));
            metal_command_buffer_commit(cmd);
            const bool ok = sme.gemv_q4_rowwise_fp16(
                x.data(), weights.data() + gpu_rows * packed_cols,
                scales.data() + gpu_rows, output + gpu_rows,
                shape.rows - gpu_rows, shape.cols);
            metal_command_buffer_wait(cmd);
            if (!ok) std::abort();
        };
        const double elapsed = measure(stacked, iterations);
        stacked();
        const double error = relative_rms(output, reference);
        std::printf("HETERO_SWEEP shape=%s gpu_percent=%d gpu_rows=%zu "
                    "sme_rows=%zu ms=%.4f rel_rms=%.6f\n",
                    shape.name, percent, gpu_rows, shape.rows - gpu_rows,
                    elapsed, error);
        if (elapsed < best_ms) {
            best_ms = elapsed;
            best_gpu_rows = gpu_rows;
            best_error = error;
        }
    }

    const double gib = static_cast<double>(packed_bytes + shape.rows * 2) /
                       (1024.0 * 1024.0 * 1024.0);
    std::printf("HETERO_Q4 shape=%s rows=%zu cols=%zu workers=%zu "
                "legacy_ms=%.4f legacy_GiBs=%.3f simd_ms=%.4f simd_GiBs=%.3f "
                "splitk_ms=%.4f splitk_GiBs=%.3f simd_speedup=%.3f "
                "simd_rel_rms=%.6f splitk_rel_rms=%.6f best_gpu_rows=%zu "
                "best_sme_rows=%zu stacked_ms=%.4f stacked_GiBs=%.3f "
                "stacked_vs_simd=%.3f stacked_rel_rms=%.6f\n",
                shape.name, shape.rows, shape.cols, workers,
                legacy_ms, gib / (legacy_ms / 1000.0),
                simd_ms, gib / (simd_ms / 1000.0),
                splitk_ms, gib / (splitk_ms / 1000.0), legacy_ms / simd_ms,
                simd_error, splitk_error, best_gpu_rows, shape.rows - best_gpu_rows,
                best_ms, gib / (best_ms / 1000.0), simd_ms / best_ms,
                best_error);

    metal_buffer_release(xb);
    metal_buffer_release(wb);
    metal_buffer_release(sb);
    metal_buffer_release(yb);
    metal_buffer_release(pb);
    metal_context_destroy(metal);
    if (!std::isfinite(simd_error) || simd_error > 0.001 ||
        !std::isfinite(splitk_error) || splitk_error > 0.001 ||
        !std::isfinite(best_error) || best_error > 0.01) {
        std::fprintf(stderr, "HETERO_Q4 validation failed\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string selected = argc > 1 ? argv[1] : "dense_down";
    const int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 100;
    const size_t workers = argc > 3
        ? static_cast<size_t>(std::max(1, std::atoi(argv[3]))) : 4;
    const Shape shapes[] = {
        {"dense_gate_up", 34816, 5120},
        {"dense_down", 5120, 17408},
        {"ling_expert_gate_up", 1024, 1536},
        {"ling_expert_down", 1536, 512},
        // Qwen3.8-Flash-Next routes ten 640-wide experts per token.
        // The stacked shapes model assigning complete selected experts to
        // Metal and SME2 without synchronizing inside an expert projection.
        {"qwen38_expert_gate_up", 640, 2560},
        {"qwen38_selected_gate_up", 6400, 2560},
        {"qwen38_selected_fused_gate_up", 12800, 2560},
        {"qwen38_selected_down", 25600, 640},
    };
    for (const auto& shape : shapes)
        if (selected == shape.name) return run(shape, workers, iterations);
    std::fprintf(stderr, "unknown shape: %s\n", selected.c_str());
    return 2;
}
