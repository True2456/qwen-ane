// SPDX-License-Identifier: Apache-2.0
// Exact-shape Q4 GEMV benchmark: SME2 versus the native Metal row-wise kernel.

#include "runtime/rindi_sme_engine.h"
#include "runtime/metal_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

uint16_t to_fp16(float value) {
    const _Float16 h = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float from_fp16(uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<float>(h);
}

struct Shape {
    const char* name;
    size_t rows;
    size_t cols;
};

struct Result {
    std::string name;
    double sme_ms{0.0};
    double metal_ms{0.0};
    double hetero_ms{0.0};
};

double epoch_seconds() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

template <class Function>
double measure_ms(const std::string& label, Function&& function, int iterations) {
    function();
    std::cout << std::fixed << std::setprecision(6)
              << "MEASURE_START " << label << ' ' << epoch_seconds() << '\n'
              << std::flush;
    const auto begin = Clock::now();
    for (int i = 0; i < iterations; ++i) function();
    const auto end = Clock::now();
    std::cout << std::fixed << std::setprecision(6)
              << "MEASURE_END " << label << ' ' << epoch_seconds() << '\n'
              << std::flush;
    return std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
}

Result run_shape(const Shape& shape, size_t workers, int iterations,
                 bool use_metal) {
    const size_t packed_bytes = shape.rows * shape.cols / 2;
    std::mt19937 random(static_cast<uint32_t>(shape.rows ^ (shape.cols << 8)));
    std::uniform_real_distribution<float> activation(-1.5f, 1.5f);
    std::uniform_int_distribution<int> nibble(-8, 7);

    std::vector<uint16_t> x(shape.cols), scales(shape.rows);
    std::vector<uint8_t> weights(packed_bytes);
    std::vector<float> sme_output(shape.rows);
    for (auto& value : x) value = to_fp16(activation(random));
    for (auto& scale : scales) scale = to_fp16(0.02f);
    for (auto& byte : weights) {
        const int low = nibble(random), high = nibble(random);
        byte = static_cast<uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
    }

    RindiSmeEngine sme(workers);
    if (!sme.is_available()) {
        std::cerr << "SME2 is not available on this machine\n";
        std::exit(2);
    }
    Result result;
    result.name = shape.name;
    result.sme_ms = measure_ms(std::string("sme2:") + shape.name, [&] {
        if (!sme.gemv_q4_rowwise_f32(x.data(), weights.data(), scales.data(),
                                     sme_output.data(), shape.rows, shape.cols))
            std::abort();
    }, iterations);

    const size_t checked_rows = std::min<size_t>(shape.rows, 64);
    std::vector<float> exact(checked_rows), full(checked_rows);
    RindiSmeEngine::gemv_q4_rowwise_quantized_reference(
        x.data(), weights.data(), scales.data(), exact.data(),
        checked_rows, shape.cols);
    RindiSmeEngine::gemv_q4_rowwise_full_reference(
        x.data(), weights.data(), scales.data(), full.data(),
        checked_rows, shape.cols);
    float exact_max = 0.0f, q8_error_sq = 0.0f, full_sq = 0.0f;
    for (size_t row = 0; row < checked_rows; ++row) {
        exact_max = std::max(exact_max, std::fabs(sme_output[row] - exact[row]));
        const float error = sme_output[row] - full[row];
        q8_error_sq += error * error;
        full_sq += full[row] * full[row];
    }
    const float q8_relative_rms = std::sqrt(q8_error_sq / std::max(full_sq, 1.0e-20f));

    float metal_relative_rms = 0.0f;
    if (use_metal) {
        MetalContext* context = metal_context_create();
        MetalBufferHandle xb = metal_buffer_create(context, x.size() * sizeof(uint16_t));
        MetalBufferHandle wb = metal_buffer_create(context, weights.size());
        MetalBufferHandle sb = metal_buffer_create(context, scales.size() * sizeof(uint16_t));
        MetalBufferHandle yb = metal_buffer_create(context, shape.rows * sizeof(uint16_t));
        if (!context || !xb || !wb || !sb || !yb) std::abort();
        std::memcpy(metal_buffer_get_contents(xb), x.data(), x.size() * sizeof(uint16_t));
        std::memcpy(metal_buffer_get_contents(wb), weights.data(), weights.size());
        std::memcpy(metal_buffer_get_contents(sb), scales.data(), scales.size() * sizeof(uint16_t));
        if (iterations >= 100) std::this_thread::sleep_for(std::chrono::seconds(2));
        result.metal_ms = measure_ms(std::string("metal:") + shape.name, [&] {
            MetalCommandBufferHandle command = metal_command_buffer_create(context);
            metal_dispatch_gemv_int4_rowwise_simd_offset(
                context, command, xb, 0, wb, sb, yb, 0,
                static_cast<int>(shape.rows), static_cast<int>(shape.cols),
                static_cast<int>(shape.cols / 2), 0,
                static_cast<int>(shape.rows));
            metal_command_buffer_commit(command);
            metal_command_buffer_wait(command);
        }, iterations);
        const auto* metal_output = static_cast<const uint16_t*>(
            metal_buffer_get_contents(yb));
        float error_sq = 0.0f;
        for (size_t row = 0; row < checked_rows; ++row) {
            const float error = from_fp16(metal_output[row]) - full[row];
            error_sq += error * error;
        }
        metal_relative_rms = std::sqrt(error_sq / std::max(full_sq, 1.0e-20f));
        const int default_percent = shape.cols >= 16000 ? 80 : 90;
        const int requested_percent = std::getenv("HETERO_GPU_PERCENT")
            ? std::atoi(std::getenv("HETERO_GPU_PERCENT")) : default_percent;
        size_t gpu_rows = shape.rows * static_cast<size_t>(
            std::max(1, std::min(99, requested_percent))) / 100;
        gpu_rows = std::max<size_t>(32, (gpu_rows / 32) * 32);
        gpu_rows = std::min(shape.rows - 32, gpu_rows);
        auto* hetero_output = static_cast<uint16_t*>(metal_buffer_get_contents(yb));
        result.hetero_ms = measure_ms(std::string("hetero:") + shape.name, [&] {
            MetalCommandBufferHandle command = metal_command_buffer_create(context);
            metal_dispatch_gemv_int4_rowwise_simd_offset(
                context, command, xb, 0, wb, sb, yb, 0,
                static_cast<int>(shape.rows), static_cast<int>(shape.cols),
                static_cast<int>(shape.cols / 2), 0, static_cast<int>(gpu_rows));
            metal_command_buffer_commit(command);
            if (!sme.gemv_q4_rowwise_fp16(
                    x.data(), weights.data() + gpu_rows * (shape.cols / 2),
                    scales.data() + gpu_rows, hetero_output + gpu_rows,
                    shape.rows - gpu_rows, shape.cols)) std::abort();
            metal_command_buffer_wait(command);
        }, iterations);
        metal_buffer_release(xb);
        metal_buffer_release(wb);
        metal_buffer_release(sb);
        metal_buffer_release(yb);
        metal_context_destroy(context);
    }

    const double gib = static_cast<double>(packed_bytes +
        shape.rows * sizeof(uint16_t)) / (1024.0 * 1024.0 * 1024.0);
    const double sme_gibs = gib / (result.sme_ms / 1000.0);
    const double metal_gibs = result.metal_ms > 0.0
        ? gib / (result.metal_ms / 1000.0) : 0.0;
    const double hetero_gibs = result.hetero_ms > 0.0
        ? gib / (result.hetero_ms / 1000.0) : 0.0;
    std::cout << std::fixed << std::setprecision(4)
              << "SME2_BENCH shape=" << shape.name
              << " rows=" << shape.rows << " cols=" << shape.cols
              << " mib=" << gib * 1024.0
              << " workers=" << workers
              << " sme_ms=" << result.sme_ms
              << " sme_GiBs=" << sme_gibs;
    if (use_metal)
        std::cout << " metal_ms=" << result.metal_ms
                  << " metal_GiBs=" << metal_gibs
                  << " hetero_ms=" << result.hetero_ms
                  << " hetero_GiBs=" << hetero_gibs
                  << " sme_vs_metal=" << result.metal_ms / result.sme_ms;
    std::cout << " sme_exact_max=" << exact_max
              << " sme_q8_rel_rms=" << q8_relative_rms;
    if (use_metal) std::cout << " metal_rel_rms=" << metal_relative_rms;
    std::cout << '\n';
    return result;
}

} // namespace

int main(int argc, char** argv) {
    size_t workers = 4;
    int iterations = 3;
    bool quick = false;
    bool use_metal = true;
    std::string selected_shape;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--workers" && i + 1 < argc)
            workers = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--iterations" && i + 1 < argc)
            iterations = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--quick") quick = true;
        else if (arg == "--no-metal") use_metal = false;
        else if (arg == "--shape" && i + 1 < argc) selected_shape = argv[++i];
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--workers N] [--iterations N] [--quick] [--no-metal]"
                         " [--shape NAME]\n";
            return 2;
        }
    }

    std::vector<Shape> shapes = {
        {"ling_expert_gate_up", 1024, 1536},
        {"ling_expert_down", 1536, 512},
    };
    if (!quick) {
        shapes.push_back({"dense_gate_up", 34816, 5120});
        shapes.push_back({"dense_down", 5120, 17408});
    }
    if (!selected_shape.empty()) {
        shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
            [&](const Shape& shape) { return selected_shape != shape.name; }),
            shapes.end());
        if (shapes.empty()) {
            std::cerr << "unknown shape: " << selected_shape << '\n';
            return 2;
        }
    }
    std::vector<Result> results;
    for (const auto& shape : shapes)
        results.push_back(run_shape(shape, workers, iterations, use_metal));

    if (!quick && results.size() == 4) {
        const double dense_layer_ms = results[2].sme_ms + results[3].sme_ms;
        const double dense_mlp_tps = 1000.0 / (64.0 * dense_layer_ms);
        const double ling_expert_ms = results[0].sme_ms + results[1].sme_ms;
        const double ling_mlp_tps = 1000.0 / (23.0 * 8.0 * ling_expert_ms);
        std::cout << std::fixed << std::setprecision(3)
                  << "SME2_MODEL_CEILING dense_all_sme_mlp_tps=" << dense_mlp_tps
                  << " ling_all_sme_routed_mlp_tps=" << ling_mlp_tps;
        if (use_metal) {
            const double dense_metal_ms = results[2].metal_ms + results[3].metal_ms;
            const double dense_best_ms =
                std::min(results[2].sme_ms, results[2].metal_ms) +
                std::min(results[3].sme_ms, results[3].metal_ms);
            const double ling_metal_ms = results[0].metal_ms + results[1].metal_ms;
            const double ling_best_ms =
                std::min(results[0].sme_ms, results[0].metal_ms) +
                std::min(results[1].sme_ms, results[1].metal_ms);
            std::cout << " dense_all_metal_mlp_tps="
                      << 1000.0 / (64.0 * dense_metal_ms)
                      << " dense_best_backend_mlp_tps="
                      << 1000.0 / (64.0 * dense_best_ms)
                      << " ling_all_metal_routed_mlp_tps="
                      << 1000.0 / (23.0 * 8.0 * ling_metal_ms)
                      << " ling_best_backend_routed_mlp_tps="
                      << 1000.0 / (23.0 * 8.0 * ling_best_ms);
        }
        std::cout << " note=excludes_attention_norm_dispatch_and_contention\n";
    }
    return 0;
}
