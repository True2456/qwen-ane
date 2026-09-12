// SPDX-License-Identifier: Apache-2.0
// Correctness and API regression test for the shared SME2 projection backend.

#include "runtime/rindi_sme_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

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

bool test_int8(RindiSmeEngine& engine) {
    constexpr int M = 3, N = 11, K = 257;
    std::vector<int8_t> a(M * K), b(N * K);
    std::vector<int32_t> got(M * N), reference(M * N, 0);
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<int8_t>(int(i % 31) - 15);
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<int8_t>(int(i % 17) - 8);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                reference[m * N + n] += int32_t(a[m * K + k]) * b[n * K + k];
    return engine.gemm_int8(a.data(), b.data(), got.data(), M, N, K) &&
           got == reference;
}

bool test_q4(RindiSmeEngine& engine) {
    constexpr size_t rows = 37, cols = 1536;
    std::mt19937 random(0x534d4532u);
    std::uniform_real_distribution<float> activation(-2.0f, 2.0f);
    std::uniform_int_distribution<int> nibble(-8, 7);

    std::vector<uint16_t> x(cols), scales(rows);
    std::vector<uint8_t> weights(rows * cols / 2);
    for (auto& value : x) value = to_fp16(activation(random));
    for (auto& scale : scales) scale = to_fp16(0.005f + 0.02f * std::fabs(activation(random)));
    for (auto& byte : weights) {
        const int low = nibble(random), high = nibble(random);
        byte = static_cast<uint8_t>((low & 0x0f) | ((high & 0x0f) << 4));
    }

    std::vector<float> got(rows), quantized(rows), full(rows);
    if (!engine.gemv_q4_rowwise_f32(x.data(), weights.data(), scales.data(),
                                    got.data(), rows, cols) ||
        !RindiSmeEngine::gemv_q4_rowwise_quantized_reference(
            x.data(), weights.data(), scales.data(), quantized.data(), rows, cols) ||
        !RindiSmeEngine::gemv_q4_rowwise_full_reference(
            x.data(), weights.data(), scales.data(), full.data(), rows, cols)) return false;

    float exact_max = 0.0f, error_sq = 0.0f, reference_sq = 0.0f;
    for (size_t row = 0; row < rows; ++row) {
        exact_max = std::max(exact_max, std::fabs(got[row] - quantized[row]));
        const float error = got[row] - full[row];
        error_sq += error * error;
        reference_sq += full[row] * full[row];
    }
    const float relative_rms = std::sqrt(error_sq / std::max(reference_sq, 1.0e-20f));
    std::cout << "SME2_Q4_ACCURACY exact_max=" << exact_max
              << " q8_vs_fp16_relative_rms=" << relative_rms << '\n';
    if (exact_max > 1.0e-6f || relative_rms > 0.025f) return false;

    std::vector<uint16_t> half(rows);
    if (!engine.gemv_q4_rowwise_fp16(x.data(), weights.data(), scales.data(),
                                     half.data(), rows, cols)) return false;
    for (size_t row = 0; row < rows; ++row) {
        const float tolerance = 0.002f * std::max(1.0f, std::fabs(got[row]));
        if (std::fabs(from_fp16(half[row]) - got[row]) > tolerance) return false;
    }
    return true;
}

bool test_q8_rowwise(RindiSmeEngine& engine) {
    constexpr size_t rows = 41, cols = 256;
    std::vector<uint16_t> x(cols), scales(rows);
    std::vector<int8_t> weights(rows * cols);
    for (size_t c = 0; c < cols; ++c)
        x[c] = to_fp16((static_cast<int>(c % 255) - 127) * 0.01f);
    for (auto& scale : scales) scale = to_fp16(0.02f);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = static_cast<int8_t>(static_cast<int>(i % 255) - 127);

    std::vector<float> got(rows), reference(rows);
    if (!engine.gemv_q8_rowwise_f32(x.data(), weights.data(), scales.data(),
                                    got.data(), rows, cols)) return false;
    for (size_t row = 0; row < rows; ++row) {
        int32_t dot = 0;
        for (size_t c = 0; c < cols; ++c)
            dot += (static_cast<int>(c % 255) - 127) * weights[row * cols + c];
        reference[row] = static_cast<float>(dot) * 0.01f * from_fp16(scales[row]);
    }
    for (size_t row = 0; row < rows; ++row)
        if (std::fabs(got[row] - reference[row]) >
            0.002f * std::max(1.0f, std::fabs(reference[row]))) return false;
    return true;
}

bool test_async(RindiSmeEngine& engine) {
    constexpr int N = 9, K = 65;
    std::vector<uint16_t> x(K), weights(N * K), sync(N), async(N);
    for (int i = 0; i < K; ++i) x[i] = to_fp16((i % 13 - 6) * 0.03f);
    for (int i = 0; i < N * K; ++i) weights[i] = to_fp16((i % 11 - 5) * 0.02f);
    if (!engine.gemv_fp16(x.data(), weights.data(), sync.data(), N, K)) return false;
    engine.async_inproj(x.data(), weights.data(), async.data(), N, K);
    engine.wait_inproj();
    return sync == async;
}

} // namespace

int main(int argc, char** argv) {
    const size_t workers = argc > 1 ? std::max(1, std::atoi(argv[1])) : 2;
    RindiSmeEngine engine(workers);
    std::cout << "SME2_AVAILABLE=" << (engine.is_available() ? 1 : 0)
              << " workers=" << engine.worker_count() << '\n';
    const bool int8_ok = test_int8(engine);
    const bool q4_ok = test_q4(engine);
    const bool q8_rowwise_ok = test_q8_rowwise(engine);
    const bool async_ok = test_async(engine);
    std::cout << "SME2_INT8=" << (int8_ok ? "PASS" : "FAIL") << '\n'
              << "SME2_Q4=" << (q4_ok ? "PASS" : "FAIL") << '\n'
              << "SME2_Q8_ROWWISE=" << (q8_rowwise_ok ? "PASS" : "FAIL") << '\n'
              << "SME2_ASYNC=" << (async_ok ? "PASS" : "FAIL") << '\n';
    return int8_ok && q4_ok && q8_rowwise_ok && async_ok ? 0 : 1;
}
