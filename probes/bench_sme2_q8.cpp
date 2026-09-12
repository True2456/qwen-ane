// SPDX-License-Identifier: Apache-2.0
// SME2 row-wise Q8 GEMV benchmark for independent projection branches.

#include "runtime/rindi_sme_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t fp16(float value) {
    const _Float16 half = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

double run(size_t rows, size_t cols, size_t workers, int iterations) {
    std::mt19937 rng(static_cast<uint32_t>(rows ^ (cols << 8)));
    std::uniform_real_distribution<float> activation(-1.5f, 1.5f);
    std::uniform_int_distribution<int> weight(-127, 127);
    std::vector<uint16_t> x(cols), scales(rows), output(rows);
    std::vector<int8_t> weights(rows * cols);
    for (auto& value : x) value = fp16(activation(rng));
    for (auto& scale : scales) scale = fp16(0.01f);
    for (auto& value : weights) value = static_cast<int8_t>(weight(rng));

    RindiSmeEngine engine(workers);
    if (!engine.is_available()) return -1.0;
    const auto evaluate = [&] {
        if (!engine.gemv_q8_rowwise_fp16(x.data(), weights.data(), scales.data(),
                                         output.data(), rows, cols)) std::abort();
    };
    evaluate();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) evaluate();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count() / iterations;
    const double gib = static_cast<double>(weights.size()) /
                       (1024.0 * 1024.0 * 1024.0);
    std::printf("SME2_Q8 rows=%zu cols=%zu workers=%zu ms=%.4f GiBs=%.3f\n",
                rows, cols, workers, ms, gib / (ms / 1000.0));
    return ms;
}

}  // namespace

int main(int argc, char** argv) {
    const size_t rows = argc > 1 ? std::max(1, std::atoi(argv[1])) : 12800;
    const size_t cols = argc > 2 ? std::max(1, std::atoi(argv[2])) : 2560;
    const size_t workers = argc > 3 ? std::max(1, std::atoi(argv[3])) : 4;
    const int iterations = argc > 4 ? std::max(1, std::atoi(argv[4])) : 100;
    return run(rows, cols, workers, iterations) < 0.0 ? 2 : 0;
}
