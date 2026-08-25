/*
 * SPDX-License-Identifier: Apache-2.0
 * Shared SME2 projection backend for the dense and MoE engines.
 */

#ifndef RINDI_SME_ENGINE_H
#define RINDI_SME_ENGINE_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

class RindiSmeEngine {
public:
    // workers=1 is the lowest-power mode. More workers split output rows over
    // the global user-interactive queue and are intended to be selected by a
    // measured execution plan, not unconditionally by a model.
    explicit RindiSmeEngine(size_t workers = 1);
    ~RindiSmeEngine();

    RindiSmeEngine(const RindiSmeEngine&) = delete;
    RindiSmeEngine& operator=(const RindiSmeEngine&) = delete;

    bool is_available() const { return available_; }
    size_t worker_count() const { return workers_; }

    // C[M,N] = A[M,K] * B[N,K]^T. A and B are signed INT8, C is INT32.
    // This is a real pointer-driven SME streaming-SVE dot path. It falls back
    // to scalar arithmetic on machines without SME2 so callers stay correct.
    bool gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                   int M, int N, int K);

    // Row-wise signed INT4 x per-token INT8 GEMV used by the native chain.
    // packed_w is [rows, cols/2], low nibble first, with two's-complement
    // nibbles in [-8,7]. fp16_scales has one multiplier per output row.
    // Input activation quantization is symmetric and per token.
    bool gemv_q4_rowwise_f32(const uint16_t* x_fp16,
                             const uint8_t* packed_w,
                             const uint16_t* fp16_scales,
                             float* y_f32,
                             size_t rows, size_t cols);
    bool gemv_q4_rowwise_fp16(const uint16_t* x_fp16,
                              const uint8_t* packed_w,
                              const uint16_t* fp16_scales,
                              uint16_t* y_fp16,
                              size_t rows, size_t cols);

    // References used by probes and regression tests. The quantized reference
    // must match SME2 apart from the final FP operation. The full reference
    // keeps FP16 activations and measures the accuracy cost of Q8 activations.
    static bool gemv_q4_rowwise_quantized_reference(
        const uint16_t* x_fp16, const uint8_t* packed_w,
        const uint16_t* fp16_scales, float* y_f32,
        size_t rows, size_t cols);
    static bool gemv_q4_rowwise_full_reference(
        const uint16_t* x_fp16, const uint8_t* packed_w,
        const uint16_t* fp16_scales, float* y_f32,
        size_t rows, size_t cols);

    // Portable FP16 reference retained for existing callers. The optimized
    // native model path should use the quantized projection above.
    bool gemv_fp16(const uint16_t* x, const uint16_t* W, uint16_t* y,
                   int N, int K);

    // Existing overlap API, now genuinely asynchronous.
    void async_inproj(const uint16_t* x, const uint16_t* W, uint16_t* y,
                      int N, int K);
    void wait_inproj();

private:
    bool available_{false};
    size_t workers_{1};
    std::mutex call_mutex_;
    std::vector<int8_t> q8_activation_;
    std::vector<int32_t> dot_scratch_;

    std::atomic<bool> async_running_{false};
    std::thread async_worker_;
};

#endif // RINDI_SME_ENGINE_H
