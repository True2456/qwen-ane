/*
 * SPDX-License-Identifier: Apache-2.0
 * Shared SME2 Q4/Q8 projection backend.
 *
 * This translation unit is compiled with -march=armv9.2-a+sme2 and with
 * automatic vectorization disabled. Apple Silicon exposes SVE registers only
 * while streaming mode is active, so the explicit intrinsics below sit
 * strictly between SMSTART SM and SMSTOP SM. Keeping this in its own object
 * also prevents an SME-targeted compiler from emitting an accidental RDVL in
 * ordinary model/server code.
 */

#include "rindi_sme_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/sysctl.h>

#if defined(__aarch64__)
#include <arm_sme.h>
#include <arm_sve.h>
#include <dispatch/dispatch.h>
#endif

namespace {

static bool check_sme2_hardware() {
    int feature = 0;
    size_t size = sizeof(feature);
    return sysctlbyname("hw.optional.arm.FEAT_SME2", &feature, &size, nullptr, 0) == 0 &&
           feature != 0;
}

static inline float fp16_to_float(uint16_t bits) {
    _Float16 value;
    static_assert(sizeof(value) == sizeof(bits), "unexpected _Float16 size");
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

static inline uint16_t float_to_fp16(float value) {
    const _Float16 half = static_cast<_Float16>(value);
    uint16_t bits;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static inline int8_t signed_nibble(uint8_t q) {
    q &= 0x0f;
    return static_cast<int8_t>(q < 8 ? q : static_cast<int>(q) - 16);
}

static bool quantize_fp16_activation(const uint16_t* input, size_t cols,
                                     std::vector<int8_t>& output,
                                     float& scale) {
    if (!input || cols == 0) return false;
    output.resize(cols);
    float max_abs = 0.0f;
    for (size_t c = 0; c < cols; ++c) {
        const float value = fp16_to_float(input[c]);
        if (!std::isfinite(value)) return false;
        max_abs = std::max(max_abs, std::fabs(value));
    }
    if (max_abs == 0.0f) {
        std::fill(output.begin(), output.end(), int8_t{0});
        scale = 0.0f;
        return true;
    }
    scale = max_abs / 127.0f;
    const float inverse = 1.0f / scale;
    for (size_t c = 0; c < cols; ++c) {
        int q = static_cast<int>(std::lrint(fp16_to_float(input[c]) * inverse));
        q = std::max(-127, std::min(127, q));
        output[c] = static_cast<int8_t>(q);
    }
    return true;
}

static int32_t dot_q8_q4_scalar(const int8_t* x, const uint8_t* packed,
                                size_t cols) {
    int32_t sum = 0;
    for (size_t c = 0; c < cols; c += 2) {
        const uint8_t byte = packed[c / 2];
        sum += static_cast<int32_t>(x[c]) * signed_nibble(byte);
        sum += static_cast<int32_t>(x[c + 1]) * signed_nibble(byte >> 4);
    }
    return sum;
}

#if defined(__aarch64__)
// Streaming SVE is the correct SME2 execution form for a single-token GEMV:
// ZA outer products are useful once M>1, while SDOT avoids replicating the
// activation vector into an artificial matrix. Each vector load expands two
// signed Q4 vectors in registers and immediately consumes them with SDOT.
__attribute__((noinline))
static void dot_q8_q4_sme_rows(const int8_t* x, const uint8_t* packed_w,
                               int32_t* dots, size_t row_begin,
                               size_t row_end, size_t cols) {
    // Entering streaming mode changes the architectural vector register file.
    // Tell the ordinary AAPCS caller that its callee-saved vector registers
    // are clobbered so Clang spills/restores v8-v15 around this function.
    __asm__ volatile("smstart sm" ::: "memory",
                     "v8", "v9", "v10", "v11",
                     "v12", "v13", "v14", "v15");

    // svdup(0) is otherwise folded to an AdvSIMD `movi v0`, which is illegal
    // while Apple Silicon is in streaming mode. An opaque GPR zero forces the
    // compiler to emit the streaming-compatible `mov zN.s, wN` form.
    int32_t vector_zero;
    __asm__ volatile("mov %w0, wzr" : "=r"(vector_zero));
    const svbool_t all_b = svptrue_b8();
    const svbool_t all_s = svptrue_b32();
    const size_t vector_bytes = static_cast<size_t>(svcntb());
    const size_t packed_cols = cols / 2;

    for (size_t row = row_begin; row < row_end; ++row) {
        const uint8_t* weights = packed_w + row * packed_cols;
        svint32_t accum = svdup_n_s32(vector_zero);
        size_t byte = 0;
        size_t x_offset = 0;

        for (; byte + vector_bytes <= packed_cols;
             byte += vector_bytes, x_offset += 2 * vector_bytes) {
            const svuint8_t packed = svld1_u8(all_b, weights + byte);
            const svuint8_t low = svand_n_u8_x(all_b, packed, 0x0f);
            const svuint8_t high = svlsr_n_u8_x(all_b, packed, 4);

            // Sign-extend each nibble: (q << 4) arithmetic-shifted back by 4.
            svint8_t low_signed = svreinterpret_s8(
                svlsl_n_u8_x(all_b, low, 4));
            svint8_t high_signed = svreinterpret_s8(
                svlsl_n_u8_x(all_b, high, 4));
            low_signed = svasr_n_s8_x(all_b, low_signed, 4);
            high_signed = svasr_n_s8_x(all_b, high_signed, 4);

            // Restore low0,high0,low1,high1... logical column order.
            const svint8_t weights0 = svzip1_s8(low_signed, high_signed);
            const svint8_t weights1 = svzip2_s8(low_signed, high_signed);
            accum = svdot_s32(accum, svld1_s8(all_b, x + x_offset), weights0);
            accum = svdot_s32(accum,
                              svld1_s8(all_b, x + x_offset + vector_bytes),
                              weights1);
        }

        int32_t sum = static_cast<int32_t>(svaddv_s32(all_s, accum));
        for (; byte < packed_cols; ++byte) {
            const uint8_t q = weights[byte];
            sum += static_cast<int32_t>(x[2 * byte]) * signed_nibble(q);
            sum += static_cast<int32_t>(x[2 * byte + 1]) * signed_nibble(q >> 4);
        }
        dots[row] = sum;
    }

    __asm__ volatile("smstop sm" ::: "memory");
}

struct Q4DispatchJob {
    const int8_t* x;
    const uint8_t* weights;
    int32_t* dots;
    size_t rows;
    size_t cols;
    size_t workers;
};

static void q4_dispatch_worker(void* opaque, size_t worker) {
    auto* job = static_cast<Q4DispatchJob*>(opaque);
    const size_t begin = job->rows * worker / job->workers;
    const size_t end = job->rows * (worker + 1) / job->workers;
    if (begin != end)
        dot_q8_q4_sme_rows(job->x, job->weights, job->dots,
                           begin, end, job->cols);
}

static void compute_q4_sme(const int8_t* x, const uint8_t* weights,
                           int32_t* dots, size_t rows, size_t cols,
                           size_t workers) {
    if (workers <= 1 || rows < workers * 4) {
        dot_q8_q4_sme_rows(x, weights, dots, 0, rows, cols);
        return;
    }
    Q4DispatchJob job{x, weights, dots, rows, cols, workers};
    dispatch_apply_f(workers,
                     dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                     &job, q4_dispatch_worker);
}

__attribute__((noinline))
static int32_t dot_q8_q8_sme(const int8_t* a, const int8_t* b, size_t cols) {
    __asm__ volatile("smstart sm" ::: "memory",
                     "v8", "v9", "v10", "v11",
                     "v12", "v13", "v14", "v15");
    int32_t vector_zero;
    __asm__ volatile("mov %w0, wzr" : "=r"(vector_zero));
    const svbool_t all_b = svptrue_b8();
    const svbool_t all_s = svptrue_b32();
    svint32_t accum = svdup_n_s32(vector_zero);
    const size_t vector_bytes = static_cast<size_t>(svcntb());
    size_t c = 0;
    for (; c + vector_bytes <= cols; c += vector_bytes)
        accum = svdot_s32(accum, svld1_s8(all_b, a + c),
                          svld1_s8(all_b, b + c));
    int32_t sum = static_cast<int32_t>(svaddv_s32(all_s, accum));
    __asm__ volatile("smstop sm" ::: "memory");
    for (; c < cols; ++c)
        sum += static_cast<int32_t>(a[c]) * static_cast<int32_t>(b[c]);
    return sum;
}
#endif

} // namespace

RindiSmeEngine::RindiSmeEngine(size_t workers)
    : available_(check_sme2_hardware()), workers_(std::max<size_t>(1, workers)) {}

RindiSmeEngine::~RindiSmeEngine() {
    wait_inproj();
}

bool RindiSmeEngine::gemm_int8(const int8_t* A, const int8_t* B, int32_t* C,
                               int M, int N, int K) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) return false;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
#if defined(__aarch64__)
            if (available_) {
                C[m * N + n] = dot_q8_q8_sme(A + m * K, B + n * K,
                                               static_cast<size_t>(K));
                continue;
            }
#endif
            int32_t sum = 0;
            for (int k = 0; k < K; ++k)
                sum += static_cast<int32_t>(A[m * K + k]) *
                       static_cast<int32_t>(B[n * K + k]);
            C[m * N + n] = sum;
        }
    }
    return true;
}

bool RindiSmeEngine::gemv_q4_rowwise_f32(
    const uint16_t* x_fp16, const uint8_t* packed_w,
    const uint16_t* fp16_scales, float* y_f32,
    size_t rows, size_t cols) {
    if (!x_fp16 || !packed_w || !fp16_scales || !y_f32 ||
        rows == 0 || cols == 0 || (cols & 1)) return false;

    std::lock_guard<std::mutex> lock(call_mutex_);
    float activation_scale = 0.0f;
    if (!quantize_fp16_activation(x_fp16, cols, q8_activation_, activation_scale))
        return false;
    dot_scratch_.resize(rows);

#if defined(__aarch64__)
    if (available_) {
        compute_q4_sme(q8_activation_.data(), packed_w, dot_scratch_.data(),
                       rows, cols, workers_);
    } else
#endif
    {
        const size_t packed_cols = cols / 2;
        for (size_t row = 0; row < rows; ++row)
            dot_scratch_[row] = dot_q8_q4_scalar(
                q8_activation_.data(), packed_w + row * packed_cols, cols);
    }

    for (size_t row = 0; row < rows; ++row) {
        y_f32[row] = static_cast<float>(dot_scratch_[row]) * activation_scale *
                     fp16_to_float(fp16_scales[row]);
    }
    return true;
}

bool RindiSmeEngine::gemv_q4_rowwise_fp16(
    const uint16_t* x_fp16, const uint8_t* packed_w,
    const uint16_t* fp16_scales, uint16_t* y_fp16,
    size_t rows, size_t cols) {
    if (!y_fp16) return false;
    // Decode calls this once per selected layer. Reuse the conversion buffer
    // rather than allocating thousands of small vectors during generation.
    static thread_local std::vector<float> output;
    output.resize(rows);
    if (!gemv_q4_rowwise_f32(x_fp16, packed_w, fp16_scales,
                             output.data(), rows, cols)) return false;
    for (size_t row = 0; row < rows; ++row)
        y_fp16[row] = float_to_fp16(output[row]);
    return true;
}

bool RindiSmeEngine::gemv_q4_rowwise_quantized_reference(
    const uint16_t* x_fp16, const uint8_t* packed_w,
    const uint16_t* fp16_scales, float* y_f32,
    size_t rows, size_t cols) {
    if (!x_fp16 || !packed_w || !fp16_scales || !y_f32 ||
        rows == 0 || cols == 0 || (cols & 1)) return false;
    std::vector<int8_t> quantized;
    float activation_scale = 0.0f;
    if (!quantize_fp16_activation(x_fp16, cols, quantized, activation_scale))
        return false;
    const size_t packed_cols = cols / 2;
    for (size_t row = 0; row < rows; ++row) {
        const int32_t dot = dot_q8_q4_scalar(
            quantized.data(), packed_w + row * packed_cols, cols);
        y_f32[row] = static_cast<float>(dot) * activation_scale *
                     fp16_to_float(fp16_scales[row]);
    }
    return true;
}

bool RindiSmeEngine::gemv_q4_rowwise_full_reference(
    const uint16_t* x_fp16, const uint8_t* packed_w,
    const uint16_t* fp16_scales, float* y_f32,
    size_t rows, size_t cols) {
    if (!x_fp16 || !packed_w || !fp16_scales || !y_f32 ||
        rows == 0 || cols == 0 || (cols & 1)) return false;
    const size_t packed_cols = cols / 2;
    for (size_t row = 0; row < rows; ++row) {
        const uint8_t* weights = packed_w + row * packed_cols;
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c += 2) {
            const uint8_t q = weights[c / 2];
            sum += fp16_to_float(x_fp16[c]) *
                   static_cast<float>(signed_nibble(q));
            sum += fp16_to_float(x_fp16[c + 1]) *
                   static_cast<float>(signed_nibble(q >> 4));
        }
        y_f32[row] = sum * fp16_to_float(fp16_scales[row]);
    }
    return true;
}

bool RindiSmeEngine::gemv_fp16(const uint16_t* x, const uint16_t* W,
                               uint16_t* y, int N, int K) {
    if (!x || !W || !y || N <= 0 || K <= 0) return false;
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += fp16_to_float(x[k]) * fp16_to_float(W[n * K + k]);
        y[n] = float_to_fp16(sum);
    }
    return true;
}

void RindiSmeEngine::async_inproj(const uint16_t* x, const uint16_t* W,
                                  uint16_t* y, int N, int K) {
    wait_inproj();
    async_running_.store(true, std::memory_order_release);
    async_worker_ = std::thread([this, x, W, y, N, K] {
        gemv_fp16(x, W, y, N, K);
        async_running_.store(false, std::memory_order_release);
    });
}

void RindiSmeEngine::wait_inproj() {
    if (async_worker_.joinable()) async_worker_.join();
    async_running_.store(false, std::memory_order_release);
}
