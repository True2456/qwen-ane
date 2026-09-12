/* Affine 4-bit GEMV matching mlx.core.dequantize(mode="affine", bits=4, gs=64).
 *
 * packed uint32, low nibble first, 8 codes/word.
 * y[o] = sum_g scale[o,g] * (q[o,g]·x_g) + bias[o,g] * sum(x_g)
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void affine_q4_gemv_scalar(
    const uint32_t *packed,
    const float *scales,
    const float *biases,
    const float *x,
    float *y,
    int out_n,
    int in_n,
    int gs)
{
    const int n_g = in_n / gs;
    const int u32_per_g = gs / 8; /* 8 for gs=64, bits=4 */
    const int packed_in = in_n / 8;
    for (int o = 0; o < out_n; o++) {
        const uint32_t *prow = packed + (size_t)o * (size_t)packed_in;
        const float *srow = scales + (size_t)o * (size_t)n_g;
        const float *brow = biases + (size_t)o * (size_t)n_g;
        float acc = 0.f;
        for (int g = 0; g < n_g; g++) {
            const uint32_t *pg = prow + g * u32_per_g;
            const float *xg = x + g * gs;
            float qdot = 0.f;
            float xsum = 0.f;
            int k = 0;
            for (int u = 0; u < u32_per_g; u++) {
                uint32_t w = pg[u];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
                for (int n = 0; n < 8; n++, k++) {
                    float q = (float)(w & 15u);
                    qdot += q * xg[k];
                    xsum += xg[k];
                    w >>= 4;
                }
            }
            acc += srow[g] * qdot + brow[g] * xsum;
        }
        y[o] = acc;
    }
}

#if defined(__aarch64__)
/* Unpack in registers. No dequantized weight matrix is written to RAM. */
static inline float32x4_t dot16(uint8x16_t codes, const float *x) {
    uint16x8_t low = vmovl_u8(vget_low_u8(codes));
    uint16x8_t high = vmovl_u8(vget_high_u8(codes));
    float32x4_t a = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))), vld1q_f32(x));
    float32x4_t b = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(low))), vld1q_f32(x+4));
    a = vfmaq_f32(a, vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))), vld1q_f32(x+8));
    b = vfmaq_f32(b, vcvtq_f32_u32(vmovl_u16(vget_high_u16(high))), vld1q_f32(x+12));
    return vaddq_f32(a,b);
}
#endif

void affine_q4_gemv(const uint32_t *packed, const float *scales,
                    const float *biases, const float *x, float *y,
                    int out_n, int in_n, int gs) {
#if defined(__aarch64__)
    if (gs <= 0 || in_n <= 0 || out_n <= 0) return;
    if (gs % 32 || in_n % gs) {
        affine_q4_gemv_scalar(packed, scales, biases, x, y, out_n, in_n, gs);
        return;
    }
    const int ng = in_n / gs;
    float sums[ng];
    for (int g=0; g<ng; ++g) {
        float32x4_t sum = vdupq_n_f32(0);
        for (int k=0; k<gs; k+=4) sum = vaddq_f32(sum, vld1q_f32(x+g*gs+k));
        sums[g] = vaddvq_f32(sum);
    }
    const uint8x16_t mask = vdupq_n_u8(15);
    for (int o=0; o<out_n; ++o) {
        const uint8_t *row = (const uint8_t *)packed + (size_t)o * in_n/2;
        float acc = 0;
        for (int g=0; g<ng; ++g) {
            float32x4_t dot = vdupq_n_f32(0);
            for (int k=0; k<gs; k+=32) {
                uint8x16_t bytes = vld1q_u8(row+(g*gs+k)/2);
                uint8x16_t lo = vandq_u8(bytes, mask), hi = vshrq_n_u8(bytes,4);
                dot = vaddq_f32(dot, dot16(vzip1q_u8(lo,hi), x+g*gs+k));
                dot = vaddq_f32(dot, dot16(vzip2q_u8(lo,hi), x+g*gs+k+16));
            }
            size_t idx = (size_t)o*ng+g;
            acc += scales[idx]*vaddvq_f32(dot) + biases[idx]*sums[g];
        }
        y[o] = acc;
    }
#else
    affine_q4_gemv_scalar(packed, scales, biases, x, y, out_n, in_n, gs);
#endif
}

struct swiglu_context {
    const uintptr_t *ptrs;
    const float *x, *scores;
    float *outputs;
    int hidden, intermediate, gs;
};

static void swiglu_expert(void *raw, size_t expert) {
    struct swiglu_context *ctx = raw;
    const uintptr_t *p = ctx->ptrs + 9*expert;
    int h=ctx->hidden, i=ctx->intermediate;
    float gate[i], up[i];
    affine_q4_gemv((const uint32_t *)p[0], (const float *)p[1], (const float *)p[2], ctx->x, gate, i,h,ctx->gs);
    affine_q4_gemv((const uint32_t *)p[3], (const float *)p[4], (const float *)p[5], ctx->x, up, i,h,ctx->gs);
    for (int j=0;j<i;++j) {
        float clipped=fmaxf(-80.f,fminf(80.f,gate[j]));
        gate[j]=(gate[j]/(1.f+expf(-clipped)))*up[j]*ctx->scores[expert];
    }
    affine_q4_gemv((const uint32_t *)p[6], (const float *)p[7], (const float *)p[8], gate, ctx->outputs+expert*h,h,i,ctx->gs);
}

void affine_q4_swiglu_topk(const uintptr_t *ptrs, const float *x,
                          const float *scores, float *outputs,
                          int count, int hidden, int intermediate, int gs) {
    struct swiglu_context ctx = {ptrs,x,scores,outputs,hidden,intermediate,gs};
#ifdef __APPLE__
    dispatch_apply_f(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), &ctx, swiglu_expert);
#else
    for (int e=0;e<count;++e) swiglu_expert(&ctx,e);
#endif
}

void affine_q8_gemv(
    const uint32_t *packed,
    const float *scales,
    const float *biases,
    const float *x,
    float *y,
    int out_n,
    int in_n,
    int gs)
{
    const int n_g = in_n / gs;
    const int u32_per_g = gs / 4; /* 8-bit: 4 codes/word */
    const int packed_in = in_n / 4;
    for (int o = 0; o < out_n; o++) {
        const uint32_t *prow = packed + (size_t)o * (size_t)packed_in;
        const float *srow = scales + (size_t)o * (size_t)n_g;
        const float *brow = biases + (size_t)o * (size_t)n_g;
        float acc = 0.f;
        for (int g = 0; g < n_g; g++) {
            const uint32_t *pg = prow + g * u32_per_g;
            const float *xg = x + g * gs;
            float qdot = 0.f;
            float xsum = 0.f;
            int k = 0;
            for (int u = 0; u < u32_per_g; u++) {
                uint32_t w = pg[u];
                for (int n = 0; n < 4; n++, k++) {
                    float q = (float)(w & 255u);
                    qdot += q * xg[k];
                    xsum += xg[k];
                    w >>= 8;
                }
            }
            acc += srow[g] * qdot + brow[g] * xsum;
        }
        y[o] = acc;
    }
}

void affine_q4_dequant(
    const uint32_t *packed,
    const float *scales,
    const float *biases,
    float *dest,
    int out_n,
    int in_n,
    int gs)
{
    const int n_g = in_n / gs;
    const int packed_in = in_n / 8;
    for (int o = 0; o < out_n; o++) {
        const uint32_t *prow = packed + (size_t)o * (size_t)packed_in;
        const float *srow = scales + (size_t)o * (size_t)n_g;
        const float *brow = biases + (size_t)o * (size_t)n_g;
        float *drow = dest + (size_t)o * (size_t)in_n;
        int k = 0;
        for (int g = 0; g < n_g; g++) {
            float s = srow[g];
            float b = brow[g];
            for (int u = 0; u < gs / 8; u++) {
                uint32_t w = prow[g * (gs / 8) + u];
                for (int n = 0; n < 8; n++, k++) {
                    drow[k] = s * (float)(w & 15u) + b;
                    w >>= 4;
                }
            }
        }
    }
}
