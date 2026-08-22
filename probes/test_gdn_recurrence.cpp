#include "runtime/ane_c_bridge.h"
#include "runtime/rindi_gdn_recurrence.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static float half_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t value = exponent == 0 ? sign
        : (exponent == 31 ? sign | 0x7f800000u | (mantissa << 13)
                          : sign | ((exponent + 112u) << 23) | (mantissa << 13));
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static uint16_t float_to_half(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = (bits >> 13) & 0x3ffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

int main() {
    ANEContext* ctx = ane_context_create();
    if (!ctx) return 2;

    RindiGdnRecurrence recurrence;
    if (!recurrence.compile(ctx)) {
        std::fprintf(stderr, "recurrence compile failed\n");
        ane_context_destroy(ctx);
        return 3;
    }

    constexpr size_t H = 48, D = 128, V = 128, HK = H * D, W = 160;
    constexpr size_t C = HK + 2 * H;
    std::vector<uint16_t> input(C * W, 0);
    auto at = [&](size_t c, size_t w) -> uint16_t& { return input[c * W + w]; };
    for (size_t c = 0; c < HK; ++c) {
        at(c, V) = 0x3c00;       // decay = 1
        at(c, V + 1) = 0x3c00;   // key = 1
        at(c, V + 2) = 0x3c00;   // query = 1
    }
    for (size_t c = HK; c < HK + H; ++c)
        for (size_t w = 0; w < V; ++w) at(c, w) = 0x3c00;
    for (size_t c = HK + H; c < HK + 2 * H; ++c) at(c, 0) = 0x3c00;

    std::vector<uint16_t> output;
    bool ok1 = recurrence.step(input.data(), output);
    const uint16_t first = output.empty() ? 0 : output[0];
    bool ok2 = recurrence.step(input.data(), output);
    // With zero state and all-one decay/key/query/value/beta, the first state
    // is all ones and y is the sum of 128 ones.  On the second identical
    // update, the delta rule is 1 + (1 - 128) = -126, so y is -16128.
    // This catches output-surface order and grouped-convolution mistakes that
    // a shape-only test misses.
    const bool numeric = output.size() == H * V &&
        std::fabs(half_to_float(first) - 128.0f) < 1.0f &&
        std::fabs(half_to_float(output[0]) + 16128.0f) < 32.0f;
    bool differential = true;
    recurrence.reset();
    std::vector<float> state(H * D * V, 0.0f);
    for (size_t step = 0; step < 3 && differential; ++step) {
        for (size_t h = 0; h < H; ++h) {
            const float decay = 0.82f + 0.001f * static_cast<float>(h % 7);
            const float beta = 0.25f + 0.002f * static_cast<float>(h % 5);
            for (size_t d = 0; d < D; ++d)
                for (size_t v = 0; v < V; ++v) state[(h * D + d) * V + v] *= decay;
            for (size_t d = 0; d < D; ++d) {
                const float k = 0.05f * std::sin(0.013f * static_cast<float>(step * D + d + h));
                const float q = 0.04f * std::cos(0.017f * static_cast<float>(step * D + d + 2 * h));
                input[(h * D + d) * W + V] = float_to_half(decay);
                input[(h * D + d) * W + V + 1] = float_to_half(k);
                input[(h * D + d) * W + V + 2] = float_to_half(q);
                for (size_t v = 0; v < V; ++v) {
                    const float value = 0.1f * std::sin(0.009f * static_cast<float>(step * V + v + 3 * h));
                    input[(HK + h) * W + v] = float_to_half(value);
                }
                input[(HK + H + h) * W] = float_to_half(beta);
            }
            for (size_t v = 0; v < V; ++v) {
                float kv = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    const float k = half_to_float(input[(h * D + d) * W + V + 1]);
                    kv += state[(h * D + d) * V + v] * k;
                }
                const float value = half_to_float(input[(HK + h) * W + v]);
                const float beta = half_to_float(input[(HK + H + h) * W]);
                const float delta = (value - kv) * beta;
                for (size_t d = 0; d < D; ++d) {
                    const float k = half_to_float(input[(h * D + d) * W + V + 1]);
                    state[(h * D + d) * V + v] += k * delta;
                }
            }
        }
        std::vector<uint16_t> got;
        differential = recurrence.step(input.data(), got) && got.size() == H * V;
        float max_error = 0.0f;
        for (size_t h = 0; h < H && differential; ++h) {
            for (size_t v = 0; v < V; ++v) {
                float expected = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    const float q = half_to_float(input[(h * D + d) * W + V + 2]);
                    expected += state[(h * D + d) * V + v] * q;
                }
                max_error = std::max(max_error, std::fabs(expected - half_to_float(got[h * V + v])));
            }
        }
        differential = differential && max_error < 0.02f;
        if (!differential) std::printf("  differential_step=%zu max_error=%g\n", step, max_error);
    }
    std::printf("GDN_RECURRENCE=%s outputs=%zu first=%04x second=%04x\n",
                (ok1 && ok2 && numeric && differential) ? "PASS" : "FAIL", output.size(),
                output.empty() ? 0 : output[0], output.size() < 2 ? 0 : output[1]);
    ane_context_destroy(ctx);
    return (ok1 && ok2 && numeric && differential) ? 0 : 1;
}
