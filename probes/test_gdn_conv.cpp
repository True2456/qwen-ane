// SPDX-License-Identifier: Apache-2.0
#include "../runtime/rindi_gdn_conv.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

static float half_to_float(uint16_t bits) {
    const uint32_t s = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t e = (bits >> 10) & 31u;
    const uint32_t m = bits & 1023u;
    uint32_t v = e == 0 ? s | (m << 13)
        : e == 31 ? s | 0x7f800000u | (m << 13)
                  : s | ((e + 112u) << 23) | (m << 13);
    float f; std::memcpy(&f, &v, sizeof(f)); return f;
}

int main() {
    SafeTensorsLoader loader;
    if (!loader.open_file("~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors")) return 2;
    ANEContext* ctx = ane_context_create();
    if (!ctx) return 2;
    RindiGdnConv conv;
    bool ok = conv.compile(ctx, loader, "layers.0.linear_attn.conv1d.weight", 32);
    std::vector<uint16_t> weights;
    ok = ok && loader.get_tensor_fp16("layers.0.linear_attn.conv1d.weight", weights);
    std::vector<uint16_t> input(conv.channels(), 0x3c00);
    std::vector<uint16_t> output;
    std::vector<float> history(conv.channels() * 3, 0.0f);
    std::cout << "  w0=" << half_to_float(weights[0]) << "," << half_to_float(weights[1])
              << "," << half_to_float(weights[2]) << "," << half_to_float(weights[3]) << "\n";
    float max_error = 0.0f;
    for (size_t step = 0; ok && step < 4; ++step) {
        ok = conv.evaluate(input.data(), 1, output) && output.size() == conv.channels();
        for (size_t c = 0; ok && c < conv.channels(); ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < 4; ++k) {
                const float x = k == 3 ? 1.0f : history[c * 3 + k];
                sum += x * half_to_float(weights[c * 4 + k]);
            }
            const float expected = sum / (1.0f + std::exp(-sum));
            const float got = half_to_float(output[c]);
            if (c == 0) std::cout << "  step=" << step << " expected=" << expected
                                  << " got=" << got << "\n";
            max_error = std::max(max_error, std::abs(expected - got));
        }
        for (size_t c = 0; c < conv.channels(); ++c) {
            float* h = history.data() + c * 3;
            h[0] = h[1]; h[1] = h[2]; h[2] = 1.0f;
        }
    }
    ok = ok && max_error < 0.02f;
    std::cout << (ok ? "GDN_CONV=PASS\n" : "GDN_CONV=FAIL\n")
              << "  max_error=" << max_error << "\n";
    ane_context_destroy(ctx);
    return ok ? 0 : 1;
}
