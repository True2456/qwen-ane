// SPDX-License-Identifier: Apache-2.0
#include "../runtime/rindi_ane_projection.h"
#include "../runtime/metal_engine.h"
#include <iostream>
#include <vector>

int main() {
    ANEContext* ctx = ane_context_create();
    if (!ctx) return 2;
    // 4x4 identity; two lanes should be copied unchanged.
    std::vector<uint16_t> weights(16, 0);
    for (int i = 0; i < 4; ++i) weights[i * 4 + i] = 0x3c00;
    RindiAneProjection projection;
    bool ok = projection.compile_fp16(ctx, weights.data(), 4, 4, 32, "identity");
    std::vector<uint16_t> input = {0x3c00, 0x4000, 0x4200, 0x4400,
                                   0x4500, 0x4600, 0x4700, 0x4800};
    std::vector<uint16_t> output;
    ok = ok && projection.evaluate(input.data(), 2, output);
    ok = ok && output == input;
    std::cout << (ok ? "ANE_PROJECTION=PASS\n" : "ANE_PROJECTION=FAIL\n");
    ane_context_destroy(ctx);
    return ok ? 0 : 1;
}
