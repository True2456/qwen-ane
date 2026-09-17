// SPDX-License-Identifier: Apache-2.0
#include "../runtime/rindi_ane_projection.h"
#include <iostream>
#include <vector>

int main() {
    SafeTensorsLoader loader;
    if (!loader.open_file("~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors")) return 2;
    ANEContext* ctx = ane_context_create();
    if (!ctx) return 2;
    RindiAneProjection projection;
    bool ok = projection.compile_int4(
        ctx, loader, "layers.0.linear_attn.in_proj_a.weight", 32, "qwen_int4_smoke");
    std::cerr << "compiled=" << (ok ? "yes" : "no") << "\n";
    std::vector<uint16_t> input(5120 * 1, 0x3c00);
    std::vector<uint16_t> output;
    ok = ok && projection.evaluate(input.data(), 1, output);
    std::cerr << "evaluated=" << (!output.empty() ? "yes" : "no") << " size=" << output.size() << "\n";
    ok = ok && output.size() == 48;
    std::cout << (ok ? "ANE_INT4_PROJECTION=PASS\n" : "ANE_INT4_PROJECTION=FAIL\n");
    ane_context_destroy(ctx);
    return ok ? 0 : 1;
}
