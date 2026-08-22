#include "runtime/ane_c_bridge.h"
#include "runtime/rindi_gdn_layer.h"
#include <cstdio>
#include <vector>

int main() {
    ANEContext* ctx = ane_context_create();
    SafeTensorsLoader loader;
    const char* path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors";
    if (!ctx || !loader.open_file(path)) return 2;

    RindiGdnLayer layer;
    if (!layer.compile(ctx, loader, 0, 32)) {
        std::fprintf(stderr, "GDN layer compile failed\n");
        ane_context_destroy(ctx);
        return 3;
    }
    std::vector<uint16_t> hidden(5120, 0x2800); // small nonzero fp16 input
    std::vector<uint16_t> core;
    const bool ok = layer.step(hidden.data(), 1, core);
    std::printf("GDN_LAYER=%s output=%zu\n", (ok && core.size() == 5120) ? "PASS" : "FAIL", core.size());
    ane_context_destroy(ctx);
    return (ok && core.size() == 5120) ? 0 : 1;
}
