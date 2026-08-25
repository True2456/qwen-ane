#include "runtime/ane_c_bridge.h"
#include "runtime/rindi_gdn_layer.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const size_t width = argc > 1 ? static_cast<size_t>(std::strtoul(argv[1], nullptr, 10)) : 32;
    const size_t lanes = argc > 2 ? static_cast<size_t>(std::strtoul(argv[2], nullptr, 10)) : 1;
    ANEContext* ctx = ane_context_create();
    SafeTensorsLoader loader;
    const char* path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors";
    if (!ctx || !loader.open_file(path)) return 2;

    RindiGdnLayer layer;
    if (lanes > 1 ? !layer.compile_core(ctx, loader, 0, width)
                  : !layer.compile(ctx, loader, 0, width)) {
        std::fprintf(stderr, "GDN layer compile failed\n");
        ane_context_destroy(ctx);
        return 3;
    }
    std::vector<uint16_t> hidden(5120 * lanes, 0x2800); // small nonzero fp16 input
    std::vector<uint16_t> core, z;
    const bool ok = lanes > 1 ? layer.core_step(hidden.data(), lanes, core, z)
                              : layer.step(hidden.data(), 1, core);
    const size_t expected = (lanes > 1 ? 6144 * lanes : 5120);
    std::printf("GDN_LAYER=%s width=%zu lanes=%zu output=%zu\n",
                (ok && core.size() == expected) ? "PASS" : "FAIL",
                width, lanes, core.size());
    ane_context_destroy(ctx);
    return (ok && core.size() == expected) ? 0 : 1;
}
