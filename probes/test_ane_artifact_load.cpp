#include "../runtime/rindi_native_chain.h"
#include <cstdio>
#include <vector>

int main() {
    RindiNativeChain chain(5120, 32);
    SafeTensorsLoader loader;
    if (!loader.open_file("/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors")) return 2;
    const char* path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/ane_layers";
    bool ok = true;
    for (int layer = 0; layer < 64; ++layer) {
        if (!chain.compile_layer(layer, path, loader)) {
            std::printf("failed_layer=%d\n", layer);
            ok = false;
            break;
        }
    }
    // Layer 0 is GDN: its tail consumes the already-gated recurrent core.
    std::vector<uint16_t> core(6144, 0);
    std::vector<uint16_t> residual(5120, 0x3c00);
    std::vector<uint16_t> output, projection;
    bool eval = ok && chain.evaluate_tail(0, core.data(), core.size(), residual.data(), output, &projection);
    std::printf("ANE_ARTIFACT_LOAD=%s TAIL_EVAL=%s layers=%zu output=%zu projection=%zu\n",
                ok ? "PASS" : "FAIL", eval ? "PASS" : "FAIL",
                chain.get_num_layers(), output.size(), projection.size());
    return eval ? 0 : 1;
}
