#include "runtime/ane_c_bridge.h"
#include "runtime/rindi_attention.h"
#include <cstdio>
#include <vector>

int main() {
    ANEContext* ctx = ane_context_create();
    SafeTensorsLoader loader;
    const char* path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors";
    if (!ctx || !loader.open_file(path)) return 2;
    RindiAttention attention;
    if (!attention.compile(ctx, loader, 3, 16, 32)) return 3;
    std::vector<uint16_t> hidden(5120, 0x2800), output;
    bool ok = attention.step(hidden.data(), 1, output);
    bool ok2 = attention.step(hidden.data(), 1, output);
    std::printf("ATTENTION_KV=%s output=%zu position=%zu\n",
                (ok && ok2 && output.size() == 5120) ? "PASS" : "FAIL",
                output.size(), attention.position());
    ane_context_destroy(ctx);
    return (ok && ok2 && output.size() == 5120) ? 0 : 1;
}
