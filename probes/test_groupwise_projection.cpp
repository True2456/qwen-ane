#include "runtime/rindi_ane_projection.h"
#include <cstdio>

int main() {
    SafeTensorsLoader loader;
    if (!loader.open_file("/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors")) return 2;
    RindiAneProjection projection;
    if (!projection.compile_int4_host(loader, "layers.0.linear_attn.in_proj_qkv.weight")) return 3;
    std::vector<uint16_t> input;
    if (!loader.get_embedding_row_fp16(20206, 5120, input)) return 4;
    std::vector<uint16_t> output;
    if (!projection.evaluate(input.data(), 1, output)) return 5;
    std::printf("GROUPWISE_PROJECTION=%s size=%zu first=%04x\n",
                output.size() == 10240 ? "PASS" : "FAIL", output.size(), output[0]);
    return output.size() == 10240 ? 0 : 1;
}
