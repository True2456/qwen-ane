// SPDX-License-Identifier: Apache-2.0
// Native checkpoint smoke test for the Rindi U32-int4 export format.

#include "../runtime/safetensors_loader.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>

int main(int argc, char** argv) {
    const std::string path = argc > 1
        ? argv[1]
        : "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors";

    SafeTensorsLoader loader;
    if (!loader.open_file(path)) return EXIT_FAILURE;

    const TensorInfo* info = loader.get_tensor_info(
        "layers.0.linear_attn.in_proj_qkv.weight");
    if (!info || info->dtype != "U32" || info->shape.size() != 2 ||
        info->shape[0] != 10240 || info->shape[1] != 640) {
        std::cerr << "unexpected first projection metadata\n";
        return EXIT_FAILURE;
    }

    std::vector<uint16_t> row;
    if (!loader.get_row_fp16("layers.0.linear_attn.in_proj_qkv.weight", 0, row) ||
        row.size() != 5120) {
        std::cerr << "failed to decode packed projection row\n";
        return EXIT_FAILURE;
    }

    std::vector<uint16_t> embedding;
    if (!loader.get_embedding_row_fp16(0, 5120, embedding) ||
        embedding.size() != 5120) {
        std::cerr << "failed to decode embedding row\n";
        return EXIT_FAILURE;
    }

    double emb_sq = 0.0;
    for (uint16_t value : embedding) {
        const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000u) << 16;
        const uint32_t exp = (value >> 10) & 0x1fu;
        const uint32_t mant = value & 0x3ffu;
        uint32_t bits = sign;
        if (exp == 0) bits |= mant << 13;
        else if (exp == 31) bits |= 0x7f800000u | (mant << 13);
        else bits |= (exp + 112u) << 23 | (mant << 13);
        float x; std::memcpy(&x, &bits, sizeof(x)); emb_sq += x * x;
    }
    std::cout << "CPP_SAFETENSORS=PASS tensors="
              << loader.tensor_names().size()
              << " projection_cols=" << row.size()
              << " embedding_cols=" << embedding.size()
              << " proj0=" << std::hex << row[0] << "," << row[1]
              << "," << row[2] << std::dec
              << " emb_rms0=" << std::sqrt(emb_sq / embedding.size())
              << " emb198=";
    std::vector<uint16_t> emb198;
    loader.get_embedding_row_fp16(198, 5120, emb198);
    double emb198_sq = 0.0;
    for (uint16_t value : emb198) {
        const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000u) << 16;
        const uint32_t exp = (value >> 10) & 0x1fu;
        const uint32_t mant = value & 0x3ffu;
        uint32_t bits = sign;
        if (exp == 0) bits |= mant << 13;
        else if (exp == 31) bits |= 0x7f800000u | (mant << 13);
        else bits |= (exp + 112u) << 23 | (mant << 13);
        float x; std::memcpy(&x, &bits, sizeof(x)); emb198_sq += x * x;
    }
    std::cout << std::sqrt(emb198_sq / emb198.size()) << '\n';
    return EXIT_SUCCESS;
}
