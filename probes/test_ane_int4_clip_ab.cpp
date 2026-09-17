// SPDX-License-Identifier: Apache-2.0
// A/B: RTN scale selection vs clip-search scale selection in compile_int4,
// evaluated ON THE ANE against an fp32 reference. Same tensor, same bridge.
#include "../runtime/rindi_ane_projection.h"
#include "../runtime/ane_c_bridge.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

float bits_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000)) << 16;
    const uint32_t exp = (bits >> 10) & 0x1f;
    const uint32_t man = bits & 0x3ff;
    uint32_t v;
    if (exp == 0) {
        v = sign | (0x38000000) | (man << 13);  // approx denorm as fp32
    } else {
        v = sign | ((static_cast<uint32_t>(exp) - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors";
    SafeTensorsLoader loader;
    if (!loader.open_file(path)) {
        std::fprintf(stderr, "open failed\n");
        return 2;
    }

    ANEContext* ctx = ane_context_create();
    if (!ctx) return 2;

    const char* names[] = {
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.10.linear_attn.in_proj_a.weight",
        "layers.20.linear_attn.in_proj_a.weight",
    };

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    printf("%-40s %12s %12s %10s\n", "tensor", "rtn_rel_err", "clip_rel_err", "impr");
    bool all_ok = true;
    for (const char* tn : names) {
        const TensorInfo* info = loader.get_tensor_info(tn);
        if (!info || info->shape.size() != 2) continue;
        const size_t O = static_cast<size_t>(info->shape[0]);
        const size_t I = static_cast<size_t>(info->shape[1]) *
                         (info->dtype == "U32" ? 8u : 1u);

        std::vector<float> W(static_cast<size_t>(O) * I);
        std::vector<uint16_t> row;
        bool ok = true;
        for (size_t r = 0; r < O && ok; ++r) {
            ok = loader.get_row_fp16(tn, r, row) && row.size() == I;
            for (size_t c = 0; c < I && ok; ++c)
                W[r * I + c] = bits_to_float(row[c]);
        }
        if (!ok) { std::fprintf(stderr, "row read failed: %s\n", tn); continue; }
        std::fprintf(stderr, "%s: O=%zu I=%zu dtype=%s row0[0..4]=", tn, O, I, info->dtype.c_str());
        for (int i = 0; i < 5 && i < (int)I; ++i) std::fprintf(stderr, "%.4g ", W[i]);
        std::fprintf(stderr, "\n");

        std::vector<float> x(I);
        for (auto& v : x) v = dist(rng);
        std::vector<float> ref(O, 0.f);
        for (size_t r = 0; r < O; ++r)
            for (size_t c = 0; c < I; ++c) ref[r] += W[r * I + c] * x[c];
        float ref_nrm = 0.f;
        for (float v : ref) ref_nrm += v * v;
        ref_nrm = std::sqrt(ref_nrm);

        double errs[2] = {0, 0};
        for (int opt = 0; opt < 2; ++opt) {
            RindiAneProjection p;
            const std::string tag = std::string("ab_") + (opt ? "clip" : "rtn");
            if (!p.compile_int4(ctx, loader, tn, 32, tag, opt != 0)) {
                std::fprintf(stderr, "compile failed (%s): %s\n", tn, opt ? "clip" : "rtn");
                all_ok = false;
                break;
            }
            const size_t LANES = 32;
            std::vector<uint16_t> xh(I * LANES), out;
            for (size_t c = 0; c < I; ++c) {
                // naive fp32->fp16
                uint16_t h;
                float xf = x[c];
                // convert via round-to-nearest fp16 using bit trick
                uint32_t xb;
                std::memcpy(&xb, &xf, 4);
                uint32_t sign = (xb >> 16) & 0x8000;
                int32_t e = static_cast<int32_t>((xb >> 23) & 0xff) - 127 + 15;
                uint32_t m = xb & 0x7fffff;
                if (e <= 0) h = static_cast<uint16_t>(sign);
                else if (e >= 31) h = static_cast<uint16_t>(sign | 0x7c00);
                else h = static_cast<uint16_t>(sign | (e << 10) | (m >> 13));
                for (size_t l = 0; l < LANES; ++l) xh[c * LANES + l] = h;  // channel-major
            }
            if (!p.evaluate(xh.data(), LANES, out) || out.size() != O * LANES) {
                std::fprintf(stderr, "evaluate failed: %s\n", tn);
                all_ok = false;
                break;
            }
            double se = 0;
            for (size_t r = 0; r < O; ++r) {
                const float d = bits_to_float(out[r * 32]) - ref[r];
                se += static_cast<double>(d) * d;
            }
            errs[opt] = std::sqrt(se) / ref_nrm;
            if (opt == 0 || errs[0] > 10) {
                std::fprintf(stderr, "  opt=%d ref[0..3]=", opt);
                for (int i = 0; i < 4; ++i) std::fprintf(stderr, "%.4g ", ref[i]);
                std::fprintf(stderr, "out[0..3]=");
                for (int i = 0; i < 4; ++i) std::fprintf(stderr, "%.4g ", bits_to_float(out[i * 32]));
                std::fprintf(stderr, "\n");
            }
        }
        if (errs[0] > 0)
            printf("%-40s %12.5f %12.5f %9.1fx\n", tn, errs[0], errs[1], errs[0] / errs[1]);
    }

    ane_context_destroy(ctx);
    std::cout << (all_ok ? "ANE_INT4_CLIP_AB=DONE\n" : "ANE_INT4_CLIP_AB=ERRORS\n");
    return all_ok ? 0 : 1;
}
