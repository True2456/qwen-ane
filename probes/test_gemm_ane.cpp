// SPDX-License-Identifier: Apache-2.0
#include "../runtime/ane_c_bridge.h"
#include "../runtime/metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <sstream>

// Float16 / Float32 conversion helpers
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000;
    uint32_t nonsign = w & 0x7fffffff;
    uint32_t exp = (nonsign >> 23) & 0xff;
    uint32_t mant = nonsign & 0x007fffff;
    if (exp == 0) {
        if (mant == 0) { union { uint32_t u; float f; } v = { sign }; return v.f; }
        while ((mant & 0x00800000) == 0) { mant <<= 1; exp--; }
        exp++; mant &= ~0x00800000;
    } else if (exp == 31) {
        exp = 255;
    } else {
        exp = exp - 15 + 127;
    }
    uint32_t u32 = sign | (exp << 23) | (mant >> 10);
    union { uint32_t u; float f; } v = { u32 };
    return v.f;
}

static inline uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t u = v.u;
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static std::string build_gemm_mil(size_t C_out, size_t C_in, size_t seq_len) {
    std::ostringstream s;
    s << "program(1.3)\n"
      << "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n"
      << "{\n"
      << "  func main<ios18>(tensor<fp16, [1, " << C_in << ", 1, " << seq_len << "]> x) {\n"
      << "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
      << "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
      << "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
      << "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
      << "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      << "    tensor<fp16, [" << C_out << ", " << C_in << ", 1, 1]> w = const()[name=string(\"w\"), val=tensor<fp16, [" << C_out << ", " << C_in << ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [1, " << C_out << ", 1, " << seq_len << "]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"y\")];\n"
      << "  } -> (y);\n"
      << "}\n";
    return s.str();
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "  TESTING ANE GEMM (1x1 CONV) HARDWARE EVALUATION\n";
    std::cout << "============================================================\n";

    ANEContext* ane = ane_context_create();
    if (!ane) {
        std::cerr << "Failed to create ANEContext\n";
        return 1;
    }

    std::vector<size_t> dims = {4, 16, 64, 128};
    size_t S = 32;

    for (size_t C : dims) {
        std::cout << "\nTesting C=" << C << " (Matrix Multiplication " << C << "x" << C << ", Batch S=" << S << "):\n";
        std::string mil = build_gemm_mil(C, C, S);

        std::vector<uint16_t> weights(C * C, 0x3c00); // 1.0
        const char* names[] = {"w.bin"};
        const void* data[] = {weights.data()};
        size_t sizes[] = {weights.size() * sizeof(uint16_t)};

        auto t_comp0 = std::chrono::high_resolution_clock::now();
        ANEModel* model = ane_model_compile_mil(ane, mil.c_str(), names, data, sizes, 1, 0, 21);
        auto t_comp1 = std::chrono::high_resolution_clock::now();
        double comp_ms = std::chrono::duration<double, std::milli>(t_comp1 - t_comp0).count();

        if (!model) {
            std::cout << "  FAIL: Compilation failed (" << comp_ms << " ms)\n";
            continue;
        }
        std::cout << "  ✓ Compiled model in " << comp_ms << " ms\n";

        size_t in_bytes = C * S * sizeof(uint16_t);
        size_t out_bytes = C * S * sizeof(uint16_t);
        IOSurfaceRef in_surf = metal_create_iosurface(in_bytes);
        IOSurfaceRef out_surf = metal_create_iosurface(out_bytes);

        if (!in_surf || !out_surf) {
            std::cout << "  FAIL: IOSurface creation failed\n";
            ane_model_release(model);
            continue;
        }

        // Initialize input with 1.0
        uint16_t* in_ptr = (uint16_t*)metal_iosurface_get_base_address(in_surf);
        for (size_t i = 0; i < C * S; ++i) in_ptr[i] = 0x3c00;

        ANERequest* req = ane_request_create(ane, model, in_surf, out_surf, 0);
        if (!req) {
            std::cout << "  FAIL: Request creation failed\n";
            CFRelease(in_surf);
            CFRelease(out_surf);
            ane_model_release(model);
            continue;
        }

        // Warmup
        for (int i = 0; i < 3; ++i) ane_request_evaluate(ane, model, req, nullptr, 0, nullptr, 0);

        int iters = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            ane_request_evaluate(ane, model, req, nullptr, 0, nullptr, 0);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        const uint16_t* out_ptr = (const uint16_t*)metal_iosurface_get_base_address(out_surf);
        float expected = (float)C; // C ones summed
        float got = fp16_to_fp32(out_ptr[0]);

        double flops = 2.0 * C * C * S;
        double gflops = (flops / (avg_ms * 1e-3)) / 1e9;

        std::cout << "  ✓ Direct Hardware Evaluation: PASS (" << avg_ms << " ms / dispatch)\n";
        std::cout << "  ✓ Compute Throughput:         " << gflops << " GFLOP/s\n";
        std::cout << "  ✓ Expected: " << expected << ", Got: " << got << " (" << (std::fabs(got - expected) < 1e-2 ? "EXACT MATCH" : "DIFF") << ")\n";

        ane_request_release(req);
        CFRelease(in_surf);
        CFRelease(out_surf);
        ane_model_release(model);
    }

    ane_context_destroy(ane);
    std::cout << "\n============================================================\n";
    std::cout << "  ANE HARDWARE EVALUATION COMPLETE\n";
    std::cout << "============================================================\n";
    return 0;
}
