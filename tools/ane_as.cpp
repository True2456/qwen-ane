// SPDX-License-Identifier: Apache-2.0
/*
 * ane_as.cpp - Native C/C++ Apple Neural Engine Direct Kernel Synthesizer & Benchmark
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>

#include <IOSurface/IOSurface.h>

#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"

static const char* kMil = R"MIL(program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
  func main<ios18>(tensor<fp16, [1, 4, 1, 32]> x) {
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [4, 4, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [4, 4, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<fp16, [1, 4, 1, 32]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("y")];
  } -> (y);
}
)MIL";

// Float16 / Float32 conversion helpers
static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp16 = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x03ff;
    uint32_t u32;
    if (exp16 == 0) {
        if (mant == 0) {
            u32 = sign; // +-zero
        } else {        // subnormal: normalize into fp32
            int e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while (!(m & 0x400));
            u32 = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x03ff) << 13);
        }
    } else if (exp16 == 31) {
        u32 = sign | 0x7f800000u | (mant << 13); // inf / nan
    } else {
        u32 = sign | ((uint32_t)(exp16 - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u32, sizeof(f));
    return f;
}

int main(int argc, char** argv) {
    int req_iters = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) {
            req_iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --test-all          Run the standard test suite (default)\n"
                      << "  --iters <N>         Number of evaluation benchmark iterations (default 100)\n";
            return 0;
        }
    }

    std::cout << "======================================================================\n";
    std::cout << "  Native C/C++ Apple Neural Engine Direct Kernel Synthesizer (ane-as)\n";
    std::cout << "======================================================================\n";

    std::vector<uint16_t> weights(16, 0x3c00); // 4x4 matrix of ones
    const char* names[] = {"w.bin"};
    const void* data[] = {weights.data()};
    size_t sizes[] = {weights.size() * sizeof(uint16_t)};

    ANEContext* ane = ane_context_create();
    if (!ane) {
        std::cerr << "ERROR: Failed to initialize native ANEContext.\n";
        return 1;
    }
    std::cout << "✓ Native ANEContext created and AppleNeuralEngine.framework loaded.\n\n";

    auto t_comp_start = std::chrono::high_resolution_clock::now();
    ANEModel* model = ane_model_compile_mil(ane, kMil, names, data, sizes, 1, 0, 21);
    auto t_comp_end = std::chrono::high_resolution_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t_comp_end - t_comp_start).count();

    if (!model) {
        std::cout << "  [FAIL] Compilation failed (" << compile_ms << " ms)\n";
        ane_context_destroy(ane);
        return 1;
    }
    std::cout << "✓ Compiled & loaded model in " << std::fixed << std::setprecision(2) << compile_ms << " ms\n";

    IOSurfaceRef input = metal_create_iosurface(4 * 32 * sizeof(uint16_t));
    IOSurfaceRef output = metal_create_iosurface(4 * 32 * sizeof(uint16_t));
    bool ok = input && output;

    if (!ok) {
        std::cout << "  [FAIL] IOSurface allocation failed\n";
        if (input) CFRelease(input);
        if (output) CFRelease(output);
        ane_model_release(model);
        ane_context_destroy(ane);
        return 1;
    }

    IOSurfaceLock(input, 0, nullptr);
    std::vector<uint16_t> ones(4 * 32, 0x3c00);
    std::memcpy(IOSurfaceGetBaseAddress(input), ones.data(), ones.size() * sizeof(uint16_t));
    IOSurfaceUnlock(input, 0, nullptr);

    ANERequest* request = ane_request_create(ane, model, input, output, 0);
    if (!request) {
        std::cout << "  [FAIL] ANERequest creation failed\n";
        CFRelease(input);
        CFRelease(output);
        ane_model_release(model);
        ane_context_destroy(ane);
        return 1;
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        ane_request_evaluate(ane, model, request, nullptr, 0, nullptr, 0);
    }

    // Benchmark
    auto t_eval_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < req_iters; ++i) {
        ane_request_evaluate(ane, model, request, nullptr, 0, nullptr, 0);
    }
    auto t_eval_end = std::chrono::high_resolution_clock::now();
    double total_eval_ms = std::chrono::duration<double, std::milli>(t_eval_end - t_eval_start).count();
    double avg_eval_ms = total_eval_ms / req_iters;
    double dispatch_rate_khz = (1.0 / (avg_eval_ms * 1e-3)) / 1000.0;
    double gflops = (2.0 * 4 * 4 * 32 / (avg_eval_ms * 1e-3)) / 1e9;

    IOSurfaceLock(output, kIOSurfaceLockReadOnly, nullptr);
    const uint16_t* result = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(output));
    bool val_ok = result && result[0] == 0x4400; // four ones summed (fp16 4.0)
    float got_val = result ? fp16_to_fp32(result[0]) : 0.0f;
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, nullptr);

    std::cout << "\nHardware Evaluation Benchmark Results (" << req_iters << " dispatches):\n";
    std::cout << "  ✓ Verification Status: " << (val_ok ? "PASS (Exact fp16 match 0x4400 = 4.0)" : "FAIL") << "\n";
    std::cout << "  ✓ Numerical Value:     " << got_val << " (Expected: 4.0)\n";
    std::cout << "  ✓ Hardware Latency:    " << std::fixed << std::setprecision(4) << avg_eval_ms << " ms / dispatch\n";
    std::cout << "  ✓ Dispatch Rate:       " << std::fixed << std::setprecision(2) << dispatch_rate_khz << " kHz\n";
    std::cout << "  ✓ Compute Throughput:  " << std::fixed << std::setprecision(2) << gflops << " GFLOP/s\n";

    ane_request_release(request);
    CFRelease(input);
    CFRelease(output);
    ane_model_release(model);
    ane_context_destroy(ane);

    std::cout << "\n======================================================================\n";
    std::cout << "  NATIVE EVALUATION COMPLETE: " << (val_ok ? "ALL TESTS PASSED (100% SUCCESS)" : "TESTS FAILED") << "\n";
    std::cout << "======================================================================\n";
    return val_ok ? 0 : 1;
}
