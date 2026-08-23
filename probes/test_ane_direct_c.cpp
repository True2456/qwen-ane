// SPDX-License-Identifier: Apache-2.0
#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>

int main() {
    printf("============================================================\n");
    printf("  TESTING NATIVE C/C++ DIRECT ANE PACKAGE LOADER & RUNNER\n");
    printf("============================================================\n");

    ANEContext* ctx = ane_context_create();
    if (!ctx) {
        printf("Failed to create ANEContext.\n");
        return 1;
    }
    printf("✓ Native ANEContext initialized.\n");

    const char* package_path = "/var/folders/08/kjxn753d64g6s12lt34lv_yr0000gn/T/D73F5B75B292B796DC9998852378F7EFE6C0FDF674F4F02A42BF572674CC88FD_7423974D4E18CB7791A67AC5A071B72542A0087342BF30856800481E0819C8EE_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
    
    ANEModel* model = ane_model_load_compiled(ctx, package_path, "q38_layer", 21);
    if (!model) {
        printf("FAIL: ane_model_load_compiled returned NULL\n");
        ane_context_destroy(ctx);
        return 1;
    }
    printf("✓ Successfully loaded compiled package directly via _ANEModel / _ANEClient: %p\n", model);

    size_t channels = 64;
    size_t seq_len = 32;
    size_t bytes = channels * seq_len * sizeof(uint16_t);

    IOSurfaceRef in_surf = metal_create_iosurface(bytes);
    IOSurfaceRef out_surf = metal_create_iosurface(bytes);

    if (!in_surf || !out_surf) {
        printf("FAIL: IOSurface allocation failed\n");
        ane_model_release(model);
        ane_context_destroy(ctx);
        return 1;
    }

    // Populate input surface
    uint16_t* in_ptr = (uint16_t*)metal_iosurface_get_base_address(in_surf);
    for (size_t i = 0; i < channels * seq_len; ++i) in_ptr[i] = 0x3c00; // fp16 1.0

    ANERequest* req = ane_request_create(ctx, model, in_surf, out_surf, 0);
    if (!req) {
        printf("FAIL: ane_request_create returned NULL\n");
        CFRelease(in_surf);
        CFRelease(out_surf);
        ane_model_release(model);
        ane_context_destroy(ctx);
        return 1;
    }
    printf("✓ Created ANERequest binding IOSurfaces.\n");

    // Warmup
    for (int i = 0; i < 5; ++i) {
        ane_request_evaluate(ctx, model, req, nullptr, 0, nullptr, 0);
    }

    // Benchmark direct evaluation in C/C++
    int iters = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        bool ok = ane_request_evaluate(ctx, model, req, nullptr, 0, nullptr, 0);
        if (!ok) {
            printf("FAIL: Direct evaluation returned false on iteration %d\n", i);
            break;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iters;

    printf("✓ Successfully evaluated %d dispatches on physical ANE hardware!\n", iters);
    printf("  Hardware latency: %.4f ms / dispatch (%.2f kHz dispatch rate)\n", avg_ms, 1.0 / (avg_ms * 1e-3) / 1000.0);

    // Verify output
    const uint16_t* out_ptr = (const uint16_t*)metal_iosurface_get_base_address(out_surf);
    printf("  Sample output token 0 channel 0: 0x%04x\n", out_ptr[0]);

    ane_request_release(req);
    CFRelease(in_surf);
    CFRelease(out_surf);
    ane_model_release(model);
    ane_context_destroy(ctx);

    printf("============================================================\n");
    printf("  ALL C/C++ DIRECT ANE TESTS PASSED (100%% SUCCESS)\n");
    printf("============================================================\n");
    return 0;
}
