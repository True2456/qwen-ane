/*
 * SPDX-License-Identifier: Apache-2.0
 * test_pure_native.cpp - Test pure C/C++ Metal + ANE hardware interface (No Python, No MLX).
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "../runtime/metal_engine.h"
#include "../runtime/ane_c_bridge.h"

int main() {
    std::cout << "=================================================================" << std::endl;
    std::cout << "  TESTING PURE C/C++ NATIVE ENGINE (ZERO MLX, ZERO PYTHON)" << std::endl;
    std::cout << "=================================================================" << std::endl;

    // 1. Initialize Metal Context
    MetalContext* metal = metal_context_create();
    if (!metal) {
        std::cerr << "Failed to create MetalContext" << std::endl;
        return 1;
    }
    std::cout << "  [Metal C Runtime] Active Device: " << metal_get_device_name(metal) << std::endl;

    // 2. Initialize ANE Context
    ANEContext* ane = ane_context_create();
    if (!ane) {
        std::cerr << "Failed to create ANEContext" << std::endl;
        return 1;
    }
    std::cout << "  [ANE C Bridge] Direct _ANEClient connection established" << std::endl;

    // 3. Create Zero-Copy IOSurface Shared Buffer
    size_t nbytes = 32 * 5120 * sizeof(uint16_t); // 32 tokens x 5120 channels in FP16
    IOSurfaceRef surface = metal_create_iosurface(nbytes);
    if (!surface) {
        std::cerr << "Failed to create IOSurface" << std::endl;
        return 1;
    }
    std::cout << "  [IOSurface] Allocated " << (nbytes / 1024.0) << " KB zero-copy unified memory" << std::endl;

    // 4. Wrap into Metal Buffer
    MetalBufferHandle mtl_buf = metal_buffer_from_iosurface(metal, surface);
    if (!mtl_buf) {
        std::cerr << "Failed to wrap IOSurface in Metal buffer" << std::endl;
        return 1;
    }

    // 5. Create Hardware MTLSharedEvent
    MetalSharedEventHandle evt = metal_shared_event_create(metal);
    std::cout << "  [Hardware Sync] MTLSharedEvent handle created" << std::endl;

    metal_shared_event_set_value(evt, 42);
    uint64_t val = metal_shared_event_get_value(evt);
    if (val != 42) {
        std::cerr << "Event value mismatch: expected 42, got " << val << std::endl;
        return 1;
    }
    std::cout << "  [Hardware Sync] Signal verified (val = " << val << ")" << std::endl;

    // Clean up
    metal_shared_event_release(evt);
    metal_buffer_release(mtl_buf);
    ane_context_destroy(ane);
    metal_context_destroy(metal);

    std::cout << "=================================================================" << std::endl;
    std::cout << "  ✓ ALL PURE C/C++ NATIVE CHECKS PASSED WITH ZERO DEPENDENCIES!" << std::endl;
    std::cout << "=================================================================" << std::endl;

    return 0;
}
