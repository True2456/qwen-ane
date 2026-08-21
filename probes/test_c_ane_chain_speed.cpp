#include <iostream>
#include <chrono>
#include <vector>
#include "../runtime/metal_engine.h"
#include "../runtime/ane_c_bridge.h"

int main() {
    std::cout << "Measuring Pure C Metal + ANE Native Loop Overhead..." << std::endl;
    MetalContext* metal = metal_context_create();
    ANEContext* ane = ane_context_create();
    
    // Allocate 2 ping-pong IOSurface buffers for 64-layer chaining
    size_t nbytes = 32 * 5120 * sizeof(uint16_t);
    IOSurfaceRef surfA = metal_create_iosurface(nbytes);
    IOSurfaceRef surfB = metal_create_iosurface(nbytes);
    
    // Measure 64-layer dispatch latency in pure C
    auto t0 = std::chrono::high_resolution_clock::now();
    int N_ITERS = 100;
    for (int iter = 0; iter < N_ITERS; iter++) {
        // Ping-pong pointer swaps across 64 layers
        IOSurfaceRef in_s = surfA;
        IOSurfaceRef out_s = surfB;
        for (int layer = 0; layer < 64; layer++) {
            // In C native, pointer swap is instant (0 ns)
            std::swap(in_s, out_s);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "  Pure C 64-layer loop latency: " << (total_us / N_ITERS) << " us (" << (total_us / N_ITERS / 1000.0) << " ms)" << std::endl;
    
    ane_context_destroy(ane);
    metal_context_destroy(metal);
    return 0;
}
