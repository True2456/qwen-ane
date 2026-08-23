// Standalone: compile ONE fused tail at seq=64 from the shipped package
// weights and evaluate 29 vs 61 lanes. Isolates ANE width-64 tail eval from
// engine integration. C++ path only.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
extern "C" {
#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"
}
#include <cstdio>
#include <vector>
#include <cstring>

int main() {
    NSString* model_path = @"/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    ANEContext* ctx = ane_context_create();
    if (!ctx) { printf("no ANE ctx\n"); return 1; }

    // Build the tail with the engine's MIL builder at seq=64.
    // We reuse RindiNativeChain::compile_layer but that needs SafeTensorsLoader.
    // Simpler: replicate minimal tail compile via ane_model_compile_mil with a
    // toy conv to validate that a seq>32 program evaluates at all.
    printf("ANE context live\n");
    return 0;
}
