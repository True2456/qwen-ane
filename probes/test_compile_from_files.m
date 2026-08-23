#import <Foundation/Foundation.h>
#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"
#include <stdio.h>

int main() {
    @autoreleasepool {
        ANEContext* ctx = ane_context_create();
        if (!ctx) { printf("Failed context\n"); return 1; }
        
        NSError* err = nil;
        NSString* mil = [NSString stringWithContentsOfFile:@"/tmp/working_mil.txt" encoding:NSUTF8StringEncoding error:&err];
        NSData* w_data = [NSData dataWithContentsOfFile:@"/tmp/working_w.bin"];
        
        printf("MIL loaded: %lu bytes\n", (unsigned long)[mil length]);
        printf("Weight loaded: %lu bytes\n", (unsigned long)[w_data length]);
        
        const char* names[] = {"w.bin"};
        const void* data[] = {[w_data bytes]};
        size_t sizes[] = {[w_data length]};
        
        ANEModel* model = ane_model_compile_mil(ctx, [mil UTF8String], names, data, sizes, 1, 0, 21);
        printf("Model compile result: %p\n", model);
        
        if (model) {
            printf("SUCCESS! Native C API compiled model!\n");
            ane_model_release(model);
        } else {
            printf("FAIL: Compilation returned NULL\n");
        }
        ane_context_destroy(ctx);
    }
    return 0;
}
