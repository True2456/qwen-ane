// SPDX-License-Identifier: Apache-2.0
#include "../runtime/ane_c_bridge.h"
#include "../runtime/metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    const char* mil = R"MIL(program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
  func main<ios18>(tensor<fp16, [1, 4, 1, 32]> x) {
    tensor<fp16, [1, 4, 1, 32]> y = mul(x=x, y=fp16(0x1p+1))[name=string("y")];
    tensor<fp16, [1, 4, 1, 32]> z = add(x=y, y=fp16(0x0p+0))[name=string("z")];
  } -> (y, z);
}
)MIL";
    std::vector<uint16_t> empty;
    ANEContext* ctx = ane_context_create();
    if (!ctx) return 2;
    ANEModel* model = ane_model_compile_mil(ctx, mil, nullptr, nullptr, nullptr, 0, 0, 21);
    IOSurfaceRef input = metal_create_iosurface(4 * 32 * sizeof(uint16_t));
    IOSurfaceRef output_a = metal_create_iosurface(4 * 32 * sizeof(uint16_t));
    IOSurfaceRef output_b = metal_create_iosurface(4 * 32 * sizeof(uint16_t));
    IOSurfaceRef outputs[] = {output_a, output_b};
    bool ok = model && input && output_a && output_b;
    if (ok) {
        std::vector<uint16_t> ones(4 * 32, 0x3c00);
        IOSurfaceLock(input, 0, nullptr);
        std::memcpy(IOSurfaceGetBaseAddress(input), ones.data(), ones.size() * sizeof(uint16_t));
        IOSurfaceUnlock(input, 0, nullptr);
        ANERequest* request = ane_request_create_multi(ctx, model, input, outputs, 2, 0);
        ok = request && ane_request_evaluate(ctx, model, request, nullptr, 0, nullptr, 0);
        for (IOSurfaceRef surface : outputs) {
            IOSurfaceLock(surface, kIOSurfaceLockReadOnly, nullptr);
            const uint16_t* values = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(surface));
            ok = ok && values && values[0] == 0x4000;
            IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, nullptr);
        }
        if (request) ane_request_release(request);
    }
    if (input) CFRelease(input);
    if (output_a) CFRelease(output_a);
    if (output_b) CFRelease(output_b);
    if (model) ane_model_release(model);
    ane_context_destroy(ctx);
    std::cout << (ok ? "ANE_MULTI_OUTPUT=PASS\n" : "ANE_MULTI_OUTPUT=FAIL\n");
    return ok ? 0 : 1;
}
