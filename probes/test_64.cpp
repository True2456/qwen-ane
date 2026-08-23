// SPDX-License-Identifier: Apache-2.0
// Minimal native MIL compile/load/execute smoke test.

#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

static const char* kMil = R"MIL(program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
  func main<ios18>(tensor<fp16, [1, 64, 1, 32]> x) {
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [64, 64, 1, 1]> w = const()[name=string("w"), val=tensor<fp16, [64, 64, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<fp16, [1, 64, 1, 32]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("y")];
  } -> (y);
}
)MIL";

int main() {
    std::vector<uint16_t> weights(64*64, 0x3c00); // 4x4 matrix of ones
    const char* names[] = {"w.bin"};
    const void* data[] = {weights.data()};
    size_t sizes[] = {weights.size() * sizeof(uint16_t)};

    ANEContext* ane = ane_context_create();
    if (!ane) return 2;
    ANEModel* model = ane_model_compile_mil(ane, kMil, names, data, sizes, 1, 0, 21);
    if (!model) {
        std::cerr << "ANE_BRIDGE=COMPILE_FAIL\n";
        ane_context_destroy(ane);
        return 1;
    }

    IOSurfaceRef input = metal_create_iosurface(64 * 32 * sizeof(uint16_t));
    IOSurfaceRef output = metal_create_iosurface(64 * 32 * sizeof(uint16_t));
    bool ok = input && output;
    if (ok) {
        IOSurfaceLock(input, 0, nullptr);
        std::vector<uint16_t> ones(64 * 32, 0x3c00);
        std::memcpy(IOSurfaceGetBaseAddress(input), ones.data(), ones.size() * sizeof(uint16_t));
        IOSurfaceUnlock(input, 0, nullptr);
        ANERequest* request = ane_request_create(ane, model, input, output, 0);
        ok = request && ane_request_evaluate(ane, model, request, nullptr, 0, nullptr, 0);
        if (ok) {
            IOSurfaceLock(output, kIOSurfaceLockReadOnly, nullptr);
            const uint16_t* result = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(output));
            ok = result && result[0] == 0x4400; // four ones summed by the 1x1 conv
            IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, nullptr);
        }
        if (request) ane_request_release(request);
    }
    if (input) CFRelease(input);
    if (output) CFRelease(output);
    ane_model_release(model);
    ane_context_destroy(ane);
    std::cout << (ok ? "ANE_BRIDGE=PASS\n" : "ANE_BRIDGE=EXEC_FAIL\n");
    return ok ? 0 : 1;
}
