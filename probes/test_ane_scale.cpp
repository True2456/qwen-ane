// SPDX-License-Identifier: Apache-2.0
#include "../runtime/ane_c_bridge.h"
#include "../runtime/metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

static std::string make_test_mil(size_t dim, size_t width) {
    std::ostringstream s;
    s << "program(1.3)\n"
      << "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
      << "{\"coremlc-version\", \"3505.4.1\"}, "
      << "{\"coremltools-component-milinternal\", \"\"}, "
      << "{\"coremltools-version\", \"9.0\"}})]\n"
      << "{\n"
      << "  func main<ios18>(tensor<fp16, [1, " << dim << ", 1, " << width << "]> x) {\n"
      << "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
      << "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
      << "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
      << "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
      << "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
      << "    tensor<fp16, [" << dim << ", " << dim << ", 1, 1]> w = const()[name=string(\"w\"), val=tensor<fp16, [" << dim << ", " << dim << ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [1, " << dim << ", 1, " << width << "]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"y\")];\n"
      << "  } -> (y);\n"
      << "}\n";
    return s.str();
}

int main() {
    ANEContext* ane = ane_context_create();
    if (!ane) return 2;

    for (size_t dim : {4, 64, 128, 512, 1024}) {
        std::vector<uint16_t> weights(dim * dim, 0x3c00);
        const char* names[] = {"w.bin"};
        const void* data[] = {weights.data()};
        size_t sizes[] = {weights.size() * sizeof(uint16_t)};
        std::string mil = make_test_mil(dim, 32);

        ANEModel* model = ane_model_compile_mil(ane, mil.c_str(), names, data, sizes, 1, 0, 21);
        std::cout << "dim=" << dim << " compiled=" << (model != nullptr) << "\n";
        if (model) ane_model_release(model);
    }
    ane_context_destroy(ane);
    return 0;
}
