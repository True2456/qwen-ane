// SPDX-License-Identifier: Apache-2.0
// EXL3-mechanism compile gate: can the ANE compiler accept the op classes a
// trellis/LUT format (EXL3/QTIP/AWQ-palette) needs? Driven through the same
// ane_model_compile_mil bridge the engine uses, not the Python prototype.
//
// Cases:
//   1. perchannel_int4  — positive control: the engine's exact int4 spelling.
//   2. blockwise_gs     — group-wise scales [O, n_groups, 1, 1], n_groups=2
//                         (gs=64-like granularity). EXL3/GGUF/AWQ all need
//                         sub-channel granularity.
//   3. lut_a            — constexpr_lut_to_dense(lut=, indices=) palettized
//                         dequant, the EXL3/QTIP core mechanism.
//   4. lut_b            — same op, alternate arity (x=, lut=, indices=).
//   5. gather_lut       — runtime-indexed codebook lookup (gather), the other
//                         way to spell a LUT.

#include "../runtime/ane_c_bridge.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr size_t kO = 64;
constexpr size_t kI = 128;  // divisible by 64 so n_groups=2 is a real gs=64
constexpr size_t kS = 32;

constexpr const char* kBuildInfo =
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

std::string prologue() {
    return "program(1.3)\n" + std::string(kBuildInfo) + "\n{\n"
        "  func main<ios18>(tensor<fp16, [1, 128, 1, 32]> x) {\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
}

std::string tail(const std::string& out_expr) {
    return "    tensor<fp16, [1, 64, 1, 32]> y = " + out_expr +
           "[name=string(\"y\")];\n  } -> (y);\n}\n";
}

struct Case {
    std::string name;
    std::string mil;
    std::vector<std::string> weight_names;
    std::vector<std::vector<uint8_t>> payloads;
};

uint16_t fp16_bits(int i) { return static_cast<uint16_t>(0x3800 + i); }

std::vector<Case> make_cases() {
    std::vector<Case> cases;

    {
        Case c;
        c.name = "perchannel_int4_positive_control";
        std::vector<uint8_t> q(kO * kI / 2, 0x08);          // nibbles 0,-8
        std::vector<uint8_t> sc(kO * sizeof(uint16_t), 0);
        for (size_t r = 0; r < kO; ++r) {
            uint16_t one = 0x3c00;                           // fp16 1.0
            std::memcpy(sc.data() + r * 2, &one, 2);
        }
        c.mil = prologue() +
            "    tensor<int4, [64, 128, 1, 1]> q = const()[name=string(\"q\"), val=tensor<int4, [64, 128, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/q.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [64, 1, 1, 1]> sc = const()[name=string(\"sc\"), val=tensor<fp16, [64, 1, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/sc.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [64, 128, 1, 1]> w = constexpr_blockwise_shift_scale(data=q, scale=sc)[name=string(\"dq\")];\n" +
            tail("conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)");
        c.weight_names = {"q.bin", "sc.bin"};
        c.payloads = {q, sc};
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "blockwise_gs64_scales";
        std::vector<uint8_t> q(kO * kI / 2, 0x08);
        std::vector<uint8_t> sc(kO * 2 * sizeof(uint16_t), 0);   // [O, 2, 1, 1]
        for (size_t i = 0; i < kO * 2; ++i) {
            uint16_t one = 0x3c00;
            std::memcpy(sc.data() + i * 2, &one, 2);
        }
        c.mil = prologue() +
            "    tensor<int4, [64, 128, 1, 1]> q = const()[name=string(\"q\"), val=tensor<int4, [64, 128, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/q.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [64, 2, 1, 1]> sc = const()[name=string(\"sc\"), val=tensor<fp16, [64, 2, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/sc.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [64, 128, 1, 1]> w = constexpr_blockwise_shift_scale(data=q, scale=sc)[name=string(\"dq\")];\n" +
            tail("conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)");
        c.weight_names = {"q.bin", "sc.bin"};
        c.payloads = {q, sc};
        cases.push_back(c);
    }
    for (int variant = 0; variant < 2; ++variant) {
        Case c;
        c.name = variant == 0 ? "lut_palettized_a (lut=, indices=)"
                              : "lut_palettized_b (x=, lut=, indices=)";
        std::vector<uint8_t> idx(kO * kI / 2, 0x01);             // 4-bit indices
        std::vector<uint8_t> lut(16 * sizeof(uint16_t), 0);
        for (size_t i = 0; i < 16; ++i) {
            uint16_t v = fp16_bits(static_cast<int>(i));
            std::memcpy(lut.data() + i * 2, &v, 2);
        }
        std::string call = variant == 0
            ? "constexpr_lut_to_dense(lut=lut, indices=idx)"
            : "constexpr_lut_to_dense(x=idx, lut=lut, indices=idx)";
        c.mil = prologue() +
            "    tensor<int4, [64, 128, 1, 1]> idx = const()[name=string(\"idx\"), val=tensor<int4, [64, 128, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/idx.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [16, 1, 1, 1]> lut = const()[name=string(\"lut\"), val=tensor<fp16, [16, 1, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/lut.bin\"), offset=uint64(64)))];\n"
            "    tensor<fp16, [64, 128, 1, 1]> w = " + call + "[name=string(\"dq\")];\n" +
            tail("conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)");
        c.weight_names = {"idx.bin", "lut.bin"};
        c.payloads = {idx, lut};
        cases.push_back(c);
    }
    {
        Case c;
        c.name = "gather_codebook_lookup";
        std::vector<uint8_t> cb(16 * sizeof(uint16_t), 0);
        for (size_t i = 0; i < 16; ++i) {
            uint16_t v = fp16_bits(static_cast<int>(i));
            std::memcpy(cb.data() + i * 2, &v, 2);
        }
        std::vector<uint8_t> gidx(kO * kS * sizeof(int32_t), 0);
        for (size_t i = 0; i < kO * kS; ++i) {
            int32_t v = 3;
            std::memcpy(gidx.data() + i * 4, &v, 4);
        }
        c.mil = prologue() +
            "    tensor<fp16, [16]> cb = const()[name=string(\"cb\"), val=tensor<fp16, [16]>(BLOBFILE(path=string(\"@model_path/weights/cb.bin\"), offset=uint64(64)))];\n"
            "    tensor<int32, [1, 64, 1, 32]> gidx = const()[name=string(\"gidx\"), val=tensor<int32, [1, 64, 1, 32]>(BLOBFILE(path=string(\"@model_path/weights/gidx.bin\"), offset=uint64(64)))];\n" +
            tail("gather(x=cb, indices=gidx, axis=0)");
        c.weight_names = {"cb.bin", "gidx.bin"};
        c.payloads = {cb, gidx};
        cases.push_back(c);
    }
    return cases;
}

}  // namespace

int main() {
    std::vector<Case> cases = make_cases();

    ANEContext* ane = ane_context_create();
    if (!ane) {
        std::cerr << "ANE_FORMAT_GATE=CONTEXT_FAIL\n";
        return 2;
    }

    std::cout << "ANE_FORMAT_GATE_BEGIN cases=" << cases.size() << "\n";
    size_t pass = 0, fail = 0;

    for (const Case& c : cases) {
        std::vector<const char*> names;
        std::vector<const void*> data;
        std::vector<size_t> sizes;
        for (size_t i = 0; i < c.weight_names.size(); ++i) {
            names.push_back(c.weight_names[i].c_str());
            data.push_back(c.payloads[i].data());
            sizes.push_back(c.payloads[i].size());
        }
        ANEModel* model = ane_model_compile_mil(
            ane, c.mil.c_str(), names.data(), data.data(), sizes.data(),
            names.size(), 0, 21);
        if (model) {
            ++pass;
            std::cout << "GATE " << c.name << " PASS\n";
            ane_model_release(model);
        } else {
            ++fail;
            std::cout << "GATE " << c.name << " FAIL\n";
        }
    }

    std::cout << "ANE_FORMAT_GATE_END pass=" << pass << " fail=" << fail << "\n";
    ane_context_destroy(ane);
    return 0;
}
