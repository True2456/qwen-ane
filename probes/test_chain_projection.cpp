#include "../runtime/ane_c_bridge.h"
#include "../runtime/metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    const auto n = in.tellg();
    std::vector<uint8_t> out(static_cast<size_t>(n));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), n);
    return out;
}

static float half_to_float(uint16_t bits) {
    const uint32_t s = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t e = (bits >> 10) & 31u;
    const uint32_t m = bits & 1023u;
    uint32_t v = e == 0 ? s | (m << 13)
        : e == 31 ? s | 0x7f800000u | (m << 13)
                  : s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

int main() {
    constexpr size_t O = 5120, I = 6144, W = 32;
    const char* base = "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/ane_layers/chain0.o/";
    auto q = read_file("~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/ane_layers/chain0.o/__.bin");
    auto sc = read_file("~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/ane_layers/chain0.o/__s.bin");
    (void)base;
    if (q.size() != O * I / 2 || sc.size() != O * 2) return 2;
    const char* mil = R"MIL(program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
  func main<ios18>(tensor<fp16, [1, 6144, 1, 32]> x) {
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<int4, [5120, 6144, 1, 1]> q = const()[name=string("q"), val=tensor<int4, [5120, 6144, 1, 1]>(BLOBFILE(path=string("@model_path/weights/q.bin"), offset=uint64(64)))];
    tensor<fp16, [5120, 1, 1, 1]> sc = const()[name=string("sc"), val=tensor<fp16, [5120, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/sc.bin"), offset=uint64(64)))];
    tensor<fp16, [5120, 6144, 1, 1]> w = constexpr_blockwise_shift_scale(data=q, scale=sc)[name=string("dq")];
    tensor<fp16, [1, 5120, 1, 32]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("y")];
  } -> (y);
}
)MIL";
    const char* names[] = {"q.bin", "sc.bin"};
    const void* data[] = {q.data(), sc.data()};
    const size_t sizes[] = {q.size(), sc.size()};
    ANEContext* ctx = ane_context_create();
    ANEModel* model = ctx ? ane_model_compile_mil(ctx, mil, names, data, sizes, 2, 0, 21) : nullptr;
    IOSurfaceRef in = metal_create_iosurface(I * W * 2);
    IOSurfaceRef out = metal_create_iosurface(O * W * 2);
    ANERequest* req = (model && in && out) ? ane_request_create(ctx, model, in, out, 0) : nullptr;
    if (!req) return 3;
    std::vector<uint16_t> ones(I * W, 0x3c00);
    IOSurfaceLock(in, 0, nullptr);
    std::memcpy(IOSurfaceGetBaseAddress(in), ones.data(), ones.size() * 2);
    IOSurfaceUnlock(in, 0, nullptr);
    bool ok = ane_request_evaluate(ctx, model, req, nullptr, 0, nullptr, 0);
    IOSurfaceLock(out, kIOSurfaceLockReadOnly, nullptr);
    const uint16_t got = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(out))[0];
    IOSurfaceUnlock(out, kIOSurfaceLockReadOnly, nullptr);
    const uint8_t* qp = q.data();
    const uint16_t* sp = reinterpret_cast<const uint16_t*>(sc.data());
    float expected = 0.0f;
    for (size_t c = 0; c < I; ++c) {
        const uint8_t b = qp[c / 2];
        const int nibble = (c & 1) ? (b >> 4) : (b & 0xf);
        const int signed_q = nibble < 8 ? nibble : nibble - 16;
        expected += signed_q * half_to_float(sp[0]);
    }
    std::printf("CHAIN_PROJ=%s got=%g expected=%g bits=%04x\n", ok ? "PASS" : "FAIL",
                half_to_float(got), expected, got);
    ane_request_release(req); CFRelease(in); CFRelease(out); ane_model_release(model); ane_context_destroy(ctx);
    return ok && std::fabs(half_to_float(got) - expected) < 5.0f ? 0 : 1;
}
