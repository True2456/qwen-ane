// SPDX-License-Identifier: Apache-2.0
#include "rindi_gdn_recurrence.h"
#include "metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace {
constexpr const char* kBuildInfo =
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

std::string slice(const char* name, size_t c0, size_t c1, size_t w0, size_t w1) {
    return "    tensor<fp16, [1, " + std::to_string(c1-c0) + ", 1, " + std::to_string(w1-w0) + "]> " + name +
        " = slice_by_index(begin=tensor<int32, [4]>([0," + std::to_string(c0) + ",0," + std::to_string(w0) + "]), end=tensor<int32, [4]>([1," + std::to_string(c1) + ",1," + std::to_string(w1) + "]), x=x)[name=string(\"" + name + "\")];\n";
}

MetalContext* shared_recurrence_metal_context() {
    static MetalContext* context = metal_context_create();
    return context;
}
}

RindiGdnRecurrence::~RindiGdnRecurrence() {
    if (metal_state_) metal_buffer_release(metal_state_);
    if (metal_decay_) metal_buffer_release(metal_decay_);
    if (metal_key_) metal_buffer_release(metal_key_);
    if (metal_query_) metal_buffer_release(metal_query_);
    if (metal_value_) metal_buffer_release(metal_value_);
    if (metal_beta_) metal_buffer_release(metal_beta_);
    if (metal_output_) metal_buffer_release(metal_output_);
    if (request_) ane_request_release(request_);
    if (model_) ane_model_release(model_);
    if (input_surface_) CFRelease(input_surface_);
    if (output_surface_) CFRelease(output_surface_);
    if (state_surface_) CFRelease(state_surface_);
}

bool RindiGdnRecurrence::compile_prepared(ANEContext* ctx, size_t heads,
                                          size_t key_dim, size_t value_dim,
                                          size_t width) {
    ctx_ = ctx; heads_ = heads; key_dim_ = key_dim; value_dim_ = value_dim;
    hidden_keys_ = heads_ * key_dim_;
    input_channels_ = hidden_keys_ + 2 * heads_;
    width_ = width;
    const size_t H = heads_, D = key_dim_, V = value_dim_, HK = hidden_keys_;
    std::vector<uint16_t> sum(H * D, 0x3c00);
    std::vector<uint16_t> repeat(HK, 0x3c00);
    const char* names[] = {"sum.bin", "repeat.bin"};
    const void* data[] = {sum.data(), repeat.data()};
    const size_t sizes[] = {sum.size() * 2, repeat.size() * 2};
    std::string mil = "program(1.3)\n" + std::string(kBuildInfo) +
        "\n{\n  func main<ios18>(tensor<fp16, [1, " +
        std::to_string(input_channels_) + ", 1, " + std::to_string(width_) +
        "]> x) {\n";
    mil += "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n";
    mil += "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n";
    mil += "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    mil += "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n";
    mil += "    int32 gh = const()[name=string(\"gh\"), val=int32(" + std::to_string(H) + ")];\n";
    mil += "    tensor<fp16, [" + std::to_string(H) + "," + std::to_string(D) + ",1,1]> gsum = const()[name=string(\"gsum\"), val=tensor<fp16, [" + std::to_string(H) + "," + std::to_string(D) + ",1,1]>(BLOBFILE(path=string(\"@model_path/weights/sum.bin\"), offset=uint64(64)))];\n";
    mil += "    tensor<fp16, [" + std::to_string(HK) + ",1,1,1]> grep = const()[name=string(\"grep\"), val=tensor<fp16, [" + std::to_string(HK) + ",1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/repeat.bin\"), offset=uint64(64)))];\n";
    mil += slice("stt", 0, HK, 0, V) + slice("dcy", 0, HK, V, V + 1) +
           slice("kk", 0, HK, V + 1, V + 2) + slice("qq", 0, HK, V + 2, V + 3);
    mil += slice("v", HK, HK + H, 0, V) + slice("beta_h", HK + H, HK + 2 * H, 0, 1);
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> s1 = mul(x=stt, y=dcy)[name=string(\"s1\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> sk = mul(x=s1, y=kk)[name=string(\"sk\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> kvm = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string(\"kvm\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> dlt = sub(x=v, y=kvm)[name=string(\"dlt\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> dbt = mul(x=dlt, y=beta_h)[name=string(\"dbt\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=dbt)[name=string(\"dup\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> upd = mul(x=dup, y=kk)[name=string(\"upd\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> s2 = add(x=s1, y=upd)[name=string(\"s2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> sq = mul(x=s2, y=qq)[name=string(\"sq\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string(\"y\")];\n  } -> (y, s2);\n}\n";
    model_ = ane_model_compile_mil(ctx_, mil.c_str(), names, data, sizes, 2, 0, 21);
    if (!model_) return false;
    input_surface_ = metal_create_iosurface(input_channels_ * width_ * 2);
    output_surface_ = metal_create_iosurface(heads_ * value_dim_ * 2);
    state_surface_ = metal_create_iosurface(hidden_keys_ * value_dim_ * 2);
    if (!input_surface_ || !output_surface_ || !state_surface_) return false;
    metal_ctx_ = shared_recurrence_metal_context();
    if (metal_ctx_) {
        metal_state_ = metal_buffer_create(metal_ctx_, hidden_keys_ * value_dim_ * sizeof(uint16_t));
        metal_decay_ = metal_buffer_create(metal_ctx_, heads_ * 32 * sizeof(uint16_t));
        metal_key_ = metal_buffer_create(metal_ctx_, hidden_keys_ * 32 * sizeof(uint16_t));
        metal_query_ = metal_buffer_create(metal_ctx_, hidden_keys_ * 32 * sizeof(uint16_t));
        metal_value_ = metal_buffer_create(metal_ctx_, heads_ * value_dim_ * 32 * sizeof(uint16_t));
        metal_beta_ = metal_buffer_create(metal_ctx_, heads_ * 32 * sizeof(uint16_t));
        metal_output_ = metal_buffer_create(metal_ctx_, heads_ * value_dim_ * 32 * sizeof(uint16_t));
    }
    reset();
    size_t channels[2] = {0, 0};
    const size_t output_count = ane_model_output_channels(model_, channels, 2);
    if (output_count != 2) return false;
    IOSurfaceRef by_channels[2] = {nullptr, nullptr};
    for (size_t i = 0; i < output_count; ++i) {
        if (channels[i] == heads_) by_channels[i] = output_surface_;
        else if (channels[i] == hidden_keys_) by_channels[i] = state_surface_;
        else return false;
    }
    IOSurfaceRef outputs[] = {by_channels[0], by_channels[1]};
    request_ = ane_request_create_multi(ctx_, model_, input_surface_, outputs, 2, 0);
    return request_ != nullptr;
}

bool RindiGdnRecurrence::compile(ANEContext* ctx, size_t heads, size_t key_dim,
                                 size_t value_dim, size_t width) {
    if (!ctx || heads == 0 || key_dim == 0 || value_dim == 0 || width < value_dim + 5) return false;
    return compile_prepared(ctx, heads, key_dim, value_dim, width);
    ctx_ = ctx; heads_ = heads; key_dim_ = key_dim; value_dim_ = value_dim;
    hidden_keys_ = heads_ * key_dim_; input_channels_ = hidden_keys_ + 2 * heads_; width_ = width;
    std::vector<uint16_t> sum(heads_ * key_dim_, 0x3c00);
    std::vector<uint16_t> mean(heads_ * key_dim_, static_cast<uint16_t>(0x2200));
    std::vector<uint16_t> repeat(hidden_keys_, 0x3c00);
    const float mean_value = 1.0f / static_cast<float>(key_dim_);
    for (auto& x : mean) {
        union { float f; uint32_t u; } bits{mean_value};
        uint32_t sign = (bits.u >> 16) & 0x8000u;
        uint32_t mant = bits.u & 0x7fffffu;
        int exp = static_cast<int>((bits.u >> 23) & 0xffu) - 127 + 15;
        x = exp <= 0 ? 0 : static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    }
    const char* names[] = {"sum.bin", "mean.bin", "repeat.bin"};
    const void* data[] = {sum.data(), mean.data(), repeat.data()};
    const size_t sizes[] = {sum.size()*2, mean.size()*2, repeat.size()*2};

    // Prepared-input recurrence: state, decay, key, query, values, beta.
    const size_t H=heads_, D=key_dim_, V=value_dim_, HK=hidden_keys_, W=width_;
    std::string mil = "program(1.3)\n" + std::string(kBuildInfo) + "\n{\n  func main<ios18>(tensor<fp16, [1, " + std::to_string(input_channels_) + ", 1, " + std::to_string(W) + "]> x) {\n";
    mil += "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n";
    mil += "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n";
    mil += "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n";
    mil += "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n";
    mil += "    int32 g1 = const()[name=string(\"g1\"), val=int32(1)];\n";
    mil += "    int32 gh = const()[name=string(\"gh\"), val=int32(" + std::to_string(H) + ")];\n";
    mil += "    tensor<fp16, [" + std::to_string(H) + "," + std::to_string(D) + ",1,1]> gsum = const()[name=string(\"gsum\"), val=tensor<fp16, [" + std::to_string(H) + "," + std::to_string(D) + ",1,1]>(BLOBFILE(path=string(\"@model_path/weights/sum.bin\"), offset=uint64(64)))];\n";
    mil += "    tensor<fp16, [" + std::to_string(H) + "," + std::to_string(D) + ",1,1]> gmean = const()[name=string(\"gmean\"), val=tensor<fp16, [" + std::to_string(H) + "," + std::to_string(D) + ",1,1]>(BLOBFILE(path=string(\"@model_path/weights/mean.bin\"), offset=uint64(64)))];\n";
    mil += "    tensor<fp16, [" + std::to_string(HK) + ",1,1,1]> grep = const()[name=string(\"grep\"), val=tensor<fp16, [" + std::to_string(HK) + ",1,1,1]>(BLOBFILE(path=string(\"@model_path/weights/repeat.bin\"), offset=uint64(64)))];\n";
    mil += slice("stt",0,HK,0,V) + slice("kraw",0,HK,V,V+1) + slice("qraw",0,HK,V+1,V+2);
    mil += slice("v",HK,HK+H,0,V) + slice("beta_h",HK+H,HK+2*H,0,1);
    mil += slice("aa",0,HK,V+2,V+3) + slice("dtc",0,HK,V+3,V+4) + slice("alog",0,HK,V+4,V+5);
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> k8 = mul(x=kraw, y=fp16(0x1p+4))[name=string(\"k8\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ksq = mul(x=k8, y=k8)[name=string(\"ksq\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> kms = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gmean, x=ksq)[name=string(\"kms\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> kmse = add(x=kms, y=fp16(0x1.0c8p-12))[name=string(\"kmse\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> ksd = sqrt(x=kmse)[name=string(\"ksd\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ksdr = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=ksd)[name=string(\"ksdr\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> kunit = real_div(x=k8, y=ksdr)[name=string(\"kunit\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> kk2 = mul(x=kunit, y=fp16(0x1.6ap-4))[name=string(\"kk2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> q8 = mul(x=qraw, y=fp16(0x1p+4))[name=string(\"q8\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> qsq0 = mul(x=q8, y=q8)[name=string(\"qsq0\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> qms = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gmean, x=qsq0)[name=string(\"qms\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> qmse = add(x=qms, y=fp16(0x1.0c8p-12))[name=string(\"qmse\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> qsd = sqrt(x=qmse)[name=string(\"qsd\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> qsdr = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=qsd)[name=string(\"qsdr\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> qunit = real_div(x=q8, y=qsdr)[name=string(\"qunit\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> qq2 = mul(x=qunit, y=fp16(0x1p-1))[name=string(\"qq2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ap = add(x=aa, y=dtc)[name=string(\"ap\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> pos = relu(x=ap)[name=string(\"pos\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ab = abs(x=ap)[name=string(\"ab\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> nab = mul(x=ab, y=fp16(-0x1p+0))[name=string(\"nab\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> tt = exp(x=nab)[name=string(\"tt\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> hp5 = mul(x=tt, y=fp16(-0x1.84p-6))[name=string(\"hp5\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ha4 = add(x=hp5, y=fp16(0x1.9acp-4))[name=string(\"ha4\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> hm4 = mul(x=ha4, y=tt)[name=string(\"hm4\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ha3 = add(x=hm4, y=fp16(-0x1.ab4p-3))[name=string(\"ha3\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> hm3 = mul(x=ha3, y=tt)[name=string(\"hm3\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ha2 = add(x=hm3, y=fp16(0x1.4c4p-2))[name=string(\"ha2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> hm2 = mul(x=ha2, y=tt)[name=string(\"hm2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ha1 = add(x=hm2, y=fp16(-0x1.ff4p-2))[name=string(\"ha1\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> hm1 = mul(x=ha1, y=tt)[name=string(\"hm1\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> ha0 = add(x=hm1, y=fp16(0x1p+0))[name=string(\"ha0\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> tail = mul(x=tt, y=ha0)[name=string(\"tail\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> soft = add(x=pos, y=tail)[name=string(\"soft\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> avec = exp(x=alog)[name=string(\"avec\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> asp = mul(x=avec, y=soft)[name=string(\"asp\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> nasp = mul(x=asp, y=fp16(-0x1p+0))[name=string(\"nasp\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1,1]> dcy2 = exp(x=nasp)[name=string(\"dcy2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> nb = mul(x=beta_h, y=fp16(-0x1p+0))[name=string(\"nb\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> enb = exp(x=nb)[name=string(\"enb\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> bden = add(x=enb, y=fp16(0x1p+0))[name=string(\"bden\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1,1]> bta = real_div(x=fp16(0x1p+0), y=bden)[name=string(\"bta\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> s1 = mul(x=stt, y=dcy2)[name=string(\"s1\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> sk = mul(x=s1, y=kk2)[name=string(\"sk\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> kvm = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string(\"kvm\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> dlt = sub(x=v, y=kvm)[name=string(\"dlt\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> dbt = mul(x=dlt, y=beta_h)[name=string(\"dbt\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=dbt)[name=string(\"dup\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> upd = mul(x=dup, y=kk2)[name=string(\"upd\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> s2 = add(x=s1, y=upd)[name=string(\"s2\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(HK) + ",1," + std::to_string(V) + "]> sq = mul(x=s2, y=qq2)[name=string(\"sq\")];\n";
    mil += "    tensor<fp16, [1," + std::to_string(H) + ",1," + std::to_string(V) + "]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string(\"y\")];\n  } -> (y, s2);\n}\n";
    model_ = ane_model_compile_mil(ctx_, mil.c_str(), names, data, sizes, 3, 0, 21);
    if (!model_) return false;
    input_surface_ = metal_create_iosurface(input_channels_ * width_ * 2);
    output_surface_ = metal_create_iosurface(heads_ * value_dim_ * 2);
    state_surface_ = metal_create_iosurface(hidden_keys_ * value_dim_ * 2);
    if (!input_surface_ || !output_surface_ || !state_surface_) return false;
    reset();
    // ANE's compiled symbol order is state (6144 channels), then y (48
    // channels), despite the MIL return tuple being written as (y, state).
    IOSurfaceRef outputs[] = {state_surface_, output_surface_};
    request_ = ane_request_create_multi(ctx_, model_, input_surface_, outputs, 2, 0);
    return request_ != nullptr;
}

void RindiGdnRecurrence::reset() {
    if (!state_surface_) return;
    IOSurfaceLock(state_surface_, 0, nullptr);
    std::memset(IOSurfaceGetBaseAddress(state_surface_), 0, hidden_keys_ * value_dim_ * 2);
    IOSurfaceUnlock(state_surface_, 0, nullptr);
    if (metal_state_) {
        std::memset(metal_buffer_get_contents(metal_state_), 0,
                    hidden_keys_ * value_dim_ * sizeof(uint16_t));
    }
}

bool RindiGdnRecurrence::step_batch(const uint16_t* decay, const uint16_t* key,
                                    const uint16_t* query, const uint16_t* value,
                                    const uint16_t* beta, size_t lanes,
                                    std::vector<uint16_t>& output) {
    if (!decay || !key || !query || !value || !beta || lanes == 0 || lanes > 32 ||
        !metal_ctx_ || !metal_state_ || !metal_decay_ || !metal_key_ ||
        !metal_query_ || !metal_value_ || !metal_beta_ || !metal_output_) return false;
    std::memcpy(metal_buffer_get_contents(metal_decay_), decay,
                heads_ * lanes * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(metal_key_), key,
                hidden_keys_ * lanes * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(metal_query_), query,
                hidden_keys_ * lanes * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(metal_value_), value,
                heads_ * value_dim_ * lanes * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(metal_beta_), beta,
                heads_ * lanes * sizeof(uint16_t));
    MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
    if (!cmd) return false;
    metal_dispatch_gdn_recurrence(
        metal_ctx_, cmd, metal_state_, metal_decay_, metal_key_, metal_query_,
        metal_value_, metal_beta_, metal_output_, static_cast<int>(heads_),
        static_cast<int>(key_dim_), static_cast<int>(value_dim_),
        static_cast<int>(lanes));
    metal_command_buffer_commit(cmd);
    metal_command_buffer_wait(cmd);
    output.resize(heads_ * value_dim_ * lanes);
    std::memcpy(output.data(), metal_buffer_get_contents(metal_output_),
                output.size() * sizeof(uint16_t));
    return true;
}

bool RindiGdnRecurrence::step(const uint16_t* packed_inputs, std::vector<uint16_t>& output) {
    if (!request_ || !packed_inputs) return false;
    IOSurfaceLock(input_surface_, 0, nullptr);
    uint16_t* dst = static_cast<uint16_t*>(IOSurfaceGetBaseAddress(input_surface_));
    std::memset(dst, 0, input_channels_ * width_ * 2);
    IOSurfaceLock(state_surface_, kIOSurfaceLockReadOnly, nullptr);
    const uint16_t* state = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(state_surface_));
    for (size_t c = 0; c < hidden_keys_; ++c) {
        std::memcpy(dst + c * width_, state + c * value_dim_,
                    value_dim_ * sizeof(uint16_t));
    }
    IOSurfaceUnlock(state_surface_, kIOSurfaceLockReadOnly, nullptr);
    // The first HK rows contain recurrent state in columns [0,V), and the
    // prepared k/q/decay lanes in columns [V,W). Preserve both groups.
    for (size_t c = 0; c < hidden_keys_; ++c) {
        std::memcpy(dst + c * width_ + value_dim_,
                    packed_inputs + c * width_ + value_dim_,
                    (width_ - value_dim_) * sizeof(uint16_t));
    }
    std::memcpy(dst + hidden_keys_ * width_, packed_inputs + hidden_keys_ * width_,
                (input_channels_ - hidden_keys_) * width_ * sizeof(uint16_t));
    IOSurfaceUnlock(input_surface_, 0, nullptr);
    if (!ane_request_evaluate(ctx_, model_, request_, nullptr, 0, nullptr, 0)) return false;
    output.resize(heads_ * value_dim_);
    IOSurfaceLock(output_surface_, kIOSurfaceLockReadOnly, nullptr);
    const uint16_t* output_base = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(output_surface_));
    std::memcpy(output.data(), output_base, output.size()*2);
    IOSurfaceUnlock(output_surface_, kIOSurfaceLockReadOnly, nullptr);
    if (std::getenv("RINDI_DEBUG_GDN_RECURRENCE")) {
        IOSurfaceLock(state_surface_, kIOSurfaceLockReadOnly, nullptr);
        const uint16_t* state_base = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(state_surface_));
        std::fprintf(stderr, "[RindiGdnRecurrence] y=%04x,%04x state=%04x,%04x\n",
                     output_base[0], output_base[1], state_base[0], state_base[1]);
        IOSurfaceUnlock(state_surface_, kIOSurfaceLockReadOnly, nullptr);
    }
    return true;
}
