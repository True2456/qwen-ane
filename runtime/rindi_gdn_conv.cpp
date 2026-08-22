// SPDX-License-Identifier: Apache-2.0
#include "rindi_gdn_conv.h"
#include "metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <algorithm>
#include <cstring>

namespace {
constexpr const char* kBuildInfo =
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";
}

RindiGdnConv::~RindiGdnConv() {
    if (request_) ane_request_release(request_);
    if (model_) ane_model_release(model_);
    if (input_surface_) CFRelease(input_surface_);
    if (output_surface_) CFRelease(output_surface_);
}

bool RindiGdnConv::compile(ANEContext* ctx, const SafeTensorsLoader& loader,
                           const std::string& weight_name, size_t width) {
    const TensorInfo* info = loader.get_tensor_info(weight_name);
    if (!ctx || !info || info->shape.size() != 3 || info->shape[1] != 4 || info->shape[2] != 1) return false;
    channels_ = static_cast<size_t>(info->shape[0]);
    width_ = std::max<size_t>(32, width);
    std::vector<uint16_t> weights;
    if (!loader.get_tensor_fp16(weight_name, weights) || weights.size() != channels_ * 4) return false;
    const std::string mil = "program(1.3)\n" + std::string(kBuildInfo) + "\n{\n"
        "  func main<ios18>(tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> x) {\n"
        "    tensor<fp16, [" + std::to_string(channels_) + ", 1, 1, 4]> w = const()[name=string(\"w\"), val=tensor<fp16, [" + std::to_string(channels_) + ", 1, 1, 4]>(BLOBFILE(path=string(\"@model_path/weights/conv.bin\"), offset=uint64(64)))];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,3,0])];\n"
        "    tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> c = conv(dilations=dl, groups=int32(" + std::to_string(channels_) + "), pad=pd, pad_type=string(\"custom\"), strides=st, weight=w, x=x)[name=string(\"causal\")];\n"
        "    tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> nc = mul(x=c, y=fp16(-0x1p+0))[name=string(\"nc\")];\n"
        "    tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> ex = exp(x=nc)[name=string(\"ex\")];\n"
        "    tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> den = add(x=ex, y=fp16(0x1p+0))[name=string(\"den\")];\n"
        "    tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> y = real_div(x=c, y=den)[name=string(\"silu\")];\n"
        "  } -> (y);\n}\n";
    const char* names[] = {"conv.bin"};
    const void* data[] = {weights.data()};
    const size_t sizes[] = {weights.size() * sizeof(uint16_t)};
    model_ = ane_model_compile_mil(ctx, mil.c_str(), names, data, sizes, 1, 0, 21);
    if (!model_) return false;
    ctx_ = ctx;
    input_surface_ = metal_create_iosurface(channels_ * width_ * sizeof(uint16_t));
    output_surface_ = metal_create_iosurface(channels_ * width_ * sizeof(uint16_t));
    if (!input_surface_ || !output_surface_) return false;
    request_ = ane_request_create(ctx_, model_, input_surface_, output_surface_, 0);
    history_.assign(channels_ * 3, 0);
    return request_ != nullptr;
}

void RindiGdnConv::reset() { std::fill(history_.begin(), history_.end(), 0); }

bool RindiGdnConv::evaluate(const uint16_t* current, size_t lanes,
                            std::vector<uint16_t>& output) {
    if (!request_ || !current || lanes == 0 || lanes > width_ - 3) return false;
    uint16_t* dst = static_cast<uint16_t*>(IOSurfaceGetBaseAddress(input_surface_));
    // History [0,3) and the live lanes [3, 3+lanes) are fully rewritten every
    // call. Columns beyond 3+lanes feed conv outputs that are never gathered,
    // so a per-call full-surface memset is only needed when the lane count
    // changes and would otherwise expose stale values in the read window.
    if (written_lanes_ != lanes) {
        std::memset(dst, 0, channels_ * width_ * sizeof(uint16_t));
        written_lanes_ = lanes;
    }
    for (size_t c = 0; c < channels_; ++c) {
        std::memcpy(dst + c * width_, history_.data() + c * 3,
                    3 * sizeof(uint16_t));
        std::memcpy(dst + c * width_ + 3, current + c * lanes,
                    lanes * sizeof(uint16_t));
    }
    if (!ane_request_evaluate(ctx_, model_, request_, nullptr, 0, nullptr, 0)) return false;
    output.resize(channels_ * lanes);
    const uint16_t* src = static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(output_surface_));
    // IOSurface tensors are channel-major: each channel owns a contiguous
    // width row.  The live decode lanes begin at column 3 after the causal
    // three-sample history, so the result must be gathered row by row.
    for (size_t c = 0; c < channels_; ++c) {
        std::memcpy(output.data() + c * lanes,
                    src + c * width_ + 3,
                    lanes * sizeof(uint16_t));
    }
    // The IOSurface and `history_` are channel-major ([channel, time]), not
    // time-major ([time, channel]).  Shift each channel's causal window so a
    // multi-channel projection cannot become the next channel's history.
    for (size_t c = 0; c < channels_; ++c) {
        uint16_t* h = history_.data() + c * 3;
        if (lanes >= 3) {
            std::memcpy(h, current + c * lanes + (lanes - 3),
                        3 * sizeof(uint16_t));
        } else {
            std::memmove(h, h + lanes, (3 - lanes) * sizeof(uint16_t));
            std::memcpy(h + (3 - lanes), current + c * lanes,
                        lanes * sizeof(uint16_t));
        }
    }
    return true;
}
