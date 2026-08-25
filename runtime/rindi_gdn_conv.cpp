// SPDX-License-Identifier: Apache-2.0
#include "rindi_gdn_conv.h"
#include "metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>

namespace {
struct GdnConvMetal {
    MetalContext* ctx{nullptr};
    MetalBufferHandle current{nullptr};
    MetalBufferHandle history{nullptr};
    MetalBufferHandle weight{nullptr};
    MetalBufferHandle output{nullptr};
    size_t current_capacity{0};
    size_t channel_capacity{0};
};

GdnConvMetal& shared_gdn_conv_metal() {
    static GdnConvMetal state;
    return state;
}

bool evaluate_metal_conv(const uint16_t* current,
                         const std::vector<uint16_t>& history,
                         const std::vector<uint16_t>& weight,
                         size_t channels, size_t lanes,
                         std::vector<uint16_t>& output) {
    GdnConvMetal& m = shared_gdn_conv_metal();
    if (!m.ctx) m.ctx = metal_context_create();
    if (!m.ctx) return false;
    const size_t current_elems = channels * lanes;
    if (m.current_capacity < current_elems) {
        if (m.current) metal_buffer_release(m.current);
        if (m.output) metal_buffer_release(m.output);
        m.current = metal_buffer_create(m.ctx, current_elems * sizeof(uint16_t));
        m.output = metal_buffer_create(m.ctx, current_elems * sizeof(uint16_t));
        m.current_capacity = (m.current && m.output) ? current_elems : 0;
    }
    if (m.channel_capacity < channels) {
        if (m.history) metal_buffer_release(m.history);
        if (m.weight) metal_buffer_release(m.weight);
        m.history = metal_buffer_create(m.ctx, channels * 3 * sizeof(uint16_t));
        m.weight = metal_buffer_create(m.ctx, channels * 4 * sizeof(uint16_t));
        m.channel_capacity = (m.history && m.weight) ? channels : 0;
    }
    if (!m.current_capacity || !m.channel_capacity) return false;
    std::memcpy(metal_buffer_get_contents(m.current), current,
                current_elems * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(m.history), history.data(),
                channels * 3 * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(m.weight), weight.data(),
                channels * 4 * sizeof(uint16_t));
    MetalCommandBufferHandle cmd = metal_command_buffer_create(m.ctx);
    if (!cmd) return false;
    metal_dispatch_gdn_conv_silu(m.ctx, cmd, m.current, m.history, m.weight,
                                 m.output, static_cast<int>(channels),
                                 static_cast<int>(lanes));
    metal_command_buffer_commit(cmd);
    metal_command_buffer_wait(cmd);
    output.resize(current_elems);
    std::memcpy(output.data(), metal_buffer_get_contents(m.output),
                current_elems * sizeof(uint16_t));
    return true;
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t man  = (h & 0x03FF);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; }
               man &= 0x3FF; bits = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float v; std::memcpy(&v, &bits, 4); return v;
}
inline uint16_t float_to_half(float f) {
    uint32_t b; std::memcpy(&b, &f, 4);
    uint16_t h = static_cast<uint16_t>((b >> 16) & 0x8000);
    const int32_t e = static_cast<int32_t>((b >> 23) & 0xFF) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(h | 0x7C00);
    if (e <= 0) {
        if (e < -10) return h;
        return static_cast<uint16_t>(h | ((b >> 13) & 0x03FF) >> (1 - e));
    }
    return static_cast<uint16_t>(h | (static_cast<uint32_t>(e) << 10) |
                                 ((b >> 13) & 0x03FF));
}
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
    if (!info || info->shape.size() != 3 || info->shape[1] != 4 || info->shape[2] != 1) return false;
    if (!ctx && !std::getenv("RINDI_TAIL_COREAI")) return false;
    channels_ = static_cast<size_t>(info->shape[0]);
    width_ = std::max<size_t>(32, width);
    std::vector<uint16_t> weights;
    if (!loader.get_tensor_fp16(weight_name, weights) || weights.size() != channels_ * 4) return false;
    const std::string mil = "program(1.3)\n" + std::string(kBuildInfo) + "\n{\n"
        "  func main<ios26>(tensor<fp16, [1, " + std::to_string(channels_) + ", 1, " + std::to_string(width_) + "]> x) {\n"
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
    if (!model_) {
        // macOS 27: legacy MIL bundles fail verification. The op is a causal
        // depthwise 4-tap FIR followed by SiLU - run it on the CPU instead.
        cpu_fallback_ = true;
        weights_ = weights;
        history_.assign(channels_ * 3, 0);
        written_lanes_ = 0;
        std::cout << "[RindiGDN] conv: using CPU fallback (channels="
                  << channels_ << ")" << std::endl;
        return true;
    }
    ctx_ = ctx;
    input_surface_ = metal_create_iosurface(channels_ * width_ * sizeof(uint16_t));
    output_surface_ = metal_create_iosurface(channels_ * width_ * sizeof(uint16_t));
    if (!input_surface_ || !output_surface_) return false;
    request_ = ane_request_create(ctx_, model_, input_surface_, output_surface_, 0);
    history_.assign(channels_ * 3, 0);
    return request_ != nullptr;
}

void RindiGdnConv::reset() { std::fill(history_.begin(), history_.end(), 0); }

static inline float fp32_silu(float v) { return v / (1.0f + expf(-v)); }

bool RindiGdnConv::evaluate(const uint16_t* current, size_t lanes,
                            std::vector<uint16_t>& output) {
    if (cpu_fallback_) {
        const bool metal_conv = std::getenv("RINDI_GDN_METAL_CONV") != nullptr ||
                                std::getenv("RINDI_QWEN_PREFILL_FAST") != nullptr;
        const size_t max_lanes = metal_conv ? width_ : width_ - 3;
        if (!current || lanes == 0 || lanes > max_lanes || weights_.empty()) {
            std::fprintf(stderr, "[GdnConv-CPU] reject: cur=%d lanes=%zu w=%zu wt=%zu\n",
                         current != nullptr, lanes, width_, weights_.size());
            return false;
        }
        if (lanes > 1 && metal_conv &&
            evaluate_metal_conv(current, history_, weights_, channels_, lanes,
                                output)) {
            for (size_t c = 0; c < channels_; ++c) {
                uint16_t* hist = history_.data() + c * 3;
                const uint16_t* cur = current + c * lanes;
                if (lanes >= 3) {
                    std::memcpy(hist, cur + (lanes - 3), 3 * sizeof(uint16_t));
                } else {
                    std::memmove(hist, hist + lanes,
                                 (3 - lanes) * sizeof(uint16_t));
                    std::memcpy(hist + (3 - lanes), cur,
                                lanes * sizeof(uint16_t));
                }
            }
            return true;
        }
        output.resize(channels_ * lanes);
        for (size_t c = 0; c < channels_; ++c) {
            const uint16_t* wv = weights_.data() + c * 4;
            const float w0 = half_to_float(wv[0]), w1 = half_to_float(wv[1]);
            const float w2 = half_to_float(wv[2]), w3 = half_to_float(wv[3]);
            const uint16_t* cur = current + c * lanes;
            uint16_t* hist = history_.data() + c * 3;
            uint16_t* out = output.data() + c * lanes;
            for (size_t t = 0; t < lanes; ++t) {
                float acc = 0.0f;
                for (size_t k = 0; k < 4; ++k) {
                    const int64_t idx = static_cast<int64_t>(t) +
                                        static_cast<int64_t>(k) - 3;
                    const float xv =
                        idx < 0 ? half_to_float(hist[3 + static_cast<size_t>(idx)])
                                : half_to_float(cur[static_cast<size_t>(idx)]);
                    acc += (k == 0 ? w0 : k == 1 ? w1 : k == 2 ? w2 : w3) * xv;
                }
                out[t] = float_to_half(fp32_silu(acc));
            }
            // identical rolling-window update to the ANE path
            if (lanes >= 3) {
                std::memcpy(hist, cur + (lanes - 3), 3 * sizeof(uint16_t));
            } else {
                std::memmove(hist, hist + lanes, (3 - lanes) * sizeof(uint16_t));
                std::memcpy(hist + (3 - lanes), cur, lanes * sizeof(uint16_t));
            }
        }
        return true;
    }
    if (!request_ || !current || lanes == 0 || lanes > width_ - 3) return false;
    uint16_t* dst = static_cast<uint16_t*>(IOSurfaceGetBaseAddress(input_surface_));
    // History [0,3) and the live lanes [3, 3+lanes) are fully rewritten every
    // call. Columns beyond 3+lanes feed conv outputs that are never gathered,
    // so a per-call full-surface memset is only needed when the lane count
    // changes and would otherwise expose stale values in the read window.
    // (P12 experiment: unconditional memset changed nothing - surface state
    // is not part of any divergence path.)
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
