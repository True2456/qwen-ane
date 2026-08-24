// SPDX-License-Identifier: Apache-2.0
#include "rindi_ane_projection.h"
#include "metal_engine.h"
#include <IOSurface/IOSurface.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace {
constexpr const char* kBuildInfo =
    "[buildInfo = dict<string, string>({"
    "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

std::string make_mil(size_t input_dim, size_t output_dim, size_t width,
                     const std::string& tag) {
    return "program(1.3)\n" + std::string(kBuildInfo) + "\n{\n"
        "  func main<ios26>(tensor<fp16, [1, " + std::to_string(input_dim) +
        ", 1, " + std::to_string(width) + "]> x) {\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "    tensor<fp16, [" + std::to_string(output_dim) + ", " +
        std::to_string(input_dim) + ", 1, 1]> w = const()[name=string(\"w\"), val=tensor<fp16, [" +
        std::to_string(output_dim) + ", " + std::to_string(input_dim) +
        ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [1, " + std::to_string(output_dim) + ", 1, " +
        std::to_string(width) + "]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"" +
        tag + "\")];\n  } -> (y);\n}\n";
}

std::string make_int4_mil(size_t input_dim, size_t output_dim, size_t width,
                          const std::string& tag) {
    return "program(1.3)\n" + std::string(kBuildInfo) + "\n{\n"
        "  func main<ios26>(tensor<fp16, [1, " + std::to_string(input_dim) +
        ", 1, " + std::to_string(width) + "]> x) {\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "    tensor<int4, [" + std::to_string(output_dim) + ", " + std::to_string(input_dim) + ", 1, 1]> q = const()[name=string(\"q\"), val=tensor<int4, [" +
        std::to_string(output_dim) + ", " + std::to_string(input_dim) + ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/q.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [" + std::to_string(output_dim) + ", 1, 1, 1]> sc = const()[name=string(\"sc\"), val=tensor<fp16, [" + std::to_string(output_dim) + ", 1, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/qsc.bin\"), offset=uint64(64)))];\n"
        "    tensor<fp16, [" + std::to_string(output_dim) + ", " + std::to_string(input_dim) + ", 1, 1]> w = constexpr_blockwise_shift_scale(data=q, scale=sc)[name=string(\"dq\")];\n"
        "    tensor<fp16, [1, " + std::to_string(output_dim) + ", 1, " + std::to_string(width) + "]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string(\"" + tag + "\")];\n  } -> (y);\n}\n";
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t value;
    if (exponent == 0) {
        value = sign;
    } else if (exponent == 31) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint16_t float_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = (bits >> 13) & 0x3ffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

MetalContext* shared_projection_metal_context() {
    static MetalContext* context = metal_context_create();
    return context;
}

std::vector<uint8_t> read_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) return {};
    return data;
}
}

RindiAneProjection::~RindiAneProjection() {
    if (metal_weights_) metal_buffer_release(metal_weights_);
    if (metal_scales_) metal_buffer_release(metal_scales_);
    if (metal_biases_) metal_buffer_release(metal_biases_);
    if (metal_input_) metal_buffer_release(metal_input_);
    if (metal_output_) metal_buffer_release(metal_output_);
    if (request_) ane_request_release(request_);
    if (model_) ane_model_release(model_);
    if (input_surface_) CFRelease(input_surface_);
    if (output_surface_) CFRelease(output_surface_);
}

bool RindiAneProjection::compile_int4_host(const SafeTensorsLoader& loader,
                                           const std::string& tensor_name) {
    const TensorInfo* info = loader.get_tensor_info(tensor_name);
    if (!info || info->shape.size() != 2) return false;
    const size_t output_dim = static_cast<size_t>(info->shape[0]);
    const size_t input_dim = static_cast<size_t>(info->shape[1]) *
        (info->dtype == "U32" ? 8 : 1);
    if (input_dim == 0 || output_dim == 0 || (input_dim & 1)) return false;

    input_dim_ = input_dim;
    output_dim_ = output_dim;
    width_ = 1;
    host_loader_ = &loader;
    host_tensor_name_ = tensor_name;
    host_groupwise_ = info->dtype == "U32";
    host_ready_ = true;
    if (host_groupwise_) {
        // Keep the CPU implementation as a fallback, but prefer the shared
        // Metal path for the native scheduler's hot projections.
        if (!std::getenv("RINDI_DISABLE_METAL_PROJECTIONS") &&
            init_metal_int4(loader, tensor_name)) width_ = 32;
    }
    return true;
}

bool RindiAneProjection::compile_chain_int4(
    MetalContext* ctx, const std::string& weights_path,
    const std::string& scales_path, size_t input_dim, size_t output_dim) {
    if (!ctx || input_dim == 0 || output_dim == 0 || (input_dim & 1)) return false;
    auto weights = read_binary(weights_path);
    auto scales = read_binary(scales_path);
    const size_t packed_cols = input_dim / 2;
    if (weights.size() != output_dim * packed_cols ||
        scales.size() != output_dim * sizeof(uint16_t)) return false;
    metal_ctx_ = ctx;
    input_dim_ = input_dim;
    output_dim_ = output_dim;
    width_ = 32;
    host_ready_ = false;
    host_groupwise_ = false;
    metal_rowwise_ = true;
    metal_weights_ = metal_buffer_create(ctx, weights.size());
    metal_scales_ = metal_buffer_create(ctx, scales.size());
    metal_input_ = metal_buffer_create(ctx, input_dim * 32 * sizeof(uint16_t));
    metal_output_ = metal_buffer_create(ctx, output_dim * 32 * sizeof(uint16_t));
    if (!metal_weights_ || !metal_scales_ || !metal_input_ || !metal_output_) return false;
    std::memcpy(metal_buffer_get_contents(metal_weights_), weights.data(), weights.size());
    std::memcpy(metal_buffer_get_contents(metal_scales_), scales.data(), scales.size());
    metal_packed_cols_ = packed_cols;
    metal_groups_ = 0;
    metal_ready_ = true;
    return true;
}

bool RindiAneProjection::compile_bf16_metal(MetalContext* ctx,
                                            const SafeTensorsLoader& loader,
                                            const std::string& tensor_name) {
    const TensorInfo* info = loader.get_tensor_info(tensor_name);
    if (!ctx || !info || info->dtype != "BF16" || info->shape.size() != 2) return false;
    const size_t rows = static_cast<size_t>(info->shape[0]);
    const size_t cols = static_cast<size_t>(info->shape[1]);
    if (rows == 0 || cols == 0 || info->nbytes != rows * cols * sizeof(uint16_t)) return false;
    const void* data = loader.get_tensor_data(tensor_name);
    if (!data) return false;
    metal_ctx_ = ctx;
    input_dim_ = cols;
    output_dim_ = rows;
    width_ = 32;
    host_ready_ = false;
    host_groupwise_ = false;
    metal_ready_ = false;
    metal_rowwise_ = false;
    bf16_ready_ = true;
    metal_weights_ = metal_buffer_create(ctx, info->nbytes);
    metal_input_ = metal_buffer_create(ctx, cols * 32 * sizeof(uint16_t));
    metal_output_ = metal_buffer_create(ctx, rows * 32 * sizeof(uint16_t));
    if (!metal_weights_ || !metal_input_ || !metal_output_) {
        bf16_ready_ = false;
        return false;
    }
    std::memcpy(metal_buffer_get_contents(metal_weights_), data, info->nbytes);
    metal_packed_cols_ = 0;
    metal_groups_ = 0;
    return true;
}

bool RindiAneProjection::compile_int4_from_bf16_fused(
    MetalContext* ctx, const SafeTensorsLoader& loader,
    const std::vector<std::pair<std::string, size_t>>& tensors) {
    if (!ctx || tensors.empty()) return false;
    size_t rows = 0, cols = 0;
    for (const auto& t : tensors) {
        const TensorInfo* info = loader.get_tensor_info(t.first);
        if (!info || info->dtype != "BF16" || info->shape.size() != 2 ||
            static_cast<size_t>(info->shape[0]) != t.second) return false;
        if (cols == 0) cols = static_cast<size_t>(info->shape[1]);
        else if (static_cast<size_t>(info->shape[1]) != cols) return false;
        rows += t.second;
    }
    if (rows == 0 || cols == 0 || cols % 64 != 0) return false;
    std::vector<uint16_t> joined(rows * cols);
    size_t row_off = 0;
    for (const auto& t : tensors) {
        const void* data = loader.get_tensor_data(t.first);
        if (!data) return false;
        std::memcpy(joined.data() + row_off * cols, data,
                    t.second * cols * sizeof(uint16_t));
        row_off += t.second;
    }
    return compile_int4_from_bf16_rows(ctx, joined.data(), rows, cols);
}

bool RindiAneProjection::compile_int4_from_bf16(MetalContext* ctx,
                                                const SafeTensorsLoader& loader,
                                                const std::string& tensor_name) {
    const TensorInfo* info = loader.get_tensor_info(tensor_name);
    if (!ctx || !info || info->dtype != "BF16" || info->shape.size() != 2) return false;
    const size_t rows = static_cast<size_t>(info->shape[0]);
    const size_t cols = static_cast<size_t>(info->shape[1]);
    if (rows == 0 || cols == 0 || cols % 64 != 0 ||
        info->nbytes != rows * cols * sizeof(uint16_t)) return false;
    const void* data = loader.get_tensor_data(tensor_name);
    if (!data) return false;
    return compile_int4_from_bf16_rows(ctx, data, rows, cols);
}

bool RindiAneProjection::compile_int4_from_bf16_rows(MetalContext* ctx,
                                                     const void* data,
                                                     size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || cols % 64 != 0) return false;
    const size_t groups = cols / 64;
    std::vector<uint8_t> packed(rows * cols / 2);
    std::vector<uint16_t> scales(rows * groups);
    std::vector<uint16_t> biases(rows * groups);
    const uint16_t* src = static_cast<const uint16_t*>(data);

    auto bf = [](uint16_t bits) {
        uint32_t v = static_cast<uint32_t>(bits) << 16;
        float f; std::memcpy(&f, &v, 4); return f;
    };
    auto to_bf = [](float f) {
        uint32_t b; std::memcpy(&b, &f, 4);
        return static_cast<uint16_t>(b >> 16);
    };
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t* row = src + r * cols;
        uint8_t* prow = packed.data() + r * cols / 2;
        for (size_t g = 0; g < groups; ++g) {
            float mn = 3.0e38f, mx = -3.0e38f;
            for (size_t j = 0; j < 64; ++j) {
                const float v = bf(row[g * 64 + j]);
                mn = std::min(mn, v); mx = std::max(mx, v);
            }
            float scale = (mx - mn) / 15.0f;
            if (!(scale > 0.0f)) scale = 1.0f;
            scales[r * groups + g] = to_bf(scale);
            biases[r * groups + g] = to_bf(mn);
            const float inv = 1.0f / scale;
            for (size_t j = 0; j < 64; ++j) {
                const float v = bf(row[g * 64 + j]);
                int q = static_cast<int>(std::lrint((v - mn) * inv));
                q = std::max(0, std::min(15, q));
                const size_t cidx = g * 64 + j;
                if ((cidx & 1) == 0) prow[cidx / 2] = static_cast<uint8_t>(q & 0xF);
                else prow[cidx / 2] |= static_cast<uint8_t>((q & 0xF) << 4);
            }
        }
    }

    metal_ctx_ = ctx;
    input_dim_ = cols;
    output_dim_ = rows;
    width_ = 32;
    host_ready_ = false;
    host_groupwise_ = false;
    metal_ready_ = true;
    metal_rowwise_ = false;
    bf16_ready_ = false;
    metal_weights_ = metal_buffer_create(ctx, packed.size());
    metal_scales_ = metal_buffer_create(ctx, scales.size() * sizeof(uint16_t));
    metal_biases_ = metal_buffer_create(ctx, biases.size() * sizeof(uint16_t));
    metal_input_ = metal_buffer_create(ctx, cols * 32 * sizeof(uint16_t));
    metal_output_ = metal_buffer_create(ctx, rows * 32 * sizeof(uint16_t));
    if (!metal_weights_ || !metal_scales_ || !metal_biases_ ||
        !metal_input_ || !metal_output_) {
        metal_ready_ = false;
        return false;
    }
    std::memcpy(metal_buffer_get_contents(metal_weights_), packed.data(), packed.size());
    std::memcpy(metal_buffer_get_contents(metal_scales_), scales.data(),
                scales.size() * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(metal_biases_), biases.data(),
                biases.size() * sizeof(uint16_t));
    metal_packed_cols_ = cols / 8;
    metal_groups_ = groups;
    return true;
}

bool RindiAneProjection::init_metal_int4(const SafeTensorsLoader& loader,
                                         const std::string& tensor_name) {
    const TensorInfo* info = loader.get_tensor_info(tensor_name);
    if (!info || info->dtype != "U32" || info->shape.size() != 2) return false;
    const size_t rows = static_cast<size_t>(info->shape[0]);
    const size_t packed_cols = static_cast<size_t>(info->shape[1]);
    const size_t logical_cols = packed_cols * 8;
    const std::string scale_name = tensor_name.substr(0, tensor_name.size() - 6) + "scales";
    const std::string bias_name = tensor_name.substr(0, tensor_name.size() - 6) + "biases";
    const TensorInfo* scale_info = loader.get_tensor_info(scale_name);
    if (!scale_info || scale_info->dtype != "BF16" || scale_info->shape.size() < 2 ||
        static_cast<size_t>(scale_info->shape[0]) < rows) return false;
    const size_t groups = (logical_cols + 63) / 64;
    if (static_cast<size_t>(scale_info->shape[1]) < groups) return false;
    const TensorInfo* bias_info = loader.get_tensor_info(bias_name);
    if (bias_info && (bias_info->dtype != "BF16" || bias_info->shape.size() < 2 ||
                      static_cast<size_t>(bias_info->shape[0]) < rows ||
                      static_cast<size_t>(bias_info->shape[1]) < groups)) return false;

    metal_ctx_ = shared_projection_metal_context();
    if (!metal_ctx_) return false;
    const void* weight_data = loader.get_tensor_data(tensor_name);
    const void* scale_data = loader.get_tensor_data(scale_name);
    if (!weight_data || !scale_data) return false;
    const size_t weight_bytes = rows * packed_cols * sizeof(uint32_t);
    const size_t scale_bytes = rows * groups * sizeof(uint16_t);
    metal_weights_ = metal_buffer_create(metal_ctx_, weight_bytes);
    metal_scales_ = metal_buffer_create(metal_ctx_, scale_bytes);
    metal_biases_ = metal_buffer_create(metal_ctx_, scale_bytes);
    metal_input_ = metal_buffer_create(metal_ctx_, logical_cols * 32 * sizeof(uint16_t));
    metal_output_ = metal_buffer_create(metal_ctx_, rows * 32 * sizeof(uint16_t));
    if (!metal_weights_ || !metal_scales_ || !metal_biases_ ||
        !metal_input_ || !metal_output_) return false;
    std::memcpy(metal_buffer_get_contents(metal_weights_), weight_data, weight_bytes);
    std::memcpy(metal_buffer_get_contents(metal_scales_), scale_data, scale_bytes);
    if (bias_info) {
        const void* bias_data = loader.get_tensor_data(bias_name);
        if (!bias_data) return false;
        std::memcpy(metal_buffer_get_contents(metal_biases_), bias_data, scale_bytes);
    } else {
        std::memset(metal_buffer_get_contents(metal_biases_), 0, scale_bytes);
    }
    metal_packed_cols_ = packed_cols;
    metal_groups_ = groups;
    metal_ready_ = true;
    return true;
}

bool RindiAneProjection::compile_fp16(ANEContext* ctx, const uint16_t* weights,
                                      size_t input_dim, size_t output_dim,
                                      size_t width, const std::string& tag) {
    if (!ctx || !weights || input_dim == 0 || output_dim == 0 || width == 0 || width > 256) return false;
    ctx_ = ctx;
    host_ready_ = false;
    host_groupwise_ = false;
    host_loader_ = nullptr;
    host_tensor_name_.clear();
    input_dim_ = input_dim;
    output_dim_ = output_dim;
    width_ = std::max<size_t>(32, width);
    const char* names[] = {"w.bin"};
    const void* data[] = {weights};
    const size_t sizes[] = {input_dim * output_dim * sizeof(uint16_t)};
    const std::string mil = make_mil(input_dim, output_dim, width_, tag);
    model_ = ane_model_compile_mil(ctx_, mil.c_str(), names, data, sizes, 1, 0, 21);
    if (!model_) return false;
    input_surface_ = metal_create_iosurface(input_dim_ * width_ * sizeof(uint16_t));
    output_surface_ = metal_create_iosurface(output_dim_ * width_ * sizeof(uint16_t));
    if (!input_surface_ || !output_surface_) return false;
    request_ = ane_request_create(ctx_, model_, input_surface_, output_surface_, 0);
    return request_ != nullptr;
}

bool RindiAneProjection::compile_int4(ANEContext* ctx, const SafeTensorsLoader& loader,
                                      const std::string& tensor_name, size_t width,
                                      const std::string& tag) {
    const TensorInfo* info = loader.get_tensor_info(tensor_name);
    if (!ctx || !info || info->shape.size() != 2 || info->shape[1] % 2 != 0) return false;
    const size_t output_dim = static_cast<size_t>(info->shape[0]);
    const size_t input_dim = static_cast<size_t>(info->shape[1]) *
        (info->dtype == "U32" ? 8 : 1);
    std::vector<uint8_t> packed(output_dim * input_dim / 2);
    std::vector<uint16_t> scales(output_dim);
    std::vector<uint16_t> row;
    for (size_t r = 0; r < output_dim; ++r) {
        if (!loader.get_row_fp16(tensor_name, r, row) || row.size() != input_dim) return false;
        float max_abs = 0.0f;
        for (uint16_t value : row) max_abs = std::max(max_abs, std::abs(fp16_to_float(value)));
        const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
        scales[r] = float_to_fp16(scale);
        for (size_t c = 0; c < input_dim; c += 2) {
            int q0 = static_cast<int>(std::lrint(fp16_to_float(row[c]) / scale));
            int q1 = static_cast<int>(std::lrint(fp16_to_float(row[c + 1]) / scale));
            q0 = std::max(-8, std::min(7, q0));
            q1 = std::max(-8, std::min(7, q1));
            packed[r * input_dim / 2 + c / 2] = static_cast<uint8_t>((q0 & 0x0f) | ((q1 & 0x0f) << 4));
        }
    }
    const char* names[] = {"q.bin", "qsc.bin"};
    const void* data[] = {packed.data(), scales.data()};
    const size_t sizes[] = {packed.size(), scales.size() * sizeof(uint16_t)};
    const std::string mil = make_int4_mil(input_dim, output_dim, std::max<size_t>(32, width), tag);
    model_ = ane_model_compile_mil(ctx, mil.c_str(), names, data, sizes, 2, 0, 21);
    if (!model_) return false;
    ctx_ = ctx; input_dim_ = input_dim; output_dim_ = output_dim; width_ = std::max<size_t>(32, width);
    host_ready_ = false;
    host_groupwise_ = false;
    host_loader_ = nullptr;
    host_tensor_name_.clear();
    input_surface_ = metal_create_iosurface(input_dim_ * width_ * sizeof(uint16_t));
    output_surface_ = metal_create_iosurface(output_dim_ * width_ * sizeof(uint16_t));
    if (!input_surface_ || !output_surface_) return false;
    request_ = ane_request_create(ctx_, model_, input_surface_, output_surface_, 0);
    return request_ != nullptr;
}

bool RindiAneProjection::evaluate(const uint16_t* input, size_t lanes,
                                  std::vector<uint16_t>& output) {
    if (!ready() || !input || lanes == 0 || lanes > width_) return false;
    if (bf16_ready_ && lanes <= 32) {
        // gemm_bf16 reads A as fp16 [M=lanes, K] row-major while callers pass
        // channel-major [K, lanes]; they coincide at lanes == 1.
        uint16_t* staged = static_cast<uint16_t*>(metal_buffer_get_contents(metal_input_));
        if (lanes == 1) {
            std::memcpy(staged, input, input_dim_ * sizeof(uint16_t));
        } else {
            for (size_t c = 0; c < input_dim_; ++c)
                for (size_t l = 0; l < lanes; ++l)
                    staged[l * input_dim_ + c] = input[c * lanes + l];
        }
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
        if (!cmd) return false;
        metal_dispatch_gemm_bf16(metal_ctx_, cmd, metal_input_, metal_weights_,
                                 metal_output_, static_cast<int>(lanes),
                                 static_cast<int>(output_dim_),
                                 static_cast<int>(input_dim_));
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
        output.resize(output_dim_ * lanes);
        std::memcpy(output.data(), metal_buffer_get_contents(metal_output_),
                    output.size() * sizeof(uint16_t));
        return true;
    }
    if (metal_ready_ && lanes <= 32) {
        if (!metal_rowwise_) {
            // P12 ROOT CAUSE of spec/base divergence: gemv_int4_groupwise
            // (lanes==1) and gemm_int4_groupwise (lanes>1) accumulate K in
            // different fp32 orders, so per-lane projection values depended on
            // the batch width. Base decode (always lanes=1) then disagreed
            // with verify/rebuild (lanes>1) in low-order bits, poisoning the
            // captured np whenever a rounding boundary was crossed.
            // Fix: ALWAYS run the lanes==1 GEMV, looping lanes, so every
            // width is bit-identical to single-lane decode.
            output.assign(output_dim_ * lanes, 0);
            std::vector<uint16_t> lane_in(input_dim_);
            for (size_t l = 0; l < lanes; ++l) {
                for (size_t c = 0; c < input_dim_; ++c)
                    lane_in[c] = input[c * lanes + l];
                std::memcpy(metal_buffer_get_contents(metal_input_), lane_in.data(),
                            input_dim_ * sizeof(uint16_t));
                MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
                if (!cmd) return false;
                metal_dispatch_gemv_int4_groupwise(
                    metal_ctx_, cmd, metal_weights_, metal_scales_,
                    metal_biases_, metal_input_, metal_output_,
                    static_cast<uint32_t>(output_dim_),
                    static_cast<uint32_t>(metal_packed_cols_),
                    static_cast<uint32_t>(input_dim_),
                    static_cast<uint32_t>(metal_groups_));
                metal_command_buffer_commit(cmd);
                metal_command_buffer_wait(cmd);
                const uint16_t* src = static_cast<const uint16_t*>(
                    metal_buffer_get_contents(metal_output_));
                for (size_t r = 0; r < output_dim_; ++r)
                    output[r * lanes + l] = src[r];
            }
            return true;
        }
        std::memcpy(metal_buffer_get_contents(metal_input_), input,
                    input_dim_ * lanes * sizeof(uint16_t));
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
        if (cmd) {
            if (!metal_rowwise_) {
                // Unreachable (handled above); retained for safety.
                metal_dispatch_gemv_int4_groupwise(
                    metal_ctx_, cmd, metal_weights_, metal_scales_,
                    metal_biases_, metal_input_, metal_output_,
                    static_cast<uint32_t>(output_dim_),
                    static_cast<uint32_t>(metal_packed_cols_),
                    static_cast<uint32_t>(input_dim_),
                    static_cast<uint32_t>(metal_groups_));
            } else {
                metal_dispatch_gemm_int4_groupwise(
                    metal_ctx_, cmd, metal_input_, metal_weights_, metal_scales_,
                    metal_biases_, metal_output_, static_cast<int>(output_dim_),
                    static_cast<int>(input_dim_), static_cast<int>(metal_packed_cols_),
                    static_cast<int>(metal_groups_), static_cast<int>(lanes));
            }
            metal_command_buffer_commit(cmd);
            metal_command_buffer_wait(cmd);
            output.resize(output_dim_ * lanes);
            std::memcpy(output.data(), metal_buffer_get_contents(metal_output_),
                        output.size() * sizeof(uint16_t));
            return true;
        }
    }
    if (host_ready_) {
        if (lanes != 1) return false;
        output.assign(output_dim_, 0);
        if (host_groupwise_) {
            if (!host_loader_) return false;
            for (size_t r = 0; r < output_dim_; ++r) {
                float sum = 0.0f;
                if (!host_loader_->dot_row_fp16(host_tensor_name_, r, input,
                                                input_dim_, sum)) return false;
                output[r] = float_to_fp16(sum);
            }
            return true;
        }
        for (size_t r = 0; r < output_dim_; ++r) {
            float sum = 0.0f;
            const uint8_t* packed = host_packed_.data() + r * input_dim_ / 2;
            for (size_t c = 0; c < input_dim_; c += 2) {
                const uint8_t byte = packed[c / 2];
                const int q0 = (byte & 0x0f) < 8 ? (byte & 0x0f) : (byte & 0x0f) - 16;
                const int q1 = ((byte >> 4) & 0x0f) < 8 ? ((byte >> 4) & 0x0f) : ((byte >> 4) & 0x0f) - 16;
                sum += fp16_to_float(input[c]) * (q0 * host_scales_[r]);
                sum += fp16_to_float(input[c + 1]) * (q1 * host_scales_[r]);
            }
            output[r] = float_to_fp16(sum);
        }
        return true;
    }
    IOSurfaceLock(input_surface_, 0, nullptr);
    std::memset(IOSurfaceGetBaseAddress(input_surface_), 0,
                input_dim_ * width_ * sizeof(uint16_t));
    std::memcpy(IOSurfaceGetBaseAddress(input_surface_), input,
                input_dim_ * lanes * sizeof(uint16_t));
    IOSurfaceUnlock(input_surface_, 0, nullptr);
    if (!ane_request_evaluate(ctx_, model_, request_, nullptr, 0, nullptr, 0)) return false;
    output.resize(output_dim_ * lanes);
    IOSurfaceLock(output_surface_, kIOSurfaceLockReadOnly, nullptr);
    std::memcpy(output.data(), IOSurfaceGetBaseAddress(output_surface_),
                output.size() * sizeof(uint16_t));
    IOSurfaceUnlock(output_surface_, kIOSurfaceLockReadOnly, nullptr);
    return true;
}

bool RindiAneProjection::metal_dispatch(
    MetalContext* ctx, MetalCommandBufferHandle cmd,
    MetalBufferHandle input, MetalBufferHandle output, size_t lanes,
    size_t input_offset, size_t output_offset) const {
    if (!metal_ready_ || !ctx || !cmd || !input || !output || lanes == 0 || lanes > 32)
        return false;
    if (metal_rowwise_) {
        if (lanes > 1 && input_offset == 0 && output_offset == 0 &&
                   !std::getenv("RINDI_METAL_GEMM_BATCH") &&
                   !std::getenv("RINDI_DISABLE_BATCH_GEMM")) {
            metal_dispatch_gemm_int4_rw_simd(
                ctx, cmd, input, metal_weights_, metal_scales_,
                output, static_cast<int>(output_dim_), static_cast<int>(input_dim_),
                static_cast<int>(metal_packed_cols_), static_cast<int>(lanes));
        } else if (std::getenv("RINDI_METAL_TAIL_TILED")) {
            metal_dispatch_gemm_int4_rowwise_tiled_offset(
                ctx, cmd, input, input_offset, metal_weights_, metal_scales_,
                output, output_offset, static_cast<int>(output_dim_),
                static_cast<int>(input_dim_), static_cast<int>(metal_packed_cols_),
                static_cast<int>(lanes));
        } else if (lanes > 1 && input_offset == 0 && output_offset == 0 &&
                   !std::getenv("RINDI_DISABLE_BATCH_GEMM")) {
            metal_dispatch_gemm_int4_rowwise_batched(
                ctx, cmd, input, metal_weights_, metal_scales_, output,
                static_cast<int>(output_dim_), static_cast<int>(input_dim_),
                static_cast<int>(metal_packed_cols_), static_cast<int>(lanes));
        } else {
            metal_dispatch_gemm_int4_rowwise_offset(
                ctx, cmd, input, input_offset, metal_weights_, metal_scales_,
                output, output_offset, static_cast<int>(output_dim_),
                static_cast<int>(input_dim_), static_cast<int>(metal_packed_cols_),
                static_cast<int>(lanes));
        }
    } else {
        if (lanes > 1 && input_offset == 0 && output_offset == 0 &&
                   !std::getenv("RINDI_METAL_GEMM_BATCH") &&
                   !std::getenv("RINDI_DISABLE_BATCH_GEMM")) {
            metal_dispatch_gemm_int4_simd(
                ctx, cmd, input, metal_weights_, metal_scales_, metal_biases_,
                output, static_cast<int>(output_dim_), static_cast<int>(input_dim_),
                static_cast<int>(metal_packed_cols_), static_cast<int>(metal_groups_),
                static_cast<int>(lanes));
        } else if (lanes > 1 && input_offset == 0 && output_offset == 0 &&
                   !std::getenv("RINDI_DISABLE_BATCH_GEMM")) {
            metal_dispatch_gemm_int4_groupwise_batch(
                ctx, cmd, input, metal_weights_, metal_scales_, metal_biases_,
                output, static_cast<int>(output_dim_), static_cast<int>(input_dim_),
                static_cast<int>(metal_packed_cols_), static_cast<int>(metal_groups_),
                static_cast<int>(lanes));
        } else {
            metal_dispatch_gemm_int4_groupwise_offset(
                ctx, cmd, input, input_offset, metal_weights_, metal_scales_,
                metal_biases_, output, output_offset, static_cast<int>(output_dim_),
                static_cast<int>(input_dim_), static_cast<int>(metal_packed_cols_),
                static_cast<int>(metal_groups_), static_cast<int>(lanes));
        }
    }
    return true;
}
