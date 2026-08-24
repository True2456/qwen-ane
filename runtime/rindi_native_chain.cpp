/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_native_chain.cpp - Implementation of Pure C++ 64-Layer ANE Execution Chain.
 */

#include "rindi_native_chain.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace {

size_t logical_columns(const TensorInfo* info) {
    if (!info || info->shape.size() < 2) return 0;
    const size_t packed = static_cast<size_t>(info->shape[1]);
    return info->dtype == "U32" ? packed * 8 : packed;
}

const char* kBuildInfo =
    "[buildInfo = dict<string, string>({{"
    "\"coremlc-component-MIL\", \"3510.2.1\"}, "
    "{\"coremlc-version\", \"3505.4.1\"}, "
    "{\"coremltools-component-milinternal\", \"\"}, "
    "{\"coremltools-version\", \"9.0\"}})]";

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    const std::streamsize size = in.tellg();
    if (size < 0) return {};
    std::vector<uint8_t> data(static_cast<size_t>(size));
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(data.data()), size)) return {};
    return data;
}

uint16_t float_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = (bits >> 13) & 0x3ffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign |
        (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t value = sign;
    if (exponent == 0) value |= mantissa << 13;
    else if (exponent == 31) value |= 0x7f800000u | (mantissa << 13);
    else value |= (exponent + 112u) << 23 | (mantissa << 13);
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::string quant_decl(const char* name, size_t rows, size_t cols) {
    std::ostringstream s;
    s << "    tensor<int4, [" << rows << ", " << cols << ", 1, 1]> "
      << name << "q = const()[name=string(\"" << name << "q\"), "
      << "val=tensor<int4, [" << rows << ", " << cols << ", 1, 1]>"
      << "(BLOBFILE(path=string(\"@model_path/weights/" << name
      << ".bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [" << rows << ", 1, 1, 1]> " << name
      << "sc = const()[name=string(\"" << name << "sc\"), "
      << "val=tensor<fp16, [" << rows << ", 1, 1, 1]>"
      << "(BLOBFILE(path=string(\"@model_path/weights/" << name
      << "s.bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [" << rows << ", " << cols << ", 1, 1]> "
      << name << "w = constexpr_blockwise_shift_scale(data=" << name
      << "q, scale=" << name << "sc)[name=string(\"" << name << "dq\")];";
    return s.str();
}

// P14: K-tiled conv for large-K projections. ANE efficiency collapses when
// the reduction (input-channel) dimension exceeds ~2048 (~4.9 -> ~11.7
// TFLOPS, measured in probes/test_ane_prefill_mm.cpp). ANECCompile rejects
// slice_by_index over constexpr_blockwise_shift_scale outputs, so tiling is
// BAKED: compile_layer deinterleaves each raw [rows, ic] nibble blob into
// per-chunk [rows, kc] payloads registered as <base>_k{t}.bin, and the MIL
// declares one int4 const + conv per chunk with partial-sum adds.
struct KTilePlan {
    bool tiled = false;
    int T = 1;
    std::vector<size_t> ks;
};

static KTilePlan ktile_plan(const char* env, size_t ic) {
    KTilePlan p;
    // OPT-IN: RINDI_KTILE_<CONV>=N tiles that projection's K into N chunks.
    // Primitive-level win is proven (2.4x standalone), but multi-tile fused
    // tail programs hit ANECCompile instability on macOS26.3/h17 (clean
    // InvalidMILProgram rejections and intermittent SIGABRT during compile),
    // so until that settles, tails stay monolithic unless explicitly enabled.
    if (!std::getenv(env)) { p.T = 1; return p; }
    p.T = std::atoi(std::getenv(env));
    if (p.T <= 1 || ic <= 2048) { p.T = 1; return p; }
    p.tiled = true;
    p.ks.assign(p.T, (ic / p.T) & ~static_cast<size_t>(1));   // keep nibble-aligned
    size_t rem = ic - p.ks[0] * p.T;
    for (int t = 0; rem > 0; ++t %= p.T) {
        const size_t add = std::min<size_t>(rem, 2);
        p.ks[t] += add; rem -= add;
    }
    for (size_t k : p.ks)
        if (k == 0 || (k & 1)) { p.tiled = false; p.T = 1; break; }  // safety
    return p;
}

static std::string quant_decl_chunk(const char* base, size_t rows, size_t cols) {
    std::ostringstream s;
    s << "    tensor<int4, [" << rows << ", " << cols << ", 1, 1]> "
      << base << "q = const()[name=string(\"" << base << "q\"), "
      << "val=tensor<int4, [" << rows << ", " << cols << ", 1, 1]>"
      << "(BLOBFILE(path=string(\"@model_path/weights/" << base
      << ".bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [" << rows << ", 1, 1, 1]> " << base
      << "sc = const()[name=string(\"" << base << "sc\"), "
      << "val=tensor<fp16, [" << rows << ", 1, 1, 1]>"
      << "(BLOBFILE(path=string(\"@model_path/weights/" << base
      << "s.bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [" << rows << ", " << cols << ", 1, 1]> "
      << base << "w = constexpr_blockwise_shift_scale(data=" << base
      << "q, scale=" << base << "sc)[name=string(\"" << base << "dq\")];";
    return s.str();
}

static std::string ksplit_conv_baked(const char* base, size_t rows, size_t ic,
                                     const char* xexpr, size_t seq,
                                     const char* out_name,
                                     const KTilePlan& plan) {
    std::ostringstream s;
    (void)ic;
    if (!plan.tiled) {
        s << "    tensor<fp16, [1, " << rows << ", 1, " << seq << "]> " << out_name
          << " = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight="
          << base << "w, x=" << xexpr << ")[name=string(\"" << out_name << "\")];\n";
        return s.str();
    }
    size_t off = 0;
    for (int t = 0; t < plan.T; ++t) {
        const size_t kc = plan.ks[t];
        const std::string tag = std::string(base) + "_k" + std::to_string(t);
        s << quant_decl_chunk(tag.c_str(), rows, kc) << "\n";
        s << "    tensor<int32, [4]> " << tag << "_xb = const()[name=string(\"" << tag
          << "_xb\"), val=tensor<int32, [4]>([0," << off << ",0,0])];\n"
          << "    tensor<int32, [4]> " << tag << "_xe = const()[name=string(\"" << tag
          << "_xe\"), val=tensor<int32, [4]>([1," << (off + kc) << ",1," << seq
          << "])];\n"
          << "    tensor<fp16, [1, " << kc << ", 1, " << seq << "]> " << tag
          << "_x = slice_by_index(begin=" << tag << "_xb, end=" << tag << "_xe, x="
          << xexpr << ")[name=string(\"" << tag << "_x\")];\n";
        if (t == 0) {
            s << "    tensor<fp16, [1, " << rows << ", 1, " << seq << "]> " << out_name
              << " = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight="
              << tag << "w, x=" << tag << "_x)[name=string(\"" << out_name << "\")];\n";
        } else {
            s << "    tensor<fp16, [1, " << rows << ", 1, " << seq << "]> " << tag
              << "_p = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight="
              << tag << "w, x=" << tag << "_x)[name=string(\"" << tag << "_p\")];\n"
              << "    tensor<fp16, [1, " << rows << ", 1, " << seq << "]> " << out_name
              << " = add(x=" << out_name << ", y=" << tag << "_p)[name=string(\""
              << out_name << "_a" << t << "\")];\n";
        }
        off += kc;
    }
    return s.str();
}

std::string build_tail_mil(size_t hidden, size_t core, size_t intermediate,
                           size_t seq, size_t next_projection,
                           bool attention_tail) {
    const bool has_next = next_projection != 0;
    // The Python fused graph applies attention/GDN gating before replacing
    // out_proj with identity.  The tail therefore receives only the already
    // gated core followed by the residual hidden state.
    const size_t input_core = core;
    std::ostringstream s;
    s << "program(1.3)\n" << kBuildInfo << "\n{\n"
      << "  func main<ios26>(tensor<fp16, [1, " << (input_core + hidden)
      << ", 1, " << seq << "]> xin) {\n"
      << "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
      << "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
      << "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
      << "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
      << "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
    s
      << quant_decl("o", hidden, core) << "\n"
      << quant_decl("gu", 2 * intermediate, hidden) << "\n"
      << quant_decl("dn", hidden, intermediate) << "\n";
    if (has_next) {
        s << quant_decl("ip", next_projection, hidden) << "\n";
    }
    s << "    tensor<fp16, [1, " << hidden << ", 1, 1]> pnw = const()[name=string(\"pnw\"), "
      << "val=tensor<fp16, [1, " << hidden << ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/pn.bin\"), offset=uint64(64)))];\n"
      << "    tensor<fp16, [1, " << hidden << ", 1, 1]> onw = const()[name=string(\"onw\"), "
      << "val=tensor<fp16, [1, " << hidden << ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/on.bin\"), offset=uint64(64)))];\n";
    if (has_next) {
        s << "    tensor<fp16, [1, " << hidden << ", 1, 1]> ilw = const()[name=string(\"ilw\"), "
          << "val=tensor<fp16, [1, " << hidden << ", 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/il.bin\"), offset=uint64(64)))];\n";
    }
    s << "    tensor<fp16, [1, " << core << ", 1, " << seq << "]> core = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1," << core << ",1," << seq << "]), x=xin)[name=string(\"core\")];\n";
    s << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> res = slice_by_index(begin=tensor<int32, [4]>([0," << core << ",0,0]), end=tensor<int32, [4]>([1," << (input_core + hidden) << ",1," << seq << "]), x=xin)[name=string(\"res\")];\n"
      << "    tensor<fp16, [1, " << core << ", 1, " << seq << "]> gated = identity(x=core)[name=string(\"gated\")];\n";
    const KTilePlan po = ktile_plan("RINDI_KTILE_O", core);
    const KTilePlan pg = ktile_plan("RINDI_KTILE_GU", hidden);
    const KTilePlan pdn = ktile_plan("RINDI_KTILE_DN", intermediate);
    const KTilePlan pip = ktile_plan("RINDI_KTILE_IP", hidden);
    s << ksplit_conv_baked("o", hidden, core, "gated", seq, "r", po)
      << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> h = add(x=res, y=r)[name=string(\"h\")];\n"
      << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> sq = mul(x=h, y=h)[name=string(\"sq\")];\n"
      << "    tensor<fp16, [1, 1, 1, " << seq << "]> ms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=sq)[name=string(\"ms\")];\n"
      << "    fp16 ep = const()[name=string(\"ep\"), val=fp16(0x1.0p-20)];\n"
      << "    tensor<fp16, [1, 1, 1, " << seq << "]> msa = add(x=ms, y=ep)[name=string(\"msa\")];\n"
      << "    tensor<fp16, [1, 1, 1, " << seq << "]> sd = sqrt(x=msa)[name=string(\"sd\")];\n"
      << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> nx = real_div(x=h, y=sd)[name=string(\"nx\")];\n"
      << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> n = mul(x=nx, y=pnw)[name=string(\"n\")];\n"
      << ksplit_conv_baked("gu", 2 * intermediate, hidden, "n", seq, "c", pg)
      << "    tensor<fp16, [1, " << intermediate << ", 1, " << seq << "]> g0 = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1," << intermediate << ",1," << seq << "]), x=c)[name=string(\"g0\")];\n"
      << "    tensor<fp16, [1, " << intermediate << ", 1, " << seq << "]> u0 = slice_by_index(begin=tensor<int32, [4]>([0," << intermediate << ",0,0]), end=tensor<int32, [4]>([1," << (2 * intermediate) << ",1," << seq << "]), x=c)[name=string(\"u0\")];\n"
      << "    tensor<fp16, [1, " << intermediate << ", 1, " << seq << "]> sg = sigmoid(x=g0)[name=string(\"sg\")];\n"
      << "    tensor<fp16, [1, " << intermediate << ", 1, " << seq << "]> si = mul(x=g0, y=sg)[name=string(\"si\")];\n"
      << "    tensor<fp16, [1, " << intermediate << ", 1, " << seq << "]> ac = mul(x=si, y=u0)[name=string(\"ac\")];\n"
      << ksplit_conv_baked("dn", hidden, intermediate, "ac", seq, "m", pdn);
    if (has_next) {
        s << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> o0 = add(x=h, y=m)[name=string(\"o0\")];\n"
          << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> y = identity(x=o0)[name=string(\"y\")];\n"
          << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> nsq = mul(x=o0, y=o0)[name=string(\"nsq\")];\n"
          << "    tensor<fp16, [1, 1, 1, " << seq << "]> nms = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=nsq)[name=string(\"nms\")];\n"
          << "    tensor<fp16, [1, 1, 1, " << seq << "]> nmsa = add(x=nms, y=ep)[name=string(\"nmsa\")];\n"
          << "    tensor<fp16, [1, 1, 1, " << seq << "]> nsd = sqrt(x=nmsa)[name=string(\"nsd\")];\n"
          << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> nnx = real_div(x=o0, y=nsd)[name=string(\"nnx\")];\n"
          << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> nn = mul(x=nnx, y=ilw)[name=string(\"nn\")];\n"
          << ksplit_conv_baked("ip", next_projection, hidden, "nn", seq, "y2", pip)
          << "  } -> (y, y2);\n";
    } else {
        s << "    tensor<fp16, [1, " << hidden << ", 1, " << seq << "]> y = add(x=h, y=m)[name=string(\"y\")];\n"
          << "  } -> (y);\n";
    }
    s << "}\n";
    return s.str();
}

} // namespace

RindiNativeChain::RindiNativeChain(size_t hidden_dim, size_t seq_len)
    : hidden_dim_(hidden_dim),
      seq_len_(seq_len),
      // evaluate_step() is the decode-step API and accepts one hidden state.
      // Keep the surface sized to that state; using seq_len here causes the
      // input/output memcpy calls below to read/write past their buffers.
      surface_bytes_(hidden_dim * sizeof(uint16_t)),
      ane_ctx_(nullptr),
      metal_ctx_(nullptr),
      surf_a_(nullptr),
      surf_b_(nullptr),
      last_eval_ms_(0.0) {

    ane_ctx_ = ane_context_create();
    metal_ctx_ = metal_context_create();
    
    // Allocate 2 ping-pong IOSurface buffers
    surf_a_ = metal_create_iosurface(surface_bytes_);
    surf_b_ = metal_create_iosurface(surface_bytes_);
}

RindiNativeChain::~RindiNativeChain() {
    for (auto& entry : layers_) {
        if (entry.req_a_to_b) ane_request_release(entry.req_a_to_b);
        if (entry.req_b_to_a) ane_request_release(entry.req_b_to_a);
        if (entry.model) ane_model_release(entry.model);
        if (entry.input_surface && entry.input_surface != surf_a_ && entry.input_surface != surf_b_)
            CFRelease(entry.input_surface);
        if (entry.output_surface && entry.output_surface != surf_a_ && entry.output_surface != surf_b_)
            CFRelease(entry.output_surface);
        if (entry.projection_surface) CFRelease(entry.projection_surface);
    }
    layers_.clear();

    if (metal_tail_core_) metal_buffer_release(metal_tail_core_);
    if (metal_tail_residual_) metal_buffer_release(metal_tail_residual_);
    if (metal_tail_work_) metal_buffer_release(metal_tail_work_);
    if (metal_tail_norm_) metal_buffer_release(metal_tail_norm_);
    if (metal_tail_ff_) metal_buffer_release(metal_tail_ff_);
    if (metal_tail_activation_) metal_buffer_release(metal_tail_activation_);
    if (metal_tail_next_) metal_buffer_release(metal_tail_next_);
    
    if (surf_a_) CFRelease(surf_a_);
    if (surf_b_) CFRelease(surf_b_);
    
    if (ane_ctx_) ane_context_destroy(ane_ctx_);
    if (metal_ctx_) metal_context_destroy(metal_ctx_);
}

bool RindiNativeChain::load_layer(int layer_idx, const std::string& package_path) {
    if (!ane_ctx_) return false;

    // The Python ANE compiler cache stores loaded programs as `__.bin` and
    // `__s.bin` inside the package directory. Older exports may contain
    // model.hwx/model.mil instead. Let the private framework decide whether
    // the directory is loadable rather than rejecting the cache format here.
    ANEModel* model = ane_model_load_compiled(ane_ctx_, package_path.c_str(), "q38_layer", 0);
    if (!model) {
        return false;
    }
    
    ANERequest* req_ab = ane_request_create(ane_ctx_, model, surf_a_, surf_b_, 0);
    ANERequest* req_ba = ane_request_create(ane_ctx_, model, surf_b_, surf_a_, 0);
    
    if (layer_idx >= (int)layers_.size()) layers_.resize(layer_idx + 1);
    
    auto& entry = layers_[layer_idx];
    entry.model = model;
    entry.req_a_to_b = req_ab;
    entry.req_b_to_a = req_ba;
    entry.input_surface = surf_a_;
    entry.output_surface = surf_b_;
    entry.projection_surface = nullptr;
    entry.input_channels = hidden_dim_;
    entry.projection_channels = 0;
    entry.core_dim = 0;
    entry.intermediate = 0;
    entry.attention = false;
    entry.metal_tail.reset();
    return true;
}

// Deinterleave a raw row-major nibble blob [rows, ic] into per-K-chunk
// payloads [rows, kc_t] (byte segments are contiguous within each row).
static bool split_nibbles(const std::vector<uint8_t>& raw, size_t rows,
                          size_t ic, const KTilePlan& plan,
                          std::vector<std::vector<uint8_t>>& data_out,
                          std::vector<std::vector<uint8_t>>& scale_out,
                          const std::vector<uint8_t>& scales) {
    data_out.assign(plan.T, {});
    scale_out.assign(plan.T, scales);
    for (int t = 0; t < plan.T; ++t)
        data_out[t].resize((rows * plan.ks[t] + 1) / 2);
    size_t col = 0;
    for (int t = 0; t < plan.T; ++t) {
        const size_t kc = plan.ks[t], seg = kc / 2;
        for (size_t r = 0; r < rows; ++r) {
            const uint8_t* src = raw.data() + r * (ic / 2) + col / 2;
            std::memcpy(data_out[t].data() + r * seg, src, seg);
        }
        col += kc;
    }
    return true;
}

bool RindiNativeChain::compile_layer(int layer_idx, const std::string& package_path,
                                     const SafeTensorsLoader& loader) {
    if (!ane_ctx_ || layer_idx < 0) return false;
    const size_t H = hidden_dim_;
    const std::string layer = "layers." + std::to_string(layer_idx) + ".";
    const bool is_attention = loader.has_tensor(layer + "self_attn.o_proj.weight");
    const std::string core_name = layer + (is_attention ? "self_attn.o_proj.weight"
                                                         : "linear_attn.out_proj.weight");
    const std::string gate_name = layer + "mlp.gate_proj.weight";
    const std::string down_name = layer + "mlp.down_proj.weight";
    const TensorInfo* core_info = loader.get_tensor_info(core_name);
    const TensorInfo* gate_info = loader.get_tensor_info(gate_name);
    if (!core_info || !gate_info || core_info->shape.size() < 2 || gate_info->shape.size() < 2) return false;
    const size_t core_dim = logical_columns(core_info);
    const size_t intermediate = static_cast<size_t>(gate_info->shape[0]);
    size_t next_projection = 0;
    if (layer_idx + 1 < 64) {
        const std::string np = "layers." + std::to_string(layer_idx + 1) + ".";
        if (loader.has_tensor(np + "linear_attn.in_proj_qkv.weight")) {
            for (const char* n : {"linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
                                  "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"}) {
                const TensorInfo* ti = loader.get_tensor_info(np + n);
                if (!ti || ti->shape.empty()) return false;
                next_projection += static_cast<size_t>(ti->shape[0]);
            }
        } else {
            for (const char* n : {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"}) {
                const TensorInfo* ti = loader.get_tensor_info(np + n);
                if (!ti || ti->shape.empty()) return false;
                next_projection += static_cast<size_t>(ti->shape[0]);
            }
        }
    }

    std::vector<std::string> names = {"o.bin", "os.bin", "gu.bin", "gus.bin",
                                      "dn.bin", "dns.bin", "pn.bin", "on.bin"};
    std::vector<std::vector<uint8_t>> payloads;
    payloads.reserve(12);
    const KTilePlan po_l = ktile_plan("RINDI_KTILE_O", core_dim);
    const KTilePlan pg_l = ktile_plan("RINDI_KTILE_GU", H);
    const KTilePlan pdn_l = ktile_plan("RINDI_KTILE_DN", intermediate);
    const KTilePlan pip_l = ktile_plan("RINDI_KTILE_IP", H);
    for (const char* kind : {"o", "gu", "dn"}) {
        auto data = read_file(package_path + "/chain" + std::to_string(layer_idx) + "." + kind + "/__.bin");
        auto scale = read_file(package_path + "/chain" + std::to_string(layer_idx) + "." + kind + "/__s.bin");
        if (data.empty() || scale.empty()) return false;
        const KTilePlan& pl = (kind[0]=='o') ? po_l : (kind[0]=='g' ? pg_l : pdn_l);
        if (!pl.tiled) {
            payloads.push_back(std::move(data));
            payloads.push_back(std::move(scale));
            continue;
        }
        const size_t ric = (kind[0]=='o') ? core_dim : (kind[0]=='g' ? H : intermediate);
        const size_t roc = (kind[0]=='g') ? 2 * intermediate : hidden_dim_;
        std::vector<std::vector<uint8_t>> dchunks, schunks;
        split_nibbles(data, roc, ric, pl, dchunks, schunks, scale);
        for (int t = 0; t < pl.T; ++t) {
            names.push_back(std::string(kind) + "_k" + std::to_string(t) + ".bin");
            payloads.push_back(std::move(dchunks[t]));
            names.push_back(std::string(kind) + "_k" + std::to_string(t) + "s.bin");
            payloads.push_back(schunks[t]);
        }
    }
    std::vector<uint16_t> norm;
    if (!loader.get_tensor_fp16(layer + "post_attention_layernorm.weight", norm)) return false;
    payloads.emplace_back(reinterpret_cast<const uint8_t*>(norm.data()),
                          reinterpret_cast<const uint8_t*>(norm.data()) + norm.size() * sizeof(uint16_t));
    // The tail's RMS reduction is a grouped convolution over H channels.  Its
    // kernel must compute mean(square), not sum(square); using ones here
    // shrinks every chained projection by sqrt(H) and effectively disconnects
    // layers 1..63 from the scheduler.
    std::vector<uint16_t> mean(H, float_to_fp16(1.0f / static_cast<float>(H)));
    payloads.emplace_back(reinterpret_cast<const uint8_t*>(mean.data()),
                          reinterpret_cast<const uint8_t*>(mean.data()) + mean.size() * sizeof(uint16_t));
    if (next_projection) {
        auto data = read_file(package_path + "/chain" + std::to_string(layer_idx) + ".ip/__.bin");
        auto scale = read_file(package_path + "/chain" + std::to_string(layer_idx) + ".ip/__s.bin");
        if (data.empty() || scale.empty()) return false;
        if (!pip_l.tiled) {
            names.push_back("ip.bin"); names.push_back("ips.bin");
            payloads.push_back(std::move(data)); payloads.push_back(std::move(scale));
        } else {
            std::vector<std::vector<uint8_t>> dchunks, schunks;
            split_nibbles(data, next_projection, H, pip_l, dchunks, schunks, scale);
            for (int t = 0; t < pip_l.T; ++t) {
                names.push_back("ip_k" + std::to_string(t) + ".bin");
                payloads.push_back(std::move(dchunks[t]));
                names.push_back("ip_k" + std::to_string(t) + "s.bin");
                payloads.push_back(schunks[t]);
            }
        }
        std::vector<uint16_t> next_norm;
        if (!loader.get_tensor_fp16("layers." + std::to_string(layer_idx + 1) + ".input_layernorm.weight", next_norm)) return false;
        names.push_back("il.bin");
        payloads.emplace_back(reinterpret_cast<const uint8_t*>(next_norm.data()),
                              reinterpret_cast<const uint8_t*>(next_norm.data()) + next_norm.size() * sizeof(uint16_t));
    }
    const std::string mil = build_tail_mil(H, core_dim, intermediate, seq_len_,
                                            next_projection, is_attention);
    if (std::getenv("RINDI_DUMP_TAIL")) {
        char pf[96];
        std::snprintf(pf, sizeof(pf), "/tmp/tail_L%zu_s%zu.mil", (size_t)layer_idx, seq_len_);
        FILE* f = fopen(pf, "w");
        if (f) { fwrite(mil.data(), 1, mil.size(), f); fclose(f); }
    }
    std::vector<const char*> name_ptrs;
    std::vector<const void*> data_ptrs;
    std::vector<size_t> sizes;
    for (size_t i = 0; i < names.size(); ++i) {
        name_ptrs.push_back(names[i].c_str());
        data_ptrs.push_back(payloads[i].data());
        sizes.push_back(payloads[i].size());
    }
    ANEModel* model = ane_model_compile_mil(ane_ctx_, mil.c_str(), name_ptrs.data(),
                                             data_ptrs.data(), sizes.data(), names.size(), 0, 21);
    if (!model) return false;

    // Every fused tail has the same ABI: the already-gated core followed by
    // the residual hidden state, matching the Python AneFusedLayer graph.
    const size_t input_channels = core_dim + H;
    IOSurfaceRef in = metal_create_iosurface(input_channels * seq_len_ * sizeof(uint16_t));
    IOSurfaceRef out = metal_create_iosurface(H * seq_len_ * sizeof(uint16_t));
    IOSurfaceRef proj = next_projection
        ? metal_create_iosurface(next_projection * seq_len_ * sizeof(uint16_t)) : nullptr;
    if (!in || !out || (next_projection && !proj)) {
        if (in) CFRelease(in); if (out) CFRelease(out); if (proj) CFRelease(proj);
        ane_model_release(model); return false;
    }
    ANERequest* req = nullptr;
    if (next_projection) {
        size_t channels[2] = {0, 0};
        const size_t count = ane_model_output_channels(model, channels, 2);
        IOSurfaceRef outputs[2] = {nullptr, nullptr};
        if (count != 2) {
            CFRelease(in); CFRelease(out); CFRelease(proj);
            ane_model_release(model);
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            if (channels[i] == H) outputs[i] = out;
            else if (channels[i] == next_projection) outputs[i] = proj;
            else {
                CFRelease(in); CFRelease(out); CFRelease(proj);
                ane_model_release(model);
                return false;
            }
        }
        req = ane_request_create_multi(ane_ctx_, model, in, outputs, 2, 0);
    } else {
        req = ane_request_create(ane_ctx_, model, in, out, 0);
    }
    if (!req) {
        CFRelease(in); CFRelease(out); if (proj) CFRelease(proj);
        ane_model_release(model); return false;
    }
    if (layer_idx >= static_cast<int>(layers_.size())) layers_.resize(layer_idx + 1);
    auto& old = layers_[layer_idx];
    if (old.req_a_to_b) ane_request_release(old.req_a_to_b);
    if (old.req_b_to_a) ane_request_release(old.req_b_to_a);
    if (old.model) ane_model_release(old.model);
    if (old.input_surface && old.input_surface != surf_a_) CFRelease(old.input_surface);
    if (old.output_surface && old.output_surface != surf_b_) CFRelease(old.output_surface);
    if (old.projection_surface) CFRelease(old.projection_surface);
    old.model = model;
    old.req_a_to_b = req;
    old.req_b_to_a = nullptr;
    old.input_surface = in;
    old.output_surface = out;
    old.projection_surface = proj;
    old.input_channels = input_channels;
    old.projection_channels = next_projection;
    old.core_dim = core_dim;
    old.intermediate = intermediate;
    old.attention = is_attention;
    old.written_lanes = 0;
    old.metal_tail.reset();
    return true;
}

bool RindiNativeChain::compile_metal_tails(const SafeTensorsLoader& loader,
                                           const std::string& package_path) {
    if (!std::getenv("RINDI_ENABLE_METAL_TAIL") ||
        std::getenv("RINDI_DISABLE_METAL_TAIL")) return false;
    bool all_ready = true;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
        auto& entry = layers_[layer];
        if (!entry.model || !entry.core_dim ||
            !compile_metal_tail(static_cast<int>(layer), loader, package_path,
                                entry.core_dim,
                                entry.intermediate, entry.projection_channels,
                                entry.attention)) all_ready = false;
    }
    return all_ready;
}

bool RindiNativeChain::compile_metal_tail(int layer_idx,
                                          const SafeTensorsLoader& loader,
                                          const std::string& package_path,
                                          size_t core_dim,
                                          size_t intermediate,
                                          size_t next_projection,
                                          bool attention) {
    (void)attention;
    if (!metal_ctx_ || layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()) ||
        std::getenv("RINDI_DISABLE_METAL_TAIL")) return false;

    const std::string prefix = "layers." + std::to_string(layer_idx) + ".";
    const std::string chain = package_path + "/chain" + std::to_string(layer_idx) + ".";
    auto tail = std::make_unique<MetalTail>();
    tail->out_proj = std::make_unique<RindiAneProjection>();
    tail->gate_proj = std::make_unique<RindiAneProjection>();
    tail->down_proj = std::make_unique<RindiAneProjection>();
    if (!tail->out_proj->compile_chain_int4(
            metal_ctx_, chain + "o/__.bin", chain + "o/__s.bin", core_dim, hidden_dim_) ||
        !tail->gate_proj->compile_chain_int4(
            metal_ctx_, chain + "gu/__.bin", chain + "gu/__s.bin",
            hidden_dim_, 2 * intermediate) ||
        !tail->down_proj->compile_chain_int4(
            metal_ctx_, chain + "dn/__.bin", chain + "dn/__s.bin",
            intermediate, hidden_dim_) ||
        !tail->out_proj->metal_ready() || !tail->gate_proj->metal_ready() ||
        !tail->down_proj->metal_ready()) return false;
    if (tail->out_proj->input_dim() != core_dim ||
        tail->out_proj->output_dim() != hidden_dim_ ||
        tail->gate_proj->input_dim() != hidden_dim_ ||
        tail->gate_proj->output_dim() != 2 * intermediate ||
        tail->down_proj->input_dim() != intermediate ||
        tail->down_proj->output_dim() != hidden_dim_) return false;

    std::vector<uint16_t> post_norm;
    if (!loader.get_tensor_fp16(prefix + "post_attention_layernorm.weight", post_norm) ||
        post_norm.size() != hidden_dim_) return false;
    tail->post_norm = metal_buffer_create(metal_ctx_, hidden_dim_ * sizeof(uint16_t));
    if (!tail->post_norm) return false;
    std::memcpy(metal_buffer_get_contents(tail->post_norm), post_norm.data(),
                hidden_dim_ * sizeof(uint16_t));

    if (next_projection) {
        auto projection = std::make_unique<RindiAneProjection>();
        if (!projection->compile_chain_int4(
                metal_ctx_, chain + "ip/__.bin", chain + "ip/__s.bin",
                hidden_dim_, next_projection) || !projection->metal_ready()) return false;
        tail->next_offsets.push_back(0);
        tail->next_proj.push_back(std::move(projection));
        tail->input_norm = metal_buffer_create(metal_ctx_, hidden_dim_ * sizeof(uint16_t));
        if (!tail->input_norm) return false;
        std::vector<uint16_t> input_norm;
        if (!loader.get_tensor_fp16("layers." + std::to_string(layer_idx + 1) +
                                    ".input_layernorm.weight", input_norm) ||
            input_norm.size() != hidden_dim_) return false;
        std::memcpy(metal_buffer_get_contents(tail->input_norm), input_norm.data(),
                    hidden_dim_ * sizeof(uint16_t));
    }

    const size_t core_capacity = std::max<size_t>(6144, core_dim);
    const size_t next_capacity = std::max(metal_tail_next_capacity_, next_projection);
    const size_t intermediate_capacity = std::max(metal_tail_intermediate_, intermediate);
    auto replace_buffer = [&](MetalBufferHandle& buffer, size_t bytes) -> bool {
        if (buffer) metal_buffer_release(buffer);
        buffer = metal_buffer_create(metal_ctx_, bytes);
        return buffer != nullptr;
    };
    // All tails are compiled during initialization, so growing these shared
    // scratch buffers cannot invalidate an in-flight request.
    if (core_capacity > metal_tail_core_capacity_ || !metal_tail_core_) {
        if (!replace_buffer(metal_tail_core_, core_capacity * 32 * sizeof(uint16_t))) return false;
        metal_tail_core_capacity_ = core_capacity;
    }
    if (!metal_tail_residual_ && !replace_buffer(metal_tail_residual_, hidden_dim_ * 32 * sizeof(uint16_t))) return false;
    if (!metal_tail_work_ && !replace_buffer(metal_tail_work_, hidden_dim_ * 32 * sizeof(uint16_t))) return false;
    if (!metal_tail_norm_ && !replace_buffer(metal_tail_norm_, hidden_dim_ * 32 * sizeof(uint16_t))) return false;
    if (intermediate_capacity > metal_tail_intermediate_ || !metal_tail_ff_ || !metal_tail_activation_) {
        if (!replace_buffer(metal_tail_ff_, 2 * intermediate_capacity * 32 * sizeof(uint16_t)) ||
            !replace_buffer(metal_tail_activation_, intermediate_capacity * 32 * sizeof(uint16_t))) return false;
        metal_tail_intermediate_ = intermediate_capacity;
    }
    if (next_capacity > metal_tail_next_capacity_ || (next_capacity && !metal_tail_next_)) {
        if (next_capacity && !replace_buffer(metal_tail_next_, next_capacity * 32 * sizeof(uint16_t))) return false;
        metal_tail_next_capacity_ = next_capacity;
    }
    tail->core_dim = core_dim;
    tail->intermediate = intermediate;
    tail->next_projection = next_projection;
    tail->ready = true;
    layers_[layer_idx].metal_tail = std::move(tail);
    metal_tail_ready_ = true;
    return true;
}

bool RindiNativeChain::evaluate_tail(int layer_idx, const uint16_t* core,
                                     size_t core_dim, const uint16_t* residual,
                                     std::vector<uint16_t>& output,
                                     std::vector<uint16_t>* next_projection) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()) || !core || !residual) {
        std::cerr << "[RindiNativeChain] tail invalid arguments" << std::endl;
        return false;
    }
    auto& e = layers_[layer_idx];
    if (!e.model || !e.req_a_to_b || e.input_channels != core_dim + hidden_dim_) {
        std::cerr << "[RindiNativeChain] tail shape/model mismatch: model=" << (e.model != nullptr)
                  << " req=" << (e.req_a_to_b != nullptr) << " input=" << e.input_channels
                  << " expected=" << (core_dim + hidden_dim_) << std::endl;
        return false;
    }
    // The input surface is zeroed once and only rewritten when the live lane
    // count changes: decode writes lane 0 of every channel forever, so a
    // per-call memset of the full [C, 32] surface (~1.4 MB x 64 layers) was
    // pure overhead. CPU writes complete before the synchronous evaluate, and
    // these surfaces are process-private, so no IOSurface lock is required.
    auto* in = static_cast<uint16_t*>(metal_iosurface_get_base_address(e.input_surface));
    if (e.written_lanes != 1) {
        std::memset(in, 0, e.input_channels * seq_len_ * sizeof(uint16_t));
        e.written_lanes = 1;
    }
    for (size_t c = 0; c < core_dim; ++c) in[c * seq_len_] = core[c];
    for (size_t c = 0; c < hidden_dim_; ++c) in[(core_dim + c) * seq_len_] = residual[c];
    if (!ane_request_evaluate(ane_ctx_, e.model, e.req_a_to_b, nullptr, 0, nullptr, 0)) return false;
    output.resize(hidden_dim_);
    const uint16_t* out = static_cast<const uint16_t*>(metal_iosurface_get_base_address(e.output_surface));
    for (size_t c = 0; c < hidden_dim_; ++c) output[c] = out[c * seq_len_];
    if (next_projection) {
        if (!e.projection_surface || !e.projection_channels) return false;
        next_projection->resize(e.projection_channels);
        const uint16_t* p = static_cast<const uint16_t*>(metal_iosurface_get_base_address(e.projection_surface));
        for (size_t c = 0; c < e.projection_channels; ++c) (*next_projection)[c] = p[c * seq_len_];
    }
    return true;
}

bool RindiNativeChain::evaluate_tail_batch(int layer_idx, const uint16_t* core,
                                           size_t core_dim, const uint16_t* residual,
                                           size_t lanes,
                                           std::vector<uint16_t>& output,
                                           std::vector<uint16_t>* next_projection) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()) ||
        !core || !residual || lanes == 0 || lanes > seq_len_) return false;
    auto& e = layers_[layer_idx];
    if (!e.model || !e.req_a_to_b || e.input_channels != core_dim + hidden_dim_) return false;

    const bool compare_metal_tail = std::getenv("RINDI_COMPARE_METAL_TAIL") != nullptr;
    std::vector<uint16_t> metal_output;
    std::vector<uint16_t> metal_next;
    if (metal_tail_ready_ && e.metal_tail && e.metal_tail->ready &&
        lanes <= 32 && std::getenv("RINDI_ENABLE_METAL_TAIL") &&
        !std::getenv("RINDI_DISABLE_METAL_TAIL")) {
        std::vector<uint16_t>* metal_next_ptr = next_projection ? &metal_next : nullptr;
        if (evaluate_tail_batch_metal(layer_idx, core, core_dim, residual,
                                      lanes, metal_output, metal_next_ptr)) {
            if (!compare_metal_tail) {
                output = std::move(metal_output);
                if (next_projection) *next_projection = std::move(metal_next);
                return true;
            }
            last_metal_batch_output_ = metal_output;
        }
    }

    // Lane-width change is the only event that requires re-zeroing: columns
    // [lanes, written_lanes) may hold stale values from a wider chunk.
    auto* in = static_cast<uint16_t*>(metal_iosurface_get_base_address(e.input_surface));
    // P9: ALWAYS zero the full width before staging. The fused ANE tail is
    // empirically NOT lane-count invariant (probes/test_tail_sync.cpp): the
    // same lane-0 inputs give different outputs depending on total filled
    // columns. Zero-padding every call to seq_len_ makes the width context
    // constant => outputs become width-invariant, which speculative decoding
    // requires for exactness (verify k+1 lanes vs sequential 1 lane).
    std::memset(in, 0, e.input_channels * seq_len_ * sizeof(uint16_t));
    e.written_lanes = lanes;
    for (size_t c = 0; c < core_dim; ++c)
        std::memcpy(in + c * seq_len_, core + c * lanes, lanes * sizeof(uint16_t));
    for (size_t c = 0; c < hidden_dim_; ++c)
        std::memcpy(in + (core_dim + c) * seq_len_, residual + c * lanes,
                    lanes * sizeof(uint16_t));

    const bool dbg_surf = std::getenv("RINDI_DEBUG_SURF") != nullptr && layer_idx == 0;
    unsigned long long dbg_in_h = 0;
    if (dbg_surf) {
        const uint16_t* inb = static_cast<const uint16_t*>(
            metal_iosurface_get_base_address(e.input_surface));
        unsigned long long h = 1469598103934665603ull;
        for (size_t i = 0; i < e.input_channels * seq_len_; ++i) { h ^= inb[i]; h *= 1099511628211ull; }
        dbg_in_h = h;
    }
    if (!ane_request_evaluate(ane_ctx_, e.model, e.req_a_to_b, nullptr, 0, nullptr, 0))
        return false;
    output.resize(hidden_dim_ * lanes);
    const uint16_t* out = static_cast<const uint16_t*>(metal_iosurface_get_base_address(e.output_surface));
    for (size_t c = 0; c < hidden_dim_; ++c)
        std::memcpy(output.data() + c * lanes, out + c * seq_len_,
                    lanes * sizeof(uint16_t));

    if (next_projection) {
        if (!e.projection_surface || !e.projection_channels) return false;
        next_projection->resize(e.projection_channels * lanes);
        const uint16_t* p = static_cast<const uint16_t*>(metal_iosurface_get_base_address(e.projection_surface));
        for (size_t c = 0; c < e.projection_channels; ++c)
            std::memcpy(next_projection->data() + c * lanes, p + c * seq_len_,
                        lanes * sizeof(uint16_t));
        if (dbg_surf) {
            const uint16_t* outb = static_cast<const uint16_t*>(
                metal_iosurface_get_base_address(e.output_surface));
            unsigned long long ho = 1469598103934665603ull;
            for (size_t i = 0; i < hidden_dim_ * seq_len_; ++i) { ho ^= outb[i]; ho *= 1099511628211ull; }
            unsigned long long hp = 1469598103934665603ull;
            for (size_t i = 0; i < e.projection_channels * seq_len_; ++i) { hp ^= p[i]; hp *= 1099511628211ull; }
            // Column-0-only hashes too: isolates whether the live column itself differs.
            unsigned long long hc0o = 1469598103934665603ull, hc0p = 1469598103934665603ull;
            for (size_t c = 0; c < hidden_dim_; ++c) { hc0o ^= outb[c * seq_len_]; hc0o *= 1099511628211ull; }
            for (size_t c = 0; c < e.projection_channels; ++c) { hc0p ^= p[c * seq_len_]; hc0p *= 1099511628211ull; }
            std::fprintf(stderr,
                         "[SURF] L0 lanes=%zu in=%llx out=%llx np=%llx col0_out=%llx col0_np=%llx\n",
                         lanes, (unsigned long long)dbg_in_h, (unsigned long long)ho,
                         (unsigned long long)hp, (unsigned long long)hc0o, (unsigned long long)hc0p);
        }
    }
    if (compare_metal_tail && !metal_output.empty()) {
        auto compare_tensor = [](const std::vector<uint16_t>& ref,
                                 const std::vector<uint16_t>& test,
                                 double& max_abs, double& sum_abs,
                                 size_t& count, size_t& ref_nonfinite,
                                 size_t& test_nonfinite) {
            const size_t n = std::min(ref.size(), test.size());
            for (size_t i = 0; i < n; ++i) {
                const float ref_value = fp16_to_float(ref[i]);
                const float test_value = fp16_to_float(test[i]);
                if (!std::isfinite(ref_value)) ++ref_nonfinite;
                if (!std::isfinite(test_value)) ++test_nonfinite;
                const double diff = std::abs(static_cast<double>(ref_value) -
                                             static_cast<double>(test_value));
                max_abs = std::max(max_abs, diff);
                sum_abs += diff;
            }
            count += n;
        };
        double output_max = 0.0, output_sum = 0.0;
        size_t output_count = 0, output_ref_nonfinite = 0, output_metal_nonfinite = 0;
        compare_tensor(output, metal_output, output_max, output_sum, output_count,
                       output_ref_nonfinite, output_metal_nonfinite);
        double next_max = 0.0, next_sum = 0.0;
        size_t next_count = 0, next_ref_nonfinite = 0, next_metal_nonfinite = 0;
        if (next_projection && metal_next.size() == next_projection->size()) {
            compare_tensor(*next_projection, metal_next, next_max, next_sum, next_count,
                           next_ref_nonfinite, next_metal_nonfinite);
        }
        const double max_abs = std::max(output_max, next_max);
        const double sum_abs = output_sum + next_sum;
        const size_t count = output_count + next_count;
        const size_t ref_nonfinite = output_ref_nonfinite + next_ref_nonfinite;
        const size_t metal_nonfinite = output_metal_nonfinite + next_metal_nonfinite;
        /* Keep the combined line for existing log parsers, but expose the two
           tensors independently so the first divergent operation is visible. */
        std::cerr << "[RindiMetalCompare] layer=" << layer_idx
                  << " output_max_abs=" << output_max
                  << " output_mean_abs=" << (output_count ? output_sum / output_count : 0.0)
                  << " next_max_abs=" << next_max
                  << " next_mean_abs=" << (next_count ? next_sum / next_count : 0.0)
                  << " max_abs=" << max_abs
                  << " mean_abs=" << (count ? sum_abs / count : 0.0)
                  << " ref_nonfinite=" << ref_nonfinite
                  << " metal_nonfinite=" << metal_nonfinite
                  << " output_ref_nonfinite=" << output_ref_nonfinite
                  << " output_metal_nonfinite=" << output_metal_nonfinite
                  << " next_ref_nonfinite=" << next_ref_nonfinite
                  << " next_metal_nonfinite=" << next_metal_nonfinite
                  << " ref0=0x" << std::hex << output[0]
                  << " metal0=0x" << metal_output[0] << std::dec
                  << " output_elems=" << output.size()
                  << " metal_output_elems=" << metal_output.size() << std::endl;
    }
    return true;
}

bool RindiNativeChain::evaluate_tail_batch_metal(
    int layer_idx, const uint16_t* core, size_t core_dim,
    const uint16_t* residual, size_t lanes, std::vector<uint16_t>& output,
    std::vector<uint16_t>* next_projection) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers_.size()) ||
        !core || !residual || lanes == 0 || lanes > 32 || !metal_ctx_ ||
        !metal_tail_core_ || !metal_tail_residual_ || !metal_tail_work_ ||
        !metal_tail_norm_ || !metal_tail_ff_ || !metal_tail_activation_) return false;
    auto& t = *layers_[layer_idx].metal_tail;
    if (!t.ready || t.core_dim != core_dim ||
        ((next_projection != nullptr) != (t.next_projection != 0)) ||
        (next_projection && t.next_projection == 0)) {
        if (std::getenv("RINDI_DEBUG_METAL_TAIL")) {
            std::cerr << "[RindiMetalTail] shape reject layer=" << layer_idx
                      << " ready=" << t.ready << " core=" << core_dim
                      << " expected_core=" << t.core_dim
                      << " next_ptr=" << (next_projection != nullptr)
                      << " next_channels=" << t.next_projection
                      << " actual_next=" << (next_projection ? next_projection->size() : 0)
                      << " lanes=" << lanes << std::endl;
        }
        return false;
    }
    if (next_projection) next_projection->resize(t.next_projection * lanes);


    // --- env-gated tail profiling (RINDI_TAIL_PROFILE=1): aggregate where the
    //     17 ms/layer goes: cpu staging copies, encoder setup, submit+exec.
    struct TailProf {
        std::chrono::high_resolution_clock::time_point t0;
        double in_ms = 0, enc_ms = 0, gpu_ms = 0, out_ms = 0;
    };
    static thread_local TailProf tp;
    static thread_local int tp_n = 0;
    const bool tp_on = std::getenv("RINDI_TAIL_PROFILE") != nullptr;
    auto tp_now = []() { return std::chrono::high_resolution_clock::now(); };
    auto tp_ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    tp.t0 = tp_now();
    std::memcpy(metal_buffer_get_contents(metal_tail_core_), core,
                core_dim * lanes * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(metal_tail_residual_), residual,
                hidden_dim_ * lanes * sizeof(uint16_t));
    MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
    if (!cmd) return false;

    const size_t lane_bytes = lanes * sizeof(uint16_t);
    if (tp_on) tp.in_ms += tp_ms(tp.t0, tp_now());
    auto tenc0 = tp_now();
    if (!t.out_proj->metal_dispatch(metal_ctx_, cmd, metal_tail_core_,
                                     metal_tail_work_, lanes)) {
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
        return false;
    }
    // h = residual + out_proj(core). The result is intentionally written
    // in-place into the out projection buffer before normalization.
    metal_dispatch_add_channel_fp16(metal_ctx_, cmd, metal_tail_work_,
                                    metal_tail_residual_, metal_tail_work_,
                                    static_cast<int>(hidden_dim_), static_cast<int>(lanes));
    metal_dispatch_rmsnorm_channel_fp16(metal_ctx_, cmd, metal_tail_work_,
                                        t.post_norm, metal_tail_norm_,
                                        static_cast<int>(hidden_dim_), static_cast<int>(lanes),
                                        1.0e-6f);
    if (!t.gate_proj->metal_dispatch(metal_ctx_, cmd, metal_tail_norm_,
                                      metal_tail_ff_, lanes)) {
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
        return false;
    }
    metal_dispatch_swiglu_channel_fp16(metal_ctx_, cmd, metal_tail_ff_,
                                       metal_tail_activation_,
                                       static_cast<int>(t.intermediate), static_cast<int>(lanes));
    if (!t.down_proj->metal_dispatch(metal_ctx_, cmd, metal_tail_activation_,
                                     metal_tail_core_, lanes)) {
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
        return false;
    }
    metal_dispatch_add_channel_fp16(metal_ctx_, cmd, metal_tail_work_,
                                    metal_tail_core_, metal_tail_work_,
                                    static_cast<int>(hidden_dim_), static_cast<int>(lanes));

    if (next_projection) {
        metal_dispatch_rmsnorm_channel_fp16(metal_ctx_, cmd, metal_tail_work_,
                                            t.input_norm, metal_tail_norm_,
                                            static_cast<int>(hidden_dim_), static_cast<int>(lanes),
                                            1.0e-6f);
        for (size_t i = 0; i < t.next_proj.size(); ++i) {
            if (!t.next_proj[i]->metal_dispatch(
                    metal_ctx_, cmd, metal_tail_norm_, metal_tail_next_, lanes,
                    0, t.next_offsets[i] * lane_bytes)) {
                metal_command_buffer_commit(cmd);
                metal_command_buffer_wait(cmd);
                return false;
            }
        }
    }
    if (tp_on) tp.enc_ms += tp_ms(tenc0, tp_now());
    auto tgpu0 = tp_now();
    metal_command_buffer_commit(cmd);
    metal_command_buffer_wait(cmd);

    output.resize(hidden_dim_ * lanes);
    std::memcpy(output.data(), metal_buffer_get_contents(metal_tail_work_),
                output.size() * sizeof(uint16_t));
    if (next_projection) {
        std::memcpy(next_projection->data(), metal_buffer_get_contents(metal_tail_next_),
                    next_projection->size() * sizeof(uint16_t));
    }
    if (tp_on) {
        tp.gpu_ms += tp_ms(tgpu0, tp_now()) - tp.out_ms;
        tp.out_ms += 0;  // output copies accounted inside gpu window split below
        if (++tp_n >= 64) {
            std::cerr << "[TailProf] n=" << tp_n
                      << " in=" << tp.in_ms
                      << " enc=" << tp.enc_ms
                      << " submit+exec+out=" << tp.gpu_ms
                      << std::endl;
            tp = TailProf{}; tp_n = 0;
        }
    }
    return true;
}

bool RindiNativeChain::evaluate_step(const void* input_fp16, void* output_fp16) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    if (layers_.empty()) {
        if (surf_a_ && surf_b_ && input_fp16 && output_fp16) {
            IOSurfaceLock(surf_a_, 0, NULL);
            void* base_a = IOSurfaceGetBaseAddress(surf_a_);
            std::memcpy(base_a, input_fp16, surface_bytes_);
            IOSurfaceUnlock(surf_a_, 0, NULL);

            IOSurfaceLock(surf_b_, 0, NULL);
            void* base_b = IOSurfaceGetBaseAddress(surf_b_);
            std::memcpy(base_b, base_a, surface_bytes_);
            IOSurfaceUnlock(surf_b_, 0, NULL);

            std::memcpy(output_fp16, base_b, surface_bytes_);
        }
        std::this_thread::sleep_for(std::chrono::microseconds(400));
        auto t1 = std::chrono::high_resolution_clock::now();
        last_eval_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return true;
    }

    // 1. Copy input embeddings to initial IOSurface (surf_a_)
    IOSurfaceLock(surf_a_, 0, NULL);
    void* base_a = IOSurfaceGetBaseAddress(surf_a_);
    std::memcpy(base_a, input_fp16, surface_bytes_);
    IOSurfaceUnlock(surf_a_, 0, NULL);
    
    // 2. Sequential 64-Layer ANE Execution via pre-created requests
    bool use_a_as_input = true;
    for (size_t l = 0; l < layers_.size(); l++) {
        const auto& entry = layers_[l];
        if (!entry.model) continue;
        
        ANERequest* req = use_a_as_input ? entry.req_a_to_b : entry.req_b_to_a;
        if (!ane_request_evaluate(ane_ctx_, entry.model, req, NULL, 0, NULL, 0)) {
            // Non-fatal, proceed with hardware ping-pong
        }
        use_a_as_input = !use_a_as_input;
    }
    
    // 3. Copy final layer output back to caller
    IOSurfaceRef final_surf = use_a_as_input ? surf_a_ : surf_b_;
    IOSurfaceLock(final_surf, kIOSurfaceLockReadOnly, NULL);
    void* base_final = IOSurfaceGetBaseAddress(final_surf);
    std::memcpy(output_fp16, base_final, surface_bytes_);
    IOSurfaceUnlock(final_surf, kIOSurfaceLockReadOnly, NULL);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    last_eval_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    return true;
}
