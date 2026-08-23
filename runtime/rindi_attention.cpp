// SPDX-License-Identifier: Apache-2.0
#include "rindi_attention.h"
#include "metal_engine.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace rindi_attn;

namespace {

// Qwen3.5's non-traditional MLX RoPE pairs the first 64 channels as two
// 32-wide halves: (0,32), (1,33), ....  The angle base is a fixed geometric
// series, so it is precomputed once instead of calling std::pow per element.
struct RopeTable {
    float inv_freq[32];
    RopeTable() {
        for (int d = 0; d < 32; ++d)
            inv_freq[d] = std::pow(10000000.0f, -static_cast<float>(d) / 32.0f);
    }
};
const RopeTable kRope = RopeTable();

float h2f(uint16_t b) {
    uint32_t s=(uint32_t(b)&0x8000u)<<16, e=(b>>10)&31u, m=b&1023u, v=s;
    if (e == 0) v=s|(m<<13); else if (e == 31) v=s|0x7f800000u|(m<<13);
    else v=s|((e+112u)<<23)|(m<<13);
    float f; std::memcpy(&f,&v,4); return f;
}
uint16_t f2h(float f) {
    uint32_t b; std::memcpy(&b,&f,4); int e=int((b>>23)&255)-127+15;
    uint32_t s=(b>>16)&0x8000u, m=(b>>13)&1023u;
    if (e<=0) return uint16_t(s); if (e>=31) return uint16_t(s|0x7c00u);
    return uint16_t(s|(uint32_t(e)<<10)|m);
}

MetalContext* shared_attention_metal_context() {
    static MetalContext* context = metal_context_create();
    return context;
}

bool metal_attention_disabled() {
    // Deliberately uncached: probes flip this between runs to exercise both
    // paths in one process.
    return std::getenv("RINDI_DISABLE_METAL_ATTENTION") != nullptr;
}

bool metal_attention_compare() {
    return std::getenv("RINDI_COMPARE_METAL_ATTENTION") != nullptr;
}

// Scratch shared by all attention layers: used strictly within one layer's
// command-buffer lifetime, and inference is serialized, so a single set of
// buffers sized to the widest layer/lane count is safe.
struct AttentionWorkspace {
    MetalBufferHandle q{nullptr};      // [32 * kQ] fp16 staged normalized q
    MetalBufferHandle scores{nullptr}; // [32 * kHQ * cap] fp32
    MetalBufferHandle probs{nullptr};  // [32 * kHQ * cap] fp32
    MetalBufferHandle out{nullptr};    // [32 * kQ] fp16
    size_t cap{0};
};
AttentionWorkspace& workspace() {
    static AttentionWorkspace ws;
    return ws;
}

} // namespace

bool RindiAttention::compile(ANEContext* ctx, const SafeTensorsLoader& loader,
                             int layer, size_t context, size_t width) {
    return compile_prefixed(ctx, loader,
                            "layers." + std::to_string(layer) + ".self_attn.",
                            context, width, ProjectionMode::Int4Auto);
}

bool RindiAttention::compile_core(ANEContext* ctx, const SafeTensorsLoader& loader,
                                  int layer, size_t context, size_t width) {
    return compile_prefixed(ctx, loader,
                            "layers." + std::to_string(layer) + ".self_attn.",
                            context, width, ProjectionMode::Int4Auto);
}

bool RindiAttention::compile_prefixed(ANEContext* ctx, const SafeTensorsLoader& loader,
                                      const std::string& p,
                                      size_t context, size_t width,
                                      ProjectionMode mode) {
    if (!ctx || context == 0) return false;
    width_ = std::max<size_t>(32, width); context_ = context;
    proj_mode_ = mode;
    auto load_proj = [&](RindiAneProjection& proj, const std::string& name) {
        if (mode == ProjectionMode::Bf16Metal)
            return proj.compile_bf16_metal(shared_attention_metal_context(),
                                           loader, name);
        if (mode == ProjectionMode::Int4FromBf16)
            return proj.compile_int4_from_bf16(shared_attention_metal_context(),
                                               loader, name);
        return proj.compile_int4_host(loader, name);
    };
    if (mode == ProjectionMode::Int4FromBf16) {
        // Single-dispatch fused q/k/v for the draft path.
        qkv_fused_ready_ = qkv_fused_.compile_int4_from_bf16_fused(
            shared_attention_metal_context(), loader,
            {{p + "q_proj.weight", 24 * 512},
             {p + "k_proj.weight", 4 * 256},
             {p + "v_proj.weight", 4 * 256}});
    }
    if (!load_proj(q_proj_, p+"q_proj.weight") ||
        !load_proj(k_proj_, p+"k_proj.weight") ||
        !load_proj(v_proj_, p+"v_proj.weight") ||
        !loader.get_tensor_fp16(p+"q_norm.weight", q_norm_) ||
        !loader.get_tensor_fp16(p+"k_norm.weight", k_norm_) ||
        q_norm_.size() != 256 || k_norm_.size() != 256) return false;
    keys_.assign(context_ * kKV, 0);
    values_.assign(context_ * kKV, 0);
    position_ = 0;
    if (!load_proj(o_proj_, p+"o_proj.weight")) return false;
    ready_ = true; return true;
}

bool RindiAttention::project(const uint16_t* hidden, size_t lanes,
                             std::vector<uint16_t>& q,
                             std::vector<uint16_t>& k,
                             std::vector<uint16_t>& v) {
    if (!ready_ || !hidden || lanes == 0 || lanes > width_) return false;
    if (qkv_fused_ready_ && lanes <= width_) {
        std::vector<uint16_t> fused;
        if (!qkv_fused_.evaluate(hidden, lanes, fused)) return false;
        const size_t QG = 24 * 512, K = 4 * 256;
        if (fused.size() != (QG + 2 * K) * lanes) return false;
        q.assign(fused.begin(), fused.begin() + QG * lanes);
        k.assign(fused.begin() + QG * lanes, fused.begin() + (QG + K) * lanes);
        v.assign(fused.begin() + (QG + K) * lanes, fused.end());
        return true;
    }
    return q_proj_.evaluate(hidden, lanes, q) &&
           k_proj_.evaluate(hidden, lanes, k) &&
           v_proj_.evaluate(hidden, lanes, v);
}

void RindiAttention::reset_metal_state() {
    if (!metal_ctx_) return;
    if (metal_k_cache_) {
        std::memset(metal_buffer_get_contents(metal_k_cache_), 0,
                    context_ * kKV * sizeof(uint16_t));
    }
    if (metal_v_cache_) {
        std::memset(metal_buffer_get_contents(metal_v_cache_), 0,
                    context_ * kKV * sizeof(uint16_t));
    }
}

void RindiAttention::reset() {
    std::fill(keys_.begin(), keys_.end(), 0);
    std::fill(values_.begin(), values_.end(), 0);
    position_ = 0;
    reset_metal_state();
}

bool RindiAttention::ensure_metal_resources() {
    if (metal_attention_disabled()) return false;
    if (!metal_ctx_) {
        metal_ctx_ = shared_attention_metal_context();
        if (!metal_ctx_) return false;
    }
    if (!metal_k_cache_) {
        metal_k_cache_ = metal_buffer_create(metal_ctx_, context_ * kKV * sizeof(uint16_t));
        metal_v_cache_ = metal_buffer_create(metal_ctx_, context_ * kKV * sizeof(uint16_t));
        if (!metal_k_cache_ || !metal_v_cache_) return false;
        std::memset(metal_buffer_get_contents(metal_k_cache_), 0,
                    context_ * kKV * sizeof(uint16_t));
        std::memset(metal_buffer_get_contents(metal_v_cache_), 0,
                    context_ * kKV * sizeof(uint16_t));
    }
    auto& ws = workspace();
    if (ws.cap < context_) {
        // Grow to the largest configured context across layers.
        auto grow = [&](MetalBufferHandle& buf, size_t bytes) {
            if (buf) metal_buffer_release(buf);
            buf = metal_buffer_create(metal_ctx_, bytes);
            return buf != nullptr;
        };
        // Workspace columns follow the compiled program width (32 shipped,
        // wider under RINDI_ANE_WIDTH), not a literal.
        const size_t ws_lanes = width_;
        if (!grow(ws.q, ws_lanes * kQ * sizeof(uint16_t))) return false;
        if (!grow(ws.scores, ws_lanes * kHQ * context_ * sizeof(float))) return false;
        if (!grow(ws.probs, ws_lanes * kHQ * context_ * sizeof(float))) return false;
        if (!grow(ws.out, ws_lanes * kQ * sizeof(uint16_t))) return false;
        ws.cap = context_;
    }
    return true;
}

// RMSNorm over head_dim, RoPE on the first 64 channels as two 32-wide halves,
// value passthrough. Pure math on prepared projection outputs; no cache state.
void RindiAttention::prepare_qkv(const uint16_t* qraw, const uint16_t* kraw,
                                 const uint16_t* vraw, size_t rope_pos,
                                 float* q, float* k, float* v) {
    for (size_t h = 0; h < kHQ; ++h) {
        float ss = 0;
        for (size_t d = 0; d < kD; ++d) {
            const float x = h2f(qraw[h * (2 * kD) + d]);
            ss += x * x;
        }
        const float den = std::sqrt(ss / kD + 1e-6f);
        for (size_t d = 0; d < kD; ++d)
            q[h * kD + d] = h2f(qraw[h * (2 * kD) + d]) / den * h2f(q_norm_[d]);
    }
    for (size_t h = 0; h < kHK; ++h) {
        float ss = 0;
        for (size_t d = 0; d < kD; ++d) { float x = h2f(kraw[h*kD+d]); ss += x*x; }
        const float den = std::sqrt(ss / kD + 1e-6f);
        for (size_t d = 0; d < kD; ++d) {
            k[h*kD+d] = h2f(kraw[h*kD+d]) / den * h2f(k_norm_[d]);
            v[h*kD+d] = h2f(vraw[h*kD+d]);
        }
    }
    for (size_t h = 0; h < kHQ; ++h) {
        for (size_t d = 0; d < 32; ++d) {
            const float angle = static_cast<float>(rope_pos) * kRope.inv_freq[d];
            const float co = std::cos(angle), si = std::sin(angle);
            const float a = q[h * kD + d], b = q[h * kD + 32 + d];
            q[h * kD + d]       = a * co - b * si;
            q[h * kD + 32 + d]  = a * si + b * co;
        }
    }
    for (size_t h = 0; h < kHK; ++h) {
        for (size_t d = 0; d < 32; ++d) {
            const float angle = static_cast<float>(rope_pos) * kRope.inv_freq[d];
            const float co = std::cos(angle), si = std::sin(angle);
            const float a = k[h * kD + d], b = k[h * kD + 32 + d];
            k[h * kD + d]       = a * co - b * si;
            k[h * kD + 32 + d]  = a * si + b * co;
        }
    }
}

// Reference attention math reading the host-side KV history (which must
// already include the current rows). Used as the CPU path and as the oracle
// for RINDI_COMPARE_METAL_ATTENTION.
void RindiAttention::attend_reference(const float* q, size_t lane_index,
                                      size_t valid, size_t lanes,
                                      std::vector<uint16_t>& attended) const {
    const float inv = 1.0f / std::sqrt(static_cast<float>(kD));
    std::vector<float> score(valid);
    for (size_t h = 0; h < kHQ; ++h) {
        const size_t kh = h / (kHQ / kHK);
        float mx = -1e30f;
        for (size_t t = 0; t < valid; ++t) {
            float s = 0;
            const uint16_t* kr = keys_.data() + t * kKV + kh * kD;
            for (size_t d = 0; d < kD; ++d) s += q[h * kD + d] * h2f(kr[d]);
            score[t] = s * inv;
            if (score[t] > mx) mx = score[t];
        }
        float den = 0;
        for (size_t t = 0; t < valid; ++t) { score[t] = std::exp(score[t] - mx); den += score[t]; }
        const float inv_den = 1.0f / den;
        for (size_t d = 0; d < kD; ++d) {
            float y = 0;
            for (size_t t = 0; t < valid; ++t) {
                y += score[t] * inv_den * h2f(values_[t * kKV + kh * kD + d]);
            }
            attended[(h * kD + d) * lanes + lane_index] = f2h(y);
        }
    }
}

// Unified core: per lane, normalize+RoPE -> append KV (host + Metal mirror) ->
// stage q; then one Metal dispatch chain or the scalar reference path.
bool RindiAttention::core_batch_impl(const uint16_t* qraw, const uint16_t* kraw,
                                     const uint16_t* vraw, size_t lanes,
                                     std::vector<uint16_t>& attended) {
    constexpr size_t QRAW = kHQ * 2 * kD;
    if (!ready_ || position_ >= context_ || lanes == 0 || lanes > width_ ||
        position_ + lanes > context_) return false;

    q_f_.resize(kQ); k_f_.resize(kKV); v_f_.resize(kKV);
    float* q = q_f_.data(); float* k = k_f_.data(); float* v = v_f_.data();

    const bool use_metal = ensure_metal_resources();
    auto& ws = workspace();

    if (use_metal) {
        auto* q_stage = static_cast<uint16_t*>(metal_buffer_get_contents(ws.q));
        auto* kc = static_cast<uint16_t*>(metal_buffer_get_contents(metal_k_cache_));
        auto* vc = static_cast<uint16_t*>(metal_buffer_get_contents(metal_v_cache_));
        for (size_t lane = 0; lane < lanes; ++lane) {
            // De-interleave this lane from the channel-major projections.
            q_lane_.resize(QRAW); k_lane_.resize(kKV); v_lane_.resize(kKV);
            for (size_t c = 0; c < QRAW; ++c) q_lane_[c] = qraw[c * lanes + lane];
            for (size_t c = 0; c < kKV; ++c) k_lane_[c] = kraw[c * lanes + lane];
            for (size_t c = 0; c < kKV; ++c) v_lane_[c] = vraw[c * lanes + lane];
            prepare_qkv(q_lane_.data(), k_lane_.data(), v_lane_.data(),
                        position_ + lane, q, k, v);
            const size_t row = position_ + lane;
            for (size_t i = 0; i < kKV; ++i) {
                keys_[row * kKV + i] = f2h(k[i]);
                values_[row * kKV + i] = f2h(v[i]);
            }
            std::memcpy(kc + row * kKV, keys_.data() + row * kKV, kKV * sizeof(uint16_t));
            std::memcpy(vc + row * kKV, values_.data() + row * kKV, kKV * sizeof(uint16_t));
            for (size_t i = 0; i < kQ; ++i)
                q_stage[(lane * kQ) + i] = f2h(q[i]);
        }

        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
        if (!cmd) return false;
        metal_dispatch_attn_scores_fp16(metal_ctx_, cmd, ws.q, metal_k_cache_,
                                        ws.scores, static_cast<uint32_t>(position_),
                                        static_cast<uint32_t>(position_ + lanes),
                                        static_cast<uint32_t>(lanes),
                                        static_cast<uint32_t>(ws.cap),
                                        kHQ, kHK, kD);
        metal_dispatch_attn_softmax_fp16(metal_ctx_, cmd, ws.scores, ws.probs,
                                         static_cast<uint32_t>(lanes * kHQ),
                                         static_cast<uint32_t>(position_),
                                         static_cast<uint32_t>(ws.cap), kHQ);
        metal_dispatch_attn_pv_fp16(metal_ctx_, cmd, ws.probs, metal_v_cache_,
                                    ws.out, static_cast<uint32_t>(position_),
                                    static_cast<uint32_t>(lanes),
                                    static_cast<uint32_t>(ws.cap),
                                    kHQ, kHK, kD);
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);

        attended.assign(kQ * lanes, 0);
        const uint16_t* out = static_cast<const uint16_t*>(
            metal_buffer_get_contents(ws.out));

        if (metal_attention_compare()) {
            // Oracle: same prepared q/k/v through the scalar reference path.
            double max_abs = 0.0, sum_abs = 0.0;
            size_t count = 0;
            for (size_t lane = 0; lane < lanes; ++lane) {
                // Rewind the staging inputs for this lane's reference pass.
                for (size_t c = 0; c < QRAW; ++c) q_lane_[c] = qraw[c * lanes + lane];
                for (size_t c = 0; c < kKV; ++c) k_lane_[c] = kraw[c * lanes + lane];
                for (size_t c = 0; c < kKV; ++c) v_lane_[c] = vraw[c * lanes + lane];
                prepare_qkv(q_lane_.data(), k_lane_.data(), v_lane_.data(),
                            position_ + lane, q, k, v);
                std::vector<uint16_t> ref(kQ * lanes);
                attend_reference(q, lane, position_ + lane + 1, lanes, ref);
                for (size_t c = 0; c < kQ; ++c) {
                    // out is lane-major [lane][Q]; ref is channel-major.
                    const size_t idx = c * lanes + lane;
                    const double diff = std::abs(static_cast<double>(h2f(out[lane * kQ + c])) -
                                                 static_cast<double>(h2f(ref[idx])));
                    max_abs = std::max(max_abs, diff);
                    sum_abs += diff;
                    ++count;
                }
            }
            std::fprintf(stderr,
                         "[RindiMetalAttention] lanes=%zu pos=%zu max_abs=%.6f mean_abs=%.6f\\n",
                         lanes, position_,
                         max_abs, count ? sum_abs / count : 0.0);
        }

        // Lane-major [lane][head][dim] -> channel-major [c][lane].
        for (size_t lane = 0; lane < lanes; ++lane) {
            const uint16_t* src = out + lane * kQ;
            for (size_t c = 0; c < kQ; ++c) attended[c * lanes + lane] = src[c];
        }
        position_ += lanes;
        return true;
    }

    // Scalar CPU fallback (also the correctness oracle).
    attended.assign(kQ * lanes, 0);
    q_lane_.resize(QRAW); k_lane_.resize(kKV); v_lane_.resize(kKV);
    for (size_t lane = 0; lane < lanes; ++lane) {
        for (size_t c = 0; c < QRAW; ++c) q_lane_[c] = qraw[c * lanes + lane];
        for (size_t c = 0; c < kKV; ++c) k_lane_[c] = kraw[c * lanes + lane];
        for (size_t c = 0; c < kKV; ++c) v_lane_[c] = vraw[c * lanes + lane];
        prepare_qkv(q_lane_.data(), k_lane_.data(), v_lane_.data(),
                    position_ + lane, q, k, v);
        const size_t row = position_ + lane;
        for (size_t i = 0; i < kKV; ++i) {
            keys_[row * kKV + i] = f2h(k[i]);
            values_[row * kKV + i] = f2h(v[i]);
        }
        attend_reference(q, lane, row + 1, lanes, attended);
    }
    position_ += lanes;
    return true;
}

bool RindiAttention::step(const uint16_t* hidden, size_t lanes,
                          std::vector<uint16_t>& output) {
    if (!ready_ || !hidden || lanes != 1 || position_ >= context_) return false;
    std::vector<uint16_t> qraw, kraw, vraw;
    if (!project(hidden, 1, qraw, kraw, vraw)) return false;
    std::vector<uint16_t> attended;
    if (!core_step(qraw.data(), qraw.size(), kraw.data(), kraw.size(),
                   vraw.data(), vraw.size(), attended)) return false;
    for (size_t h = 0; h < kHQ; ++h) {
        for (size_t d = 0; d < kD; ++d) {
            const size_t c = h * kD + d;
            const float gate = h2f(qraw[h * 512 + 256 + d]);
            attended[c] = f2h(h2f(attended[c]) /
                              (1.0f + std::exp(-gate)));
        }
    }
    if (!o_proj_.evaluate(attended.data(), 1, output)) return false;
    for(size_t c=0;c<5120;++c){float y=h2f(output[c])+h2f(hidden[c]);output[c]=f2h(y);}
    return true;
}

bool RindiAttention::core_step(const std::vector<uint16_t>& qraw,
                               const std::vector<uint16_t>& kraw,
                               const std::vector<uint16_t>& vraw,
                               std::vector<uint16_t>& attended) {
    return core_step(qraw.data(), qraw.size(), kraw.data(), kraw.size(),
                     vraw.data(), vraw.size(), attended);
}

bool RindiAttention::core_step(const uint16_t* qraw, size_t qraw_len,
                               const uint16_t* kraw, size_t kraw_len,
                               const uint16_t* vraw, size_t vraw_len,
                               std::vector<uint16_t>& attended) {
    if (!ready_ || qraw_len < kHQ * 2 * kD ||
        kraw_len < kKV || vraw_len < kKV) return false;
    // The pointer API carries contiguous single-lane slices.
    std::vector<uint16_t> q(qraw, qraw + kHQ * 2 * kD);
    std::vector<uint16_t> k(kraw, kraw + kKV);
    std::vector<uint16_t> v(vraw, vraw + kKV);
    return core_batch_impl(q.data(), k.data(), v.data(), 1, attended);
}

bool RindiAttention::core_step_batch(const std::vector<uint16_t>& qraw,
                                     const std::vector<uint16_t>& kraw,
                                     const std::vector<uint16_t>& vraw,
                                     size_t lanes,
                                     std::vector<uint16_t>& attended) {
    if (!ready_ || lanes == 0 || lanes > width_ ||
        qraw.size() != (kHQ * 2 * kD) * lanes ||
        kraw.size() != kKV * lanes || vraw.size() != kKV * lanes) return false;
    return core_batch_impl(qraw.data(), kraw.data(), vraw.data(), lanes, attended);
}

bool RindiAttention::core_step_batch(const uint16_t* qraw, const uint16_t* kraw,
                                     const uint16_t* vraw, size_t lanes,
                                     std::vector<uint16_t>& attended) {
    if (!ready_ || lanes == 0 || lanes > width_) return false;
    return core_batch_impl(qraw, kraw, vraw, lanes, attended);
}
