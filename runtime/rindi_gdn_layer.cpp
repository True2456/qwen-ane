// SPDX-License-Identifier: Apache-2.0
#include "rindi_gdn_layer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace {
float half_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t exp = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x3ffu;
    uint32_t value = sign;
    if (exp == 0) value = sign | (mant << 13);
    else if (exp == 31) value = sign | 0x7f800000u | (mant << 13);
    else value = sign | ((exp + 112u) << 23) | (mant << 13);
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

float stable_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}
}

bool RindiGdnLayer::compile(ANEContext* ctx, const SafeTensorsLoader& loader,
                            int layer, size_t width) {
    if (!ctx || layer < 0) return false;
    const std::string prefix = "layers." + std::to_string(layer) + ".linear_attn.";
    const std::array<std::string, 4> names = {
        prefix + "in_proj_qkv.weight",
        prefix + "in_proj_z.weight",
        prefix + "in_proj_b.weight",
        prefix + "in_proj_a.weight",
    };
    if (!compile_core(ctx, loader, layer, width)) return false;
    const std::string post_norm_name = "layers." + std::to_string(layer) + ".post_attention_layernorm.weight";
    const std::string out_name = prefix + "out_proj.weight";
    const std::string gate_name = "layers." + std::to_string(layer) + ".mlp.gate_proj.weight";
    const std::string up_name = "layers." + std::to_string(layer) + ".mlp.up_proj.weight";
    const std::string down_name = "layers." + std::to_string(layer) + ".mlp.down_proj.weight";
    if (!loader.get_tensor_fp16(post_norm_name, post_norm_) || post_norm_.size() != 5120 ||
        !out_proj_.compile_int4(ctx, loader, out_name, width_, "gdn_out") ||
        !mlp_gate_.compile_int4(ctx, loader, gate_name, width_, "gdn_gate") ||
        !mlp_up_.compile_int4(ctx, loader, up_name, width_, "gdn_up") ||
        !mlp_down_.compile_int4(ctx, loader, down_name, width_, "gdn_down")) return false;
    ready_ = true;
    return true;
}

bool RindiGdnLayer::compile_core(ANEContext* ctx, const SafeTensorsLoader& loader,
                                  int layer, size_t width) {
    if (!ctx || layer < 0) return false;
    const std::string prefix = "layers." + std::to_string(layer) + ".linear_attn.";
    const std::array<std::string, 4> names = {
        prefix + "in_proj_qkv.weight",
        prefix + "in_proj_z.weight",
        prefix + "in_proj_b.weight",
        prefix + "in_proj_a.weight",
    };
    width_ = std::max<size_t>(32, width);
    for (size_t i = 0; i < names.size(); ++i) {
        if (!loader.has_tensor(names[i]) ||
            !projections_[i].compile_int4_host(loader, names[i])) {
            return false;
        }
    }
    const std::string conv_name = prefix + "conv1d.weight";
    const std::string alog_name = prefix + "A_log";
    const std::string dt_name = prefix + "dt_bias";
    const std::string gdn_norm_name = prefix + "norm.weight";
    if (!conv_.compile(ctx, loader, conv_name, width_)) {
        std::cerr << "[RindiGDN] conv compile failed layer=" << layer
                  << " width=" << width_ << std::endl;
        return false;
    }
    // Recurrence graph only compiles at its proven 160-column width; that
    // covers every prefill chunk width we use (<= 157 live lanes).
    if (!recurrence_.compile(ctx, 48, 128, 128, 160)) {
        std::cerr << "[RindiGDN] recurrence compile failed layer=" << layer << std::endl;
        return false;
    }
    if (!loader.get_tensor_fp16(alog_name, a_log_) ||
        !loader.get_tensor_fp16(dt_name, dt_bias_) ||
        !loader.get_tensor_fp16(gdn_norm_name, gdn_norm_) ||
        a_log_.size() != 48 || dt_bias_.size() != 48 ||
        gdn_norm_.size() != 128) return false;
    ready_ = true;
    return true;
}

void RindiGdnLayer::reset() {
    conv_.reset();
    recurrence_.reset();
}

bool RindiGdnLayer::gate_core(const std::vector<uint16_t>& core,
                              const std::vector<uint16_t>& z,
                              std::vector<uint16_t>& gated) const {
    constexpr size_t H = 48, D = 128, C = H * D;
    if (core.size() != C || z.size() != C || gdn_norm_.size() != D) return false;
    gated.resize(C);
    for (size_t h = 0; h < H; ++h) {
        float sum = 0.0f;
        for (size_t d = 0; d < D; ++d) {
            const float value = half_to_float(core[h * D + d]);
            sum += value * value;
        }
        const float inv = 1.0f / std::sqrt(sum / D + 1.0e-6f);
        for (size_t d = 0; d < D; ++d) {
            const float value = half_to_float(core[h * D + d]) * inv * half_to_float(gdn_norm_[d]);
            const float gate = half_to_float(z[h * D + d]);
            // Qwen3.5 uses Qwen3NextRMSNormGated's precise SiLU gate,
            // i.e. silu(z) * rms_norm(x), rather than sigmoid(z) alone.
            gated[h * D + d] = float_to_half(value * gate /
                                              (1.0f + std::exp(-gate)));
        }
    }
    return true;
}

bool RindiGdnLayer::gate_core_batch(const std::vector<uint16_t>& core,
                                    const std::vector<uint16_t>& z,
                                    size_t lanes,
                                    std::vector<uint16_t>& gated) const {
    constexpr size_t H = 48, D = 128, C = H * D;
    if (lanes == 0 || core.size() != C * lanes || z.size() != C * lanes ||
        gdn_norm_.size() != D) return false;
    gated.resize(C * lanes);
    for (size_t h = 0; h < H; ++h) {
        for (size_t lane = 0; lane < lanes; ++lane) {
            float sum = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                const float value = half_to_float(core[(h * D + d) * lanes + lane]);
                sum += value * value;
            }
            const float inv = 1.0f / std::sqrt(sum / D + 1.0e-6f);
            for (size_t d = 0; d < D; ++d) {
                const size_t c = h * D + d;
                const float value = half_to_float(core[c * lanes + lane]) * inv *
                                     half_to_float(gdn_norm_[d]);
                const float gate = half_to_float(z[c * lanes + lane]);
                gated[c * lanes + lane] = float_to_half(
                    value * gate / (1.0f + std::exp(-gate)));
            }
        }
    }
    return true;
}

bool RindiGdnLayer::project(const uint16_t* hidden, size_t lanes,
                            RindiGdnProjectionOutput& output) {
    if (!ready_ || !hidden) return false;
    return projections_[0].evaluate(hidden, lanes, output.qkv) &&
           projections_[1].evaluate(hidden, lanes, output.z) &&
           projections_[2].evaluate(hidden, lanes, output.beta) &&
           projections_[3].evaluate(hidden, lanes, output.a);
}

bool RindiGdnLayer::step(const uint16_t* hidden, size_t lanes,
                         std::vector<uint16_t>& output) {
    if (!ready_ || !hidden || lanes == 0 || lanes > width_) return false;
    std::vector<uint16_t> core, z;
    if (!core_step(hidden, lanes, core, z)) return false;

    constexpr size_t H = 5120, CORE = 6144, I = 17408;
    std::vector<uint16_t> gated(CORE * lanes);
    for (size_t lane = 0; lane < lanes; ++lane) {
        for (size_t h = 0; h < 48; ++h) {
            float sum = 0.0f;
            for (size_t d = 0; d < 128; ++d) {
                const float v = half_to_float(core[(h * 128 + d) * lanes + lane]);
                sum += v * v;
            }
            const float den = std::sqrt(sum / 128.0f + 1.0e-6f);
            for (size_t d = 0; d < 128; ++d) {
                const size_t c = h * 128 + d;
                const float cv = half_to_float(core[c * lanes + lane]) / den;
                const float zv = half_to_float(z[c * lanes + lane]);
                const float silu = zv / (1.0f + std::exp(-zv));
                gated[c * lanes + lane] = float_to_half(cv * half_to_float(gdn_norm_[d]) * silu);
            }
        }
    }

    std::vector<uint16_t> attention;
    if (!out_proj_.evaluate(gated.data(), lanes, attention)) return false;
    std::vector<uint16_t> normalized(H * lanes);
    for (size_t lane = 0; lane < lanes; ++lane) {
        float sum = 0.0f;
        for (size_t c = 0; c < H; ++c) {
            const float v = half_to_float(attention[c * lanes + lane]) +
                            half_to_float(hidden[c * lanes + lane]);
            sum += v * v;
        }
        const float den = std::sqrt(sum / H + 1.0e-6f);
        for (size_t c = 0; c < H; ++c) {
            const float v = (half_to_float(attention[c * lanes + lane]) +
                             half_to_float(hidden[c * lanes + lane])) / den;
            normalized[c * lanes + lane] = float_to_half(v * half_to_float(post_norm_[c]));
        }
    }

    std::vector<uint16_t> gate, up;
    if (!mlp_gate_.evaluate(normalized.data(), lanes, gate) ||
        !mlp_up_.evaluate(normalized.data(), lanes, up)) return false;
    std::vector<uint16_t> activation(I * lanes);
    for (size_t i = 0; i < I * lanes; ++i) {
        const float g = half_to_float(gate[i]);
        const float u = half_to_float(up[i]);
        activation[i] = float_to_half((g / (1.0f + std::exp(-g))) * u);
    }
    std::vector<uint16_t> mlp;
    if (!mlp_down_.evaluate(activation.data(), lanes, mlp)) return false;
    output = std::move(mlp);
    return output.size() == H * lanes;
}

bool RindiGdnLayer::core_step(const uint16_t* hidden, size_t lanes,
                              std::vector<uint16_t>& core,
                              std::vector<uint16_t>& z) {
    if (!ready_ || !hidden || lanes == 0 || lanes > width_) return false;
    RindiGdnProjectionOutput p;
    if (!project(hidden, lanes, p)) return false;
    return core_from_projected(p, lanes, core, z);
}

bool RindiGdnLayer::core_from_projected(const RindiGdnProjectionOutput& p,
                                        size_t lanes,
                                        std::vector<uint16_t>& core,
                                        std::vector<uint16_t>& z) {
    RindiGdnProjectionView view{p.qkv.data(), p.z.data(), p.beta.data(), p.a.data()};
    return core_from_projected_view(view, lanes, core, z);
}

bool RindiGdnLayer::core_from_projected_view(const RindiGdnProjectionView& p,
                                             size_t lanes,
                                             std::vector<uint16_t>& core,
                                             std::vector<uint16_t>& z) {
    if (!ready_ || lanes == 0 || lanes > width_) return false;
    constexpr size_t H = 48, D = 128, V = 128, HK = H * D, W = 160;
    constexpr size_t C = HK + 2 * H;
    if (!conv_.evaluate(p.qkv, lanes, activated_scratch_)) return false;
    const std::vector<uint16_t>& activated = activated_scratch_;
    // z carries in_proj_z's HK channels per lane; the packed-surface count
    // C includes the value/beta channel groups and must not size it.
    z.assign(p.z, p.z + HK * lanes);

    // The packed [C, W] surface is only consumed by the legacy per-lane ANE
    // fallback below; building it for the batch path was pure waste.
    decay_batch_.resize(H * lanes);
    key_batch_.resize(HK * lanes);
    query_batch_.resize(HK * lanes);
    value_batch_.resize(H * V * lanes);
    beta_batch_.resize(H * lanes);
    uint16_t* decay_batch = decay_batch_.data();
    uint16_t* key_batch = key_batch_.data();
    uint16_t* query_batch = query_batch_.data();
    uint16_t* value_batch = value_batch_.data();
    uint16_t* beta_batch = beta_batch_.data();
    for (size_t lane = 0; lane < lanes; ++lane) {
        for (size_t h = 0; h < H; ++h) {
            const size_t source_h = h / 3;
            float qsq = 0.0f, ksq = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                // The recurrence now consumes already-normalized q/k.  The
                // old ANE graph normalized temporary values after a *16
                // expansion; retaining that expansion here would make the
                // final q/k 16x too small.
                const float q = half_to_float(activated[(source_h * D + d) * lanes + lane]);
                const float k = half_to_float(activated[(2048 + source_h * D + d) * lanes + lane]);
                qsq += q * q; ksq += k * k;
            }
            const float qden = std::sqrt(qsq / D + 1.0e-6f);
            const float kden = std::sqrt(ksq / D + 1.0e-6f);
            const float raw_a = half_to_float(p.a[h * lanes + lane]);
            const float raw_b = half_to_float(p.beta[h * lanes + lane]);
            const float decay = std::exp(-std::exp(half_to_float(a_log_[h])) *
                                         stable_softplus(raw_a + half_to_float(dt_bias_[h])));
            const float beta = 1.0f / (1.0f + std::exp(-raw_b));
            decay_batch[h * lanes + lane] = float_to_half(decay);
            beta_batch[h * lanes + lane] = float_to_half(beta);
            for (size_t d = 0; d < D; ++d) {
                const size_t c = h * D + d;
                // Match mlx_lm's Qwen3.5 gated-delta normalization exactly:
                // q = rms_norm(q) * inv_scale^2 and k = rms_norm(k) *
                // inv_scale, with inv_scale = 1/sqrt(128).  The previous
                // 0.5/0.0889 factors were not equivalent and destabilized
                // every linear-attention block.
                const float q = half_to_float(activated[(source_h * D + d) * lanes + lane]) /
                                qden / static_cast<float>(D);
                const float k = half_to_float(activated[(2048 + source_h * D + d) * lanes + lane]) /
                                kden / std::sqrt(static_cast<float>(D));
                key_batch[c * lanes + lane] = float_to_half(k);
                query_batch[c * lanes + lane] = float_to_half(q);
            }
            for (size_t d = 0; d < V; ++d) {
                value_batch[(h * V + d) * lanes + lane] =
                    activated[(4096 + h * V + d) * lanes + lane];
            }
        }
    }

    // Always take the Metal batched recurrence, lanes == 1 included: base
    // decoding and speculative verification must execute the IDENTICAL
    // recurrence implementation, otherwise fp16 rounding differences change
    // logits and speculative output stops being exact. The legacy ANE
    // single-step path remains reachable only via RINDI_DISABLE_METAL_RECURRENCE.
    if (!std::getenv("RINDI_DISABLE_METAL_RECURRENCE")) {
        if (recurrence_.step_batch(decay_batch, key_batch,
                                   query_batch, value_batch,
                                   beta_batch, lanes, core) &&
            core.size() == H * V * lanes) {
            return true;
        }
    }

    // CPU/ANE fallback for single-token decode or if the Metal recurrence is
    // unavailable. Rebuild the compact packed input for each lane because the
    // legacy recurrence request owns one sequential state transition.
    packed_scratch_.assign(C * W, 0);
    uint16_t* packed = packed_scratch_.data();
    core.resize(H * V * lanes);
    for (size_t lane = 0; lane < lanes; ++lane) {
        std::fill(packed, packed + C * W, 0);
        for (size_t h = 0; h < H; ++h) {
            for (size_t d = 0; d < D; ++d) {
                const size_t c = h * D + d;
                packed[c * W + V] = decay_batch[h * lanes + lane];
                packed[c * W + V + 1] = key_batch[c * lanes + lane];
                packed[c * W + V + 2] = query_batch[c * lanes + lane];
            }
            for (size_t d = 0; d < V; ++d)
                packed[(HK + h) * W + d] = value_batch[(h * V + d) * lanes + lane];
            packed[(HK + H + h) * W] = beta_batch[h * lanes + lane];
        }
        std::vector<uint16_t> lane_core;
        if (!recurrence_.step(packed, lane_core) || lane_core.size() != H * V)
            return false;
        for (size_t c = 0; c < H * V; ++c) core[c * lanes + lane] = lane_core[c];
    }
    return true;
}
