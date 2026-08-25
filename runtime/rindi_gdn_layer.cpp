// SPDX-License-Identifier: Apache-2.0
#include "rindi_gdn_layer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {
struct GdnGateMetal {
    MetalContext* ctx{nullptr};
    MetalBufferHandle core{nullptr};
    MetalBufferHandle z{nullptr};
    MetalBufferHandle weight{nullptr};
    MetalBufferHandle output{nullptr};
    size_t capacity{0};
};

GdnGateMetal& shared_gdn_gate_metal() {
    static GdnGateMetal state;
    return state;
}

bool gate_core_metal(const std::vector<uint16_t>& core,
                     const std::vector<uint16_t>& z,
                     const std::vector<uint16_t>& weight,
                     size_t lanes, std::vector<uint16_t>& output) {
    constexpr size_t HK = 48 * 128;
    if (core.size() != HK * lanes || z.size() != core.size() ||
        weight.size() != 128) return false;
    GdnGateMetal& m = shared_gdn_gate_metal();
    if (!m.ctx) m.ctx = metal_context_create();
    if (!m.ctx) return false;
    if (m.capacity < core.size()) {
        if (m.core) metal_buffer_release(m.core);
        if (m.z) metal_buffer_release(m.z);
        if (m.output) metal_buffer_release(m.output);
        m.core = metal_buffer_create(m.ctx, core.size() * sizeof(uint16_t));
        m.z = metal_buffer_create(m.ctx, core.size() * sizeof(uint16_t));
        m.output = metal_buffer_create(m.ctx, core.size() * sizeof(uint16_t));
        m.capacity = (m.core && m.z && m.output) ? core.size() : 0;
    }
    if (!m.weight)
        m.weight = metal_buffer_create(m.ctx, weight.size() * sizeof(uint16_t));
    if (!m.capacity || !m.weight) return false;
    std::memcpy(metal_buffer_get_contents(m.core), core.data(),
                core.size() * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(m.z), z.data(),
                z.size() * sizeof(uint16_t));
    std::memcpy(metal_buffer_get_contents(m.weight), weight.data(),
                weight.size() * sizeof(uint16_t));
    MetalCommandBufferHandle cmd = metal_command_buffer_create(m.ctx);
    if (!cmd) return false;
    metal_dispatch_gdn_gate_core(m.ctx, cmd, m.core, m.z, m.weight, m.output,
                                 static_cast<int>(lanes));
    metal_command_buffer_commit(cmd);
    metal_command_buffer_wait(cmd);
    output.resize(core.size());
    std::memcpy(output.data(), metal_buffer_get_contents(m.output),
                output.size() * sizeof(uint16_t));
    return true;
}

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
    if (layer < 0) return false;
    if (!ctx && !std::getenv("RINDI_TAIL_COREAI")) return false;
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
    if (layer < 0) return false;
    if (!ctx && !std::getenv("RINDI_TAIL_COREAI")) return false;
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
    if (lanes > 1 && (std::getenv("RINDI_GDN_METAL_GATE") ||
                      std::getenv("RINDI_QWEN_PREFILL_FAST")) &&
        gate_core_metal(core, z, gdn_norm_, lanes, gated)) return true;
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
    if (!ready_ || !hidden || lanes == 0) return false;
    const size_t projection_width = projections_[0].width();
    if (lanes <= projection_width) {
        return projections_[0].evaluate(hidden, lanes, output.qkv) &&
               projections_[1].evaluate(hidden, lanes, output.z) &&
               projections_[2].evaluate(hidden, lanes, output.beta) &&
               projections_[3].evaluate(hidden, lanes, output.a);
    }

    // The first transformer layer has no folded previous-tail projection.
    // Its groupwise Metal projections retain the proven <=32-lane exact path,
    // so split a wide prefill chunk here and stitch channel-major results.
    // Later layers consume the projection folded into their CoreAI tail.
    std::array<std::vector<uint16_t>*, 4> destinations = {
        &output.qkv, &output.z, &output.beta, &output.a
    };
    for (size_t i = 0; i < projections_.size(); ++i)
        destinations[i]->assign(projections_[i].output_dim() * lanes, 0);

    std::vector<uint16_t> chunk_input;
    std::vector<uint16_t> chunk_output;
    for (size_t offset = 0; offset < lanes; offset += projection_width) {
        const size_t chunk_lanes = std::min(projection_width, lanes - offset);
        chunk_input.resize(5120 * chunk_lanes);
        for (size_t c = 0; c < 5120; ++c)
            std::memcpy(chunk_input.data() + c * chunk_lanes,
                        hidden + c * lanes + offset,
                        chunk_lanes * sizeof(uint16_t));
        for (size_t i = 0; i < projections_.size(); ++i) {
            if (!projections_[i].evaluate(chunk_input.data(), chunk_lanes,
                                          chunk_output)) return false;
            const size_t rows = projections_[i].output_dim();
            if (chunk_output.size() != rows * chunk_lanes) return false;
            for (size_t r = 0; r < rows; ++r)
                std::memcpy(destinations[i]->data() + r * lanes + offset,
                            chunk_output.data() + r * chunk_lanes,
                            chunk_lanes * sizeof(uint16_t));
        }
    }
    return true;
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
    if (!ready_ || lanes == 0 || lanes > width_) {
        std::fprintf(stderr, "[GdnCore] reject: ready=%d lanes=%zu width=%zu\n",
                     ready_, lanes, width_);
        return false;
    }
    constexpr size_t H = 48, D = 128, V = 128, HK = H * D, W = 160;
    constexpr size_t C = HK + 2 * H;
    const bool profile = std::getenv("RINDI_GDN_PROFILE") != nullptr;
    const auto profile_start = std::chrono::high_resolution_clock::now();
    if (!conv_.evaluate(p.qkv, lanes, activated_scratch_)) {
        std::fprintf(stderr, "[GdnCore] conv evaluate failed\n");
        return false;
    }
    const auto conv_end = std::chrono::high_resolution_clock::now();
    const std::vector<uint16_t>& activated = activated_scratch_;
    // z carries in_proj_z's HK channels per lane; the packed-surface count
    // C includes the value/beta channel groups and must not size it.
    z.assign(p.z, p.z + HK * lanes);

    // Wide-prefill fast path: normalize Q/K, form decay/beta, and advance the
    // recurrent state in one Metal command buffer. Decode deliberately keeps
    // the existing lane-1 path so this feature cannot change decode behavior.
    if (lanes > 1 && (std::getenv("RINDI_GDN_METAL_PREP") ||
                      std::getenv("RINDI_QWEN_PREFILL_FAST"))) {
        const bool ok = recurrence_.step_batch_raw(
            activated.data(), p.a, p.beta, a_log_.data(), dt_bias_.data(),
            lanes, core);
        if (ok && core.size() == H * V * lanes) return true;
    }

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
    const auto prepare_end = std::chrono::high_resolution_clock::now();

    // Always take the Metal batched recurrence, lanes == 1 included: base
    // decoding and speculative verification must execute the IDENTICAL
    // recurrence implementation, otherwise fp16 rounding differences change
    // logits and speculative output stops being exact. The legacy ANE
    // single-step path remains reachable only via RINDI_DISABLE_METAL_RECURRENCE.
    if (!std::getenv("RINDI_DISABLE_METAL_RECURRENCE")) {
        const bool recurrence_ok = recurrence_.step_batch(
            decay_batch, key_batch, query_batch, value_batch,
            beta_batch, lanes, core) && core.size() == H * V * lanes;
        const auto recurrence_end = std::chrono::high_resolution_clock::now();
        if (profile) {
            struct ProfileTotals {
                double conv_ms{0.0};
                double prepare_ms{0.0};
                double recurrence_ms{0.0};
                size_t calls{0};
            };
            static thread_local ProfileTotals totals;
            const auto elapsed = [](const auto& begin, const auto& end) {
                return std::chrono::duration<double, std::milli>(end - begin).count();
            };
            totals.conv_ms += elapsed(profile_start, conv_end);
            totals.prepare_ms += elapsed(conv_end, prepare_end);
            totals.recurrence_ms += elapsed(prepare_end, recurrence_end);
            if (++totals.calls == 48) {
                std::fprintf(stderr,
                    "[GdnProfile] calls=%zu conv_ms=%.3f prepare_ms=%.3f "
                    "recurrence_ms=%.3f total_ms=%.3f\n",
                    totals.calls, totals.conv_ms, totals.prepare_ms,
                    totals.recurrence_ms,
                    totals.conv_ms + totals.prepare_ms + totals.recurrence_ms);
                totals = ProfileTotals{};
            }
        }
        if (recurrence_ok) {
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
