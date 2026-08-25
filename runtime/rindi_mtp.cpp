// SPDX-License-Identifier: Apache-2.0
// runtime/rindi_mtp.cpp - Native MTP draft block (see rindi_mtp.h).
#include "rindi_mtp.h"
#include <cmath>
#include <cstring>
#include <cstdio>

namespace {

float h2f(uint16_t b) {
    uint32_t s = (uint32_t(b) & 0x8000u) << 16, e = (b >> 10) & 31u,
             m = b & 1023u, v = s;
    if (e == 0) v = s | (m << 13);
    else if (e == 31) v = s | 0x7f800000u | (m << 13);
    else v = s | ((e + 112u) << 23) | (m << 13);
    float f; std::memcpy(&f, &v, 4); return f;
}

uint16_t f2h(float f) {
    uint32_t b; std::memcpy(&b, &f, 4);
    int e = int((b >> 23) & 255) - 127 + 15;
    uint32_t s = (b >> 16) & 0x8000u, m = (b >> 13) & 1023u;
    if (e <= 0) return uint16_t(s);
    if (e >= 31) return uint16_t(s | 0x7c00u);
    return uint16_t(s | (uint32_t(e) << 10) | m);
}

MetalContext* shared_mtp_metal_context() {
    static MetalContext* context = metal_context_create();
    return context;
}

constexpr float kEps = 1.0e-6f;

} // namespace

void MtpBlock::rmsnorm16(const std::vector<uint16_t>& x,
                         const std::vector<float>& weight,
                         float eps, std::vector<float>& out) {
    const size_t n = x.size();
    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = h2f(x[i]);
        ss += v * v;
    }
    const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = h2f(x[i]) * inv * weight[i];
}

bool MtpBlock::compile(ANEContext* ane, const SafeTensorsLoader& shard,
                       size_t hidden_dim, size_t context) {
    if ((!ane && !std::getenv("RINDI_TAIL_COREAI")) || hidden_dim == 0 || !shard.has_tensor("mtp.fc.weight")) return false;
    metal_ctx_ = shared_mtp_metal_context();
    if (!metal_ctx_) return false;
    hidden_dim_ = hidden_dim;

    // Hybrid applies +1.0 in fp32 to every 1-dim mtp.* norm weight.
    auto load_norm = [&](const char* name, std::vector<float>& out) {
        std::vector<uint16_t> raw;
        if (!shard.get_tensor_fp16(name, raw) || raw.size() != hidden_dim) return false;
        out.resize(hidden_dim);
        for (size_t i = 0; i < hidden_dim; ++i)
            out[i] = h2f(raw[i]) + 1.0f;
        return true;
    };
    if (!load_norm("mtp.pre_fc_norm_embedding.weight", pre_e_) ||
        !load_norm("mtp.pre_fc_norm_hidden.weight", pre_h_) ||
        !load_norm("mtp.norm.weight", mtp_norm_) ||
        !load_norm("mtp.layers.0.input_layernorm.weight", input_ln_w_) ||
        !load_norm("mtp.layers.0.post_attention_layernorm.weight", post_ln_w_)) {
        std::fprintf(stderr, "[MtpBlock] norm weights failed\n");
        return false;
    }

    if (!attn_.compile_prefixed(ane, shard, "mtp.layers.0.self_attn.",
                                context, 32,
                                RindiAttention::ProjectionMode::Int4FromBf16)) {
        std::fprintf(stderr, "[MtpBlock] draft attention compile failed\n");
        return false;
    }
    if (!fc_.compile_int4_from_bf16(metal_ctx_, shard, "mtp.fc.weight") ||
        !down_.compile_int4_from_bf16(metal_ctx_, shard, "mtp.layers.0.mlp.down_proj.weight")) {
        std::fprintf(stderr, "[MtpBlock] bf16 projections failed\n");
        return false;
    }
    gateup_ready_ = gateup_.compile_int4_from_bf16_fused(
        metal_ctx_, shard,
        {{"mtp.layers.0.mlp.gate_proj.weight", 17408},
         {"mtp.layers.0.mlp.up_proj.weight", 17408}});
    if (!gateup_ready_ &&
        (!gate_.compile_int4_from_bf16(metal_ctx_, shard,
            "mtp.layers.0.mlp.gate_proj.weight") ||
         !up_.compile_int4_from_bf16(metal_ctx_, shard,
            "mtp.layers.0.mlp.up_proj.weight"))) {
        std::fprintf(stderr, "[MtpBlock] mlp projections failed\n");
        return false;
    }
    ready_ = true;
    return true;
}

void MtpBlock::reset() { attn_.reset(); }

bool MtpBlock::forward_core(const float* hidden_prev_f32,
                            const std::vector<float>* /*unused*/,
                            int token_id, std::vector<float>& hidden_out_f32) {
    if (!ready_ || !embed_loader_ || !hidden_prev_f32) return false;
    namespace ch = std::chrono;
    if (!std::getenv("RINDI_DEBUG_MTP_PHASE")) {
        // timed path below shares the same code; keep one control flow
    }
    const bool phase_dbg = std::getenv("RINDI_DEBUG_MTP_PHASE") != nullptr;
    auto t_last = ch::high_resolution_clock::now();
    auto tick = [&](const char* tag) {
        if (!phase_dbg) return;
        const auto now = ch::high_resolution_clock::now();
        std::fprintf(stderr, "[MTPSTAGE] %s=%.3f\n", tag,
                     ch::duration<double, std::milli>(now - t_last).count());
        t_last = now;
    };

    // embedding -> pre_fc_norm_embedding
    emb_row_.resize(hidden_dim_);
    if (!embed_loader_->get_embedding_row_fp16(token_id, hidden_dim_, emb_row_) ||
        emb_row_.size() != hidden_dim_) return false;
    rmsnorm16(emb_row_, pre_e_, kEps, en_);

    // previous hidden -> pre_fc_norm_hidden (input already fp32)
    hn_.resize(hidden_dim_);
    {
        float ss = 0.0f;
        for (size_t i = 0; i < hidden_dim_; ++i) ss += hidden_prev_f32[i] * hidden_prev_f32[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(hidden_dim_) + kEps);
        for (size_t i = 0; i < hidden_dim_; ++i)
            hn_[i] = hidden_prev_f32[i] * inv * pre_h_[i];
    }

    concat_.resize(hidden_dim_ * 2);
    for (size_t i = 0; i < hidden_dim_; ++i) {
        concat_[i] = f2h(en_[i]);
        concat_[hidden_dim_ + i] = f2h(hn_[i]);
    }
    if (!fc_.evaluate(concat_.data(), 1, fc_out_) ||
        fc_out_.size() != hidden_dim_) return false;
    tick("fc");

    // input_layernorm(h0)
    a_in_f_.resize(hidden_dim_);
    {
        float ss = 0.0f;
        for (size_t i = 0; i < hidden_dim_; ++i) {
            const float v = h2f(fc_out_[i]);
            a_in_f_[i] = v;              // reuse as raw holder briefly
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(hidden_dim_) + kEps);
        for (size_t i = 0; i < hidden_dim_; ++i)
            a_in_f_[i] = a_in_f_[i] * inv * input_ln_w_[i];
    }
    attn_in_.resize(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) attn_in_[i] = f2h(a_in_f_[i]);

    if (!attn_.step(attn_in_.data(), 1, attn_out_) ||
        attn_out_.size() != hidden_dim_) return false;
    tick("attn");

    attn_out_f_.resize(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) attn_out_f_[i] = h2f(attn_out_[i]);

    // post_attention_layernorm(a) -> MLP
    {
        float ss = 0.0f;
        for (size_t i = 0; i < hidden_dim_; ++i) ss += attn_out_f_[i] * attn_out_f_[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(hidden_dim_) + kEps);
        attn_in_.resize(hidden_dim_);  // reuse as normalized fp16 staging
        for (size_t i = 0; i < hidden_dim_; ++i)
            attn_in_[i] = f2h(attn_out_f_[i] * inv * post_ln_w_[i]);
    }
    const size_t I = hidden_dim_ * 12 / 10 > 0 ? 17408 : 17408;  // fixed by checkpoint
    (void)I;
    if (gateup_ready_) {
        if (!gateup_.evaluate(attn_in_.data(), 1, gate_out_) ||
            gate_out_.size() != 2 * 17408) return false;
        act_.resize(17408);
        for (size_t i = 0; i < 17408; ++i) {
            const float g = h2f(gate_out_[i]);
            const float u = h2f(gate_out_[17408 + i]);
            act_[i] = f2h((g / (1.0f + std::exp(-g))) * u);
        }
    } else {
        if (!gate_.evaluate(attn_in_.data(), 1, gate_out_) ||
            !up_.evaluate(attn_in_.data(), 1, up_out_) ||
            gate_out_.empty() || gate_out_.size() != up_out_.size()) return false;
        act_.resize(gate_out_.size());
        for (size_t i = 0; i < gate_out_.size(); ++i) {
            const float g = h2f(gate_out_[i]);
            const float u = h2f(up_out_[i]);
            act_[i] = f2h((g / (1.0f + std::exp(-g))) * u);
        }
    }
    if (!down_.evaluate(act_.data(), 1, down_out_) ||
        down_out_.size() != hidden_dim_) return false;
    tick("mlp");

    hidden_out_f32.resize(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i)
        hidden_out_f32[i] = attn_out_f_[i] + h2f(down_out_[i]);
    return true;
}

bool MtpBlock::draft(int token_id,
                     const std::vector<uint16_t>& hidden_in,
                     std::vector<uint16_t>& hidden_out,
                     int& out_token) {
    if (!ready_ || hidden_in.size() != hidden_dim_) return false;
    std::vector<float> hin(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) hin[i] = h2f(hidden_in[i]);

    if (!forward_core(hin.data(), nullptr, token_id, hn_)) return false;

    // mtp_norm over the block output, then greedy lm_head via the engine.
    norm16_.resize(hidden_dim_);
    {
        float ss = 0.0f;
        for (size_t i = 0; i < hidden_dim_; ++i) ss += hn_[i] * hn_[i];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(hidden_dim_) + kEps);
        for (size_t i = 0; i < hidden_dim_; ++i)
            norm16_[i] = f2h(hn_[i] * inv * mtp_norm_[i]);
    }
    const bool ph = std::getenv("RINDI_DEBUG_MTP_PHASE") != nullptr;
    const auto th = std::chrono::high_resolution_clock::now();
    const int token = argmax_fn_ ? argmax_fn_(norm16_) : -1;
    if (ph) std::fprintf(stderr, "[MTPSTAGE] head=%.3f\n",
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - th).count());
    if (token < 0) return false;
    out_token = token;

    hidden_out.resize(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) hidden_out[i] = f2h(hn_[i]);
    return true;
}

bool MtpBlock::advance(int token_id, const std::vector<uint16_t>& hidden_in) {
    std::vector<uint16_t> ho;
    int t;
    return draft(token_id, hidden_in, ho, t);
}
