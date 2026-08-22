// SPDX-License-Identifier: Apache-2.0
// runtime/rindi_mtp.h - Native MTP draft block for speculative decoding.
//
// Mirrors tools/hybrid_serve.py::_init_mtp_head exactly:
//   h0 = fc @ [pre_fc_norm_embedding(embed(t)) ; pre_fc_norm_hidden(h)]
//   a  = mtp_layer(input_layernorm(h0))            (own KV cache)
//   h  = a + mlp(post_attention_layernorm(a))
//   draft_token = argmax(lm_head(mtp_norm(h)))
// Every 1-dim mtp.* norm weight carries the +1.0 offset hybrid applies;
// replicated verbatim.

#ifndef RINDI_MTP_H
#define RINDI_MTP_H

#include "rindi_attention.h"
#include "rindi_ane_projection.h"
#include "safetensors_loader.h"
#include "metal_engine.h"
#include <chrono>
#include <functional>
#include <string>
#include <vector>

class MtpBlock {
public:
    using ArgmaxFn = std::function<int(const std::vector<uint16_t>& normalized)>;

    bool compile(ANEContext* ane, const SafeTensorsLoader& shard,
                 size_t hidden_dim, size_t context);
    bool ready() const { return ready_; }
    void reset();

    void set_argmax_fn(ArgmaxFn fn) { argmax_fn_ = std::move(fn); }
    void set_embed_loader(const SafeTensorsLoader* loader) { embed_loader_ = loader; }

    // One greedy draft step. Consumes one slot of the draft KV cache.
    bool draft(int token_id,
               const std::vector<uint16_t>& hidden_in,
               std::vector<uint16_t>& hidden_out,
               int& out_token);

    // Re-run the block over an already-confirmed token purely to keep the
    // draft KV cache aligned after a rollback+replay. Outputs discarded.
    bool advance(int token_id, const std::vector<uint16_t>& hidden_in);

    size_t position() const { return attn_.position(); }
    void snapshot_kv(size_t from_pos, std::vector<uint16_t>& k,
                     std::vector<uint16_t>& v) const {
        attn_.snapshot_kv(from_pos, k, v);
    }
    void restore_kv(size_t from_pos, const std::vector<uint16_t>& k,
                    const std::vector<uint16_t>& v) {
        attn_.restore_kv(from_pos, k, v);
    }

private:
    static void rmsnorm16(const std::vector<uint16_t>& x,
                          const std::vector<float>& weight,
                          float eps, std::vector<float>& out);
    // Shared pipeline: returns the post-block hidden in fp32.
    bool forward_core(const float* hidden_prev_f32,
                      const std::vector<float>* hidden_prev_normed /*unused*/,
                      int token_id, std::vector<float>& hidden_out);

    RindiAttention attn_;
    RindiAneProjection fc_;
    RindiAneProjection gate_, up_, down_;
    // Optional fused gate/up (rows [gate; up]); silu splits the halves.
    RindiAneProjection gateup_;
    bool gateup_ready_{false};
    std::vector<float> pre_e_, pre_h_, mtp_norm_, input_ln_w_, post_ln_w_;
    MetalContext* metal_ctx_{nullptr};
    ArgmaxFn argmax_fn_;
    const SafeTensorsLoader* embed_loader_{nullptr};
    size_t hidden_dim_{0};
    bool ready_{false};
    std::chrono::high_resolution_clock::time_point t_last_;

    // Reused scratch.
    std::vector<uint16_t> emb_row_, concat_, fc_out_, attn_in_, attn_out_,
        gate_out_, up_out_, act_, down_out_, norm16_;
    std::vector<float> en_, hn_, a_in_f_, attn_out_f_;
};

#endif // RINDI_MTP_H
