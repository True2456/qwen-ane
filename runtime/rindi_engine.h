/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_engine.h - Full Standalone C++ 27B Model Engine on Apple Silicon (ANE + Metal GPU).
 */

#ifndef RINDI_ENGINE_H
#define RINDI_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include "bpe_tokenizer.h"
#include "safetensors_loader.h"
#include "rindi_native_chain.h"
#include "rindi_attention.h"
#include "rindi_gdn_layer.h"
#include "rindi_mtp.h"
#include "metal_engine.h"
#include "ane_c_bridge.h"

// Measured generation counters. Every field is filled from a timer inside
// RindiEngine::generate(); nothing here is estimated.
struct GenerationStats {
    size_t prompt_tokens{0};
    size_t generated_tokens{0};   // includes the prefill-derived first token
    double prefill_ms{0.0};
    double ttft_ms{0.0};          // request -> first sampled token
    double decode_ms{0.0};        // first sampled token -> last
    double total_ms{0.0};
    size_t spec_steps{0};          // speculative verify rounds
    double accepted_per_step{0.0}; // confirmed tokens per verify round
    bool spec_used{false};
    double prefill_tps() const {
        return prefill_ms > 0.0 ? static_cast<double>(prompt_tokens) * 1000.0 / prefill_ms : 0.0;
    }
    // The first token costs prefill, not decode, so the decode window holds
    // generated_tokens - 1 tokens.
    double decode_tps() const {
        return decode_ms > 0.0 && generated_tokens > 1
            ? static_cast<double>(generated_tokens - 1) * 1000.0 / decode_ms : 0.0;
    }
};

class RindiEngine {
public:
    RindiEngine(const std::string& model_path);
    ~RindiEngine();

    bool is_ready() const { return ready_; }
    const std::string& get_model_name() const { return model_name_; }

    // Full real generation over 27B weights
    std::string generate(
        const std::string& prompt_text,
        int max_tokens = 128,
        float temperature = 0.7f,
        std::function<void(const std::string& token_text)> stream_cb = nullptr
    );

    // Chat completion formatting & generation
    std::string chat_completion(
        const std::vector<std::pair<std::string, std::string>>& messages,
        const std::string& tools_json = "",
        int max_tokens = 256,
        float temperature = 0.7f,
        std::function<void(const std::string& token_text)> stream_cb = nullptr,
        bool enable_thinking = true,
        const std::string& reasoning_effort = "xhigh"
    );

    double get_last_eval_ms() const { return last_eval_ms_; }
    const GenerationStats& get_last_stats() const { return last_stats_; }

private:
    std::string model_path_;
    std::string model_name_{"Qwen3.8-27B"};
    bool ready_{false};

    size_t hidden_dim_{5120};
    size_t ane_width_{32};   // ANE program lane width (RINDI_ANE_WIDTH)
    size_t num_layers_{64};

    BPETokenizer tokenizer_;
    SafeTensorsLoader safetensors_;
    SafeTensorsLoader lm_head_loader_;
    bool has_separate_lm_head_{false};
    std::unique_ptr<RindiNativeChain> chain_;
    MetalContext* metal_ctx_{nullptr};
    MetalBufferHandle lm_head_gpu_{nullptr};
    MetalBufferHandle lm_head_logits_gpu_{nullptr};
    MetalBufferHandle lm_head_input_gpu_{nullptr};
    MetalBufferHandle lm_head_token_gpu_{nullptr};
    MetalBufferHandle lm_head_batch_input_{nullptr};
    MetalBufferHandle lm_head_batch_logits_{nullptr};
    MetalBufferHandle lm_head_batch_tokens_{nullptr};
    bool lm_head_gpu_ready_{false};

    std::vector<std::unique_ptr<RindiAttention>> attention_layers_;
    std::vector<std::unique_ptr<RindiGdnLayer>> gdn_layers_;
    std::vector<uint16_t> first_input_norm_;
    std::vector<uint16_t> final_norm_;
    bool scheduler_ready_{false};

    double last_eval_ms_{0.0};
    GenerationStats last_stats_;
    MtpBlock mtp_;
    bool mtp_ready_{false};
    int mtp_depth_{2};
    // int4 copy of lm_head used ONLY for draft-token argmax (4x less weight
    // traffic per drafted token than the BF16 head).
    RindiAneProjection lm_head_draft_;
    bool lm_head_draft_ready_{false};

    // Per-token scratch reused across layers and steps to keep the decode
    // loop free of large allocations. next_projection_ deliberately lives
    // across loop iterations: tail N folds layer N+1's projection into it.
    std::vector<uint16_t> hidden_scratch_;
    std::vector<uint16_t> normalized_scratch_;
    std::vector<uint16_t> next_projection_;
    std::vector<uint16_t> core_scratch_;
    std::vector<uint16_t> z_scratch_;
    std::vector<uint16_t> residual_scratch_;
    std::vector<uint16_t> gated_scratch_;
    std::vector<uint16_t> batch_core_scratch_;
    std::vector<uint16_t> batch_gated_scratch_;
    std::vector<uint16_t> batch_z_scratch_;

    bool init_model();
    bool init_gpu_lm_head();
    bool init_scheduler();
    bool init_mtp();

    // Greedy lm-head argmax over an already-normalized hidden vector.
    int argmax_token(const std::vector<uint16_t>& logits_input);
    // Batched greedy predictions over channel-major hidden [C, lanes].
    bool argmax_over_hidden(const std::vector<uint16_t>& hidden,
                            size_t lanes, std::vector<int>& tokens);

    // Speculative-decode rollback. Attention caches rewind by position only:
    // rows beyond position_ are unreachable through causal masking. The GDN
    // conv windows and recurrent states mutate in place, so they need real
    // copies (~75 MB per snapshot, reused across steps).
    struct DecodeSnapshot {
        std::vector<size_t> attn_pos;
        std::vector<std::vector<uint16_t>> gdn_conv, gdn_state;
        size_t mtp_pos{0};
        std::vector<uint16_t> last_hidden;
    };
    void capture_snapshot(DecodeSnapshot& snap) const;
    void restore_snapshot(const DecodeSnapshot& snap);

    // ---- APC: exact-prefix prompt cache ----
    // Stores the full recurrent/KV state at end-of-prefill keyed by prompt
    // tokens. A later request whose tokens extend the cached prefix restores
    // that state and prefills only the suffix.
    struct ApcEntry {
        std::vector<int> tokens;
        std::vector<size_t> attn_pos;
        std::vector<std::vector<uint16_t>> attn_keys, attn_values;
        std::vector<std::vector<uint16_t>> gdn_conv, gdn_state;
        size_t mtp_pos{0};
        std::vector<uint16_t> last_hidden;   // hidden of final prompt token
        bool valid{false};
    };
    ApcEntry apc_;
    bool apc_last_hit_{false};
    size_t apc_last_saved_{0};

public:
    bool last_apc_hit() const { return apc_last_hit_; }
    size_t last_apc_saved() const { return apc_last_saved_; }

private:
    void apc_store(const std::vector<int>& tokens,
                   const std::vector<uint16_t>& last_hidden);
    bool apc_restore();
    bool forward_token(const std::vector<uint16_t>& input,
                       std::vector<uint16_t>& output);
    bool forward_prompt_batch(const std::vector<uint16_t>& input,
                              size_t lanes,
                              std::vector<uint16_t>& output);
    bool apply_rms_norm(const std::vector<uint16_t>& input,
                        const std::vector<uint16_t>& weight,
                        std::vector<uint16_t>& output) const;
    void reset_scheduler();
    int sample_next_token(const std::vector<uint16_t>& hidden,
                          float temperature, std::mt19937& rng);
    bool debug_compare_logits(const std::vector<uint16_t>& reference_input,
                              const std::vector<uint16_t>& metal_input);
};

#endif // RINDI_ENGINE_H
