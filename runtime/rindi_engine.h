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
#include "metal_engine.h"
#include "ane_c_bridge.h"

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

private:
    std::string model_path_;
    std::string model_name_{"Qwen3.8-27B"};
    bool ready_{false};

    size_t hidden_dim_{5120};
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
    bool lm_head_gpu_ready_{false};

    std::vector<std::unique_ptr<RindiAttention>> attention_layers_;
    std::vector<std::unique_ptr<RindiGdnLayer>> gdn_layers_;
    std::vector<uint16_t> first_input_norm_;
    std::vector<uint16_t> final_norm_;
    bool scheduler_ready_{false};

    double last_eval_ms_{0.0};

    bool init_model();
    bool init_gpu_lm_head();
    bool init_scheduler();
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
