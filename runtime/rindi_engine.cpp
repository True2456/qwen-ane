/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_engine.cpp - Full Implementation of Standalone C++ 27B Inference Engine.
 */

#include "rindi_engine.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

namespace {
float engine_half_to_float(uint16_t bits) {
    uint32_t s=(uint32_t(bits)&0x8000u)<<16, e=(bits>>10)&31u, m=bits&1023u, v=s;
    if(e==0)v=s|(m<<13);else if(e==31)v=s|0x7f800000u|(m<<13);else v=s|((e+112u)<<23)|(m<<13);
    float f;std::memcpy(&f,&v,4);return f;
}

uint16_t engine_float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = (bits >> 13) & 0x3ffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}

}

RindiEngine::RindiEngine(const std::string& model_path)
    : model_path_(model_path) {
    ready_ = init_model();
}

RindiEngine::~RindiEngine() {
    if (lm_head_gpu_) metal_buffer_release(lm_head_gpu_);
    if (lm_head_logits_gpu_) metal_buffer_release(lm_head_logits_gpu_);
    if (lm_head_input_gpu_) metal_buffer_release(lm_head_input_gpu_);
    if (lm_head_token_gpu_) metal_buffer_release(lm_head_token_gpu_);
    if (metal_ctx_) {
        metal_context_destroy(metal_ctx_);
    }
}

bool RindiEngine::init_model() {
    std::cout << "[RindiEngine] Initializing Apple Silicon 27B Model Engine..." << std::endl;

    // 1. Initialize Metal Context
    metal_ctx_ = metal_context_create();
    if (metal_ctx_) {
        std::cout << "  [Metal GPU] " << metal_get_device_name(metal_ctx_) << " initialized." << std::endl;
    }

    // 2. Load Tokenizer
    std::string tok_path = model_path_ + "/tokenizer.json";
    if (!tokenizer_.load(tok_path)) {
        tok_path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B/tokenizer.json";
        tokenizer_.load(tok_path);
    }

    // 3. Load Safetensors Backbone
    std::string st_path = model_path_ + "/gpu_backbone.safetensors";
    if (!safetensors_.open_file(st_path)) {
        st_path = model_path_ + "/model.safetensors";
        if (!safetensors_.open_file(st_path)) {
            st_path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi/gpu_backbone.safetensors";
            safetensors_.open_file(st_path);
        }
    }

    // Qwen3.8 is untied: lm_head.weight is not present in the packaged GPU
    // backbone.  Keep it memory-mapped from the original final safetensors
    // shard so native sampling uses the model's actual output projection.
    std::vector<std::string> lm_head_candidates;
    if (const char* explicit_head = std::getenv("RINDI_LM_HEAD_FILE"))
        lm_head_candidates.emplace_back(explicit_head);
    std::string original_model = model_path_;
    const std::string rindi_suffix = ".rindi";
    if (original_model.size() > rindi_suffix.size() &&
        original_model.compare(original_model.size() - rindi_suffix.size(),
                               rindi_suffix.size(), rindi_suffix) == 0) {
        original_model.resize(original_model.size() - rindi_suffix.size());
    }
    lm_head_candidates.push_back(original_model + "/model-00018-of-00018.safetensors");
    lm_head_candidates.push_back(model_path_ + "/lm_head.safetensors");
    for (const auto& candidate : lm_head_candidates) {
        if (access(candidate.c_str(), R_OK) != 0) continue;
        if (lm_head_loader_.open_file(candidate) &&
            lm_head_loader_.has_tensor("lm_head.weight")) {
            has_separate_lm_head_ = true;
            std::cout << "  [Native LM head] Using separate lm_head.weight from "
                      << candidate << std::endl;
            break;
        }
        lm_head_loader_.close();
    }
    if (!has_separate_lm_head_) {
        std::cerr << "[RindiEngine] Separate lm_head.weight not found; "
                      << "falling back to tied embedding head" << std::endl;
    } else if (!init_gpu_lm_head()) {
        std::cerr << "[RindiEngine] Metal LM head setup failed; "
                  << "falling back to CPU LM-head sampling" << std::endl;
    }

    // 4. Reconstruct and compile the native transformer scheduler. The raw
    // .rindi/ane_layers entries are quantized weight caches, so they must be
    // paired with the safetensors metadata and compiled into the exact fused
    // tail programs before execution.
    // ANE program lane-width (columns per eval). 32 = shipped default; wider
    // values recompile the cores with more parallel lanes for prefill chunks.
    // GDN causal conv reserves 3 columns, so live prefill lanes = width - 3.
    {
        size_t w = 32;
        if (const char* e = std::getenv("RINDI_ANE_WIDTH")) {
            size_t v = (size_t)std::atoi(e);
            if (v >= 32 && v <= 256 && (v % 32) == 0) w = v;
        }
        ane_width_ = w;
    }
    chain_ = std::make_unique<RindiNativeChain>(hidden_dim_, ane_width_);
    scheduler_ready_ = init_scheduler();
    if (!scheduler_ready_) {
        std::cerr << "[RindiEngine] Native transformer scheduler failed to initialize" << std::endl;
        return false;
    }

    init_mtp();
    std::cout << "[RindiEngine] Model Engine Online (native C++ scheduler, 27B Parameters, 64 Layers, 5120 Hidden Dim)." << std::endl;
    return true;
}

bool RindiEngine::init_gpu_lm_head() {
    if (!metal_ctx_ || !has_separate_lm_head_) return false;
    const TensorInfo* info = lm_head_loader_.get_tensor_info("lm_head.weight");
    if (!info || info->dtype != "BF16" || info->shape.size() != 2 ||
        info->shape[0] <= 0 || info->shape[1] != static_cast<int64_t>(hidden_dim_)) {
        std::cerr << "[RindiEngine] Unsupported LM-head tensor layout" << std::endl;
        return false;
    }
    const size_t vocab = static_cast<size_t>(info->shape[0]);
    const size_t bytes = static_cast<size_t>(info->nbytes);
    const void* source = lm_head_loader_.get_tensor_data("lm_head.weight");
    if (!source || bytes != vocab * hidden_dim_ * sizeof(uint16_t)) return false;

    lm_head_gpu_ = metal_buffer_create(metal_ctx_, bytes);
    lm_head_logits_gpu_ = metal_buffer_create(metal_ctx_, vocab * sizeof(uint16_t));
    lm_head_input_gpu_ = metal_buffer_create(metal_ctx_, hidden_dim_ * sizeof(uint16_t));
    lm_head_token_gpu_ = metal_buffer_create(metal_ctx_, sizeof(int32_t));
    if (!lm_head_gpu_ || !lm_head_logits_gpu_ || !lm_head_input_gpu_ || !lm_head_token_gpu_) {
        if (lm_head_gpu_) { metal_buffer_release(lm_head_gpu_); lm_head_gpu_ = nullptr; }
        if (lm_head_logits_gpu_) { metal_buffer_release(lm_head_logits_gpu_); lm_head_logits_gpu_ = nullptr; }
        if (lm_head_input_gpu_) { metal_buffer_release(lm_head_input_gpu_); lm_head_input_gpu_ = nullptr; }
        if (lm_head_token_gpu_) { metal_buffer_release(lm_head_token_gpu_); lm_head_token_gpu_ = nullptr; }
        return false;
    }
    std::memcpy(metal_buffer_get_contents(lm_head_gpu_), source, bytes);
    lm_head_gpu_ready_ = true;
    std::cout << "  [Metal GPU] LM head resident (BF16, " << vocab
              << " x " << hidden_dim_ << ")" << std::endl;
    return true;
}

bool RindiEngine::init_scheduler() {
    if (!chain_ || !chain_->ane_context()) return false;
    const std::string ane_dir = model_path_ + "/ane_layers";
    if (!safetensors_.has_tensor("layers.0.input_layernorm.weight") ||
        !safetensors_.get_tensor_fp16("layers.0.input_layernorm.weight", first_input_norm_) ||
        first_input_norm_.size() != hidden_dim_ ||
        !safetensors_.get_tensor_fp16("norm.weight", final_norm_) ||
        final_norm_.size() != hidden_dim_) {
        std::cerr << "[RindiEngine] Missing input/final RMSNorm weights" << std::endl;
        return false;
    }

    for (size_t layer = 0; layer < num_layers_; ++layer) {
        if (!chain_->compile_layer(static_cast<int>(layer), ane_dir, safetensors_)) {
            std::cerr << "[RindiEngine] Failed to compile fused tail " << layer << std::endl;
            return false;
        }
    }
    std::cout << "  [ANE Hardware] Compiled 64 fused transformer tails." << std::endl;

    attention_layers_.resize(num_layers_);
    gdn_layers_.resize(num_layers_);
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        if (safetensors_.has_tensor(p + "self_attn.o_proj.weight")) {
            attention_layers_[layer] = std::make_unique<RindiAttention>();
            if (!attention_layers_[layer]->compile_core(chain_->ane_context(), safetensors_,
                                                        static_cast<int>(layer), 4096,
                                                        ane_width_)) {
                std::cerr << "[RindiEngine] Failed to compile attention core " << layer << std::endl;
                return false;
            }
        } else {
            gdn_layers_[layer] = std::make_unique<RindiGdnLayer>();
            if (!gdn_layers_[layer]->compile_core(chain_->ane_context(), safetensors_,
                                                  static_cast<int>(layer),
                                                  ane_width_)) {
                std::cerr << "[RindiEngine] Failed to compile GDN core " << layer << std::endl;
                return false;
            }
        }
    }
    std::cout << "  [ANE Hardware] Compiled native attention/GDN cores and state." << std::endl;
    if (chain_->compile_metal_tails(safetensors_, ane_dir)) {
        std::cout << "  [Metal GPU] Compiled batched transformer tails." << std::endl;
    } else {
        std::cout << "  [Metal GPU] Batched tail unavailable; retaining ANE tail fallback." << std::endl;
    }
    return true;
}

bool RindiEngine::init_mtp() {
    const char* depth_env = std::getenv("RINDI_MTP_DEPTH");
    if (depth_env) mtp_depth_ = std::max(0, std::atoi(depth_env));
    if (std::getenv("RINDI_DISABLE_MTP")) mtp_depth_ = 0;
    if (mtp_depth_ <= 0) {
        std::cout << "  [MTP] disabled" << std::endl;
        return false;
    }
    if (!lm_head_loader_.has_tensor("mtp.fc.weight")) {
        std::cout << "  [MTP] no mtp tensors in shard; disabled" << std::endl;
        return false;
    }
    if (!mtp_.compile(chain_->ane_context(), lm_head_loader_, hidden_dim_, 4096)) {
        std::cout << "  [MTP] compile failed; speculative decode unavailable" << std::endl;
        return false;
    }
    // Draft-time head: quantized once at load; argmax scans the fp16 logits.
    lm_head_draft_ready_ =
        lm_head_draft_.compile_int4_from_bf16(metal_ctx_, lm_head_loader_,
                                              "lm_head.weight");
    if (!lm_head_draft_ready_)
        std::cout << "  [MTP] draft lm_head unavailable; using BF16 head"
                  << std::endl;
    mtp_.set_embed_loader(&safetensors_);
    mtp_.set_argmax_fn([this](const std::vector<uint16_t>& norm) {
        if (lm_head_draft_ready_) {
            std::vector<uint16_t> logits;
            if (lm_head_draft_.evaluate(norm.data(), 1, logits) &&
                !logits.empty()) {
                const size_t vocab = std::min(tokenizer_.vocab_size(), logits.size());
                int best = 0;
                float best_val = engine_half_to_float(logits[0]);
                for (size_t v = 1; v < vocab; ++v) {
                    const float val = engine_half_to_float(logits[v]);
                    if (val > best_val) { best_val = val; best = static_cast<int>(v); }
                }
                return best;
            }
        }
        return this->argmax_token(norm);
    });
    mtp_ready_ = true;
    std::cout << "  [MTP] draft block resident (depth " << mtp_depth_
              << ", greedy only)" << std::endl;
    return true;
}

int RindiEngine::argmax_token(const std::vector<uint16_t>& logits_input) {
    if (logits_input.size() != hidden_dim_) return -1;
    const size_t vocab = std::min(tokenizer_.vocab_size(), size_t(248320));
    if (!lm_head_gpu_ready_ || vocab == 0) return -1;
    std::memcpy(metal_buffer_get_contents(lm_head_input_gpu_),
                logits_input.data(), hidden_dim_ * sizeof(uint16_t));
    MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
    if (!cmd) return -1;
    metal_dispatch_gemm_bf16(metal_ctx_, cmd, lm_head_input_gpu_,
                             lm_head_gpu_, lm_head_logits_gpu_,
                             1, static_cast<int>(vocab),
                             static_cast<int>(hidden_dim_));
    if (lm_head_token_gpu_) {
        metal_dispatch_argmax_fp16(metal_ctx_, cmd, lm_head_logits_gpu_,
                                   lm_head_token_gpu_, 1,
                                   static_cast<int>(vocab));
    }
    metal_command_buffer_commit(cmd);
    metal_command_buffer_wait(cmd);
    if (lm_head_token_gpu_) {
        const int32_t token = *static_cast<const int32_t*>(
            metal_buffer_get_contents(lm_head_token_gpu_));
        if (token >= 0 && static_cast<size_t>(token) < vocab) return token;
        return -1;
    }
    // No argmax buffer: scan the logits row on CPU.
    const uint16_t* logits = static_cast<const uint16_t*>(
        metal_buffer_get_contents(lm_head_logits_gpu_));
    if (!logits) return -1;
    int best = 0;
    float best_val = engine_half_to_float(logits[0]);
    for (size_t v = 1; v < vocab; ++v) {
        const float val = engine_half_to_float(logits[v]);
        if (val > best_val) { best_val = val; best = static_cast<int>(v); }
    }
    return best;
}

bool RindiEngine::argmax_over_hidden(const std::vector<uint16_t>& hidden,
                                     size_t lanes, std::vector<int>& tokens) {
    tokens.assign(lanes, -1);
    const size_t vocab = std::min(tokenizer_.vocab_size(), size_t(248320));
    if (!lm_head_gpu_ready_ || vocab == 0 || lanes == 0 || lanes > 32 ||
        hidden.size() != hidden_dim_ * lanes) return false;
    // Normalize each lane with the final norm into a row-major staging block.
    if (!lm_head_batch_input_) {
        lm_head_batch_input_ = metal_buffer_create(metal_ctx_, 32 * hidden_dim_ * sizeof(uint16_t));
        lm_head_batch_logits_ = metal_buffer_create(metal_ctx_, 32 * vocab * sizeof(uint16_t));
        lm_head_batch_tokens_ = metal_buffer_create(metal_ctx_, 32 * sizeof(int32_t));
        if (!lm_head_batch_input_ || !lm_head_batch_logits_ || !lm_head_batch_tokens_)
            return false;
    }
    auto* staged = static_cast<uint16_t*>(metal_buffer_get_contents(lm_head_batch_input_));
    std::vector<uint16_t> lane_in(hidden_dim_), lane_norm;
    for (size_t lane = 0; lane < lanes; ++lane) {
        for (size_t c = 0; c < hidden_dim_; ++c)
            lane_in[c] = hidden[c * lanes + lane];
        if (!apply_rms_norm(lane_in, final_norm_, lane_norm)) return false;
        std::memcpy(staged + lane * hidden_dim_, lane_norm.data(),
                    hidden_dim_ * sizeof(uint16_t));
    }
    MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
    if (!cmd) return false;
    metal_dispatch_gemm_bf16(metal_ctx_, cmd, lm_head_batch_input_,
                             lm_head_gpu_, lm_head_batch_logits_,
                             static_cast<int>(lanes), static_cast<int>(vocab),
                             static_cast<int>(hidden_dim_));
    metal_dispatch_argmax_fp16(metal_ctx_, cmd, lm_head_batch_logits_,
                               lm_head_batch_tokens_, static_cast<int>(lanes),
                               static_cast<int>(vocab));
    metal_command_buffer_commit(cmd);
    metal_command_buffer_wait(cmd);
    const int32_t* toks = static_cast<const int32_t*>(
        metal_buffer_get_contents(lm_head_batch_tokens_));
    for (size_t lane = 0; lane < lanes; ++lane) {
        if (toks[lane] >= 0 && static_cast<size_t>(toks[lane]) < vocab)
            tokens[lane] = toks[lane];
    }
    return true;
}

void RindiEngine::capture_snapshot(DecodeSnapshot& snap) const {
    snap.attn_pos.clear();
    for (const auto& a : attention_layers_) {
        snap.attn_pos.push_back(a ? a->position() : 0);
    }
    snap.gdn_conv.resize(gdn_layers_.size());
    snap.gdn_state.resize(gdn_layers_.size());
    for (size_t i = 0; i < gdn_layers_.size(); ++i) {
        if (gdn_layers_[i]) gdn_layers_[i]->snapshot_state(snap.gdn_conv[i],
                                                           snap.gdn_state[i]);
    }
    snap.mtp_pos = mtp_.position();
}

void RindiEngine::restore_snapshot(const DecodeSnapshot& snap) {
    for (size_t i = 0; i < attention_layers_.size() && i < snap.attn_pos.size(); ++i) {
        if (attention_layers_[i]) attention_layers_[i]->set_position(snap.attn_pos[i]);
    }
    for (size_t i = 0; i < gdn_layers_.size() && i < snap.gdn_conv.size(); ++i) {
        if (gdn_layers_[i]) gdn_layers_[i]->restore_state(snap.gdn_conv[i],
                                                          snap.gdn_state[i]);
    }
    mtp_.restore_kv(snap.mtp_pos, {}, {});
}

// Snapshot the complete prefill-end state (attention KV incl. host+Metal
// mirrors, GDN conv/recurrence, MTP draft position) keyed by prompt tokens.
void RindiEngine::apc_store(const std::vector<int>& tokens,
                            const std::vector<uint16_t>& last_hidden) {
    apc_ = ApcEntry{};
    apc_.tokens = tokens;
    for (const auto& a : attention_layers_) {
        if (!a) { apc_.attn_pos.push_back(0);
                  apc_.attn_keys.emplace_back();
                  apc_.attn_values.emplace_back(); continue; }
        apc_.attn_pos.push_back(a->position());
        std::vector<uint16_t> k, v;
        a->snapshot_kv(0, k, v);
        apc_.attn_keys.push_back(std::move(k));
        apc_.attn_values.push_back(std::move(v));
    }
    apc_.gdn_conv.resize(gdn_layers_.size());
    apc_.gdn_state.resize(gdn_layers_.size());
    for (size_t i = 0; i < gdn_layers_.size(); ++i)
        if (gdn_layers_[i])
            gdn_layers_[i]->snapshot_state(apc_.gdn_conv[i], apc_.gdn_state[i]);
    apc_.mtp_pos = mtp_.position();
    apc_.last_hidden = last_hidden;
    apc_.valid = !tokens.empty();
}

// Restore the cached state so prefill can resume at the matched prefix.
bool RindiEngine::apc_restore() {
    if (!apc_.valid) return false;
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
        auto* a = attention_layers_[i].get();
        if (!a || i >= apc_.attn_keys.size()) continue;
        a->restore_kv(0, apc_.attn_keys[i], apc_.attn_values[i]); // pos <- row count
        a->set_position(apc_.attn_pos[i]);                        // then exact pos
    }
    for (size_t i = 0; i < gdn_layers_.size() && i < apc_.gdn_conv.size(); ++i)
        if (gdn_layers_[i])
            gdn_layers_[i]->restore_state(apc_.gdn_conv[i], apc_.gdn_state[i]);
    mtp_.restore_kv(apc_.mtp_pos, {}, {});
    return true;
}

bool RindiEngine::apply_rms_norm(const std::vector<uint16_t>& input,
                                 const std::vector<uint16_t>& weight,
                                 std::vector<uint16_t>& output) const {
    if (input.size() != hidden_dim_ || weight.size() != hidden_dim_) return false;
    float sum = 0.0f;
    for (uint16_t value : input) {
        const float x = engine_half_to_float(value);
        sum += x * x;
    }
    const float inv = 1.0f / std::sqrt(sum / static_cast<float>(hidden_dim_) + 1.0e-6f);
    output.resize(hidden_dim_);
    for (size_t i = 0; i < hidden_dim_; ++i) {
        output[i] = engine_float_to_half(engine_half_to_float(input[i]) * inv *
                                         engine_half_to_float(weight[i]));
    }
    return true;
}

void RindiEngine::reset_scheduler() {
    for (auto& layer : attention_layers_) {
        if (layer) layer->reset();
    }
    for (auto& layer : gdn_layers_) {
        if (layer) layer->reset();
    }
}

bool RindiEngine::forward_token(const std::vector<uint16_t>& input,
                                std::vector<uint16_t>& output) {
    if (!scheduler_ready_ || input.size() != hidden_dim_ ||
        attention_layers_.size() != num_layers_ || gdn_layers_.size() != num_layers_) {
        return false;
    }

    static thread_local size_t debug_forward_index = 0;
    const auto timing_start = std::chrono::high_resolution_clock::now();
    const size_t this_forward_index = debug_forward_index++;
    const char* debug_index_text = std::getenv("RINDI_DEBUG_FORWARD_INDEX");
    const bool debug_layers = std::getenv("RINDI_DEBUG_LAYER_STATS") &&
                              debug_index_text &&
                              this_forward_index == static_cast<size_t>(std::strtoull(debug_index_text, nullptr, 10));

    std::vector<uint16_t> hidden = input;
    std::vector<uint16_t> normalized;
    if (!apply_rms_norm(hidden, first_input_norm_, normalized)) {
        std::cerr << "[RindiEngine] first RMSNorm failed" << std::endl;
        return false;
    }

    // These bind to the member scratch buffers: tail N folds layer N+1's
    // projection into next_projection_, and each layer's core output lands in
    // core_scratch_ without a fresh allocation.
    std::vector<uint16_t>& next_projection = next_projection_;
    std::vector<uint16_t>& core = core_scratch_;
    std::vector<uint16_t>& z = z_scratch_;
    std::vector<uint16_t>& gated = gated_scratch_;
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        const auto layer_core_start = std::chrono::high_resolution_clock::now();
        residual_scratch_ = hidden;
        const std::vector<uint16_t>& residual = residual_scratch_;
        const bool attention = attention_layers_[layer] != nullptr;

        if (layer == 0) {
            if (attention) {
                std::vector<uint16_t> q, k, v;
                std::vector<uint16_t> attended;
                if (!attention_layers_[layer]->project(normalized.data(), 1, q, k, v) ||
                    !attention_layers_[layer]->core_step(q, k, v, attended)) {
                    std::cerr << "[RindiEngine] attention core failed at layer " << layer << std::endl;
                    return false;
                }
                // q_proj is [q, gate]. Match mlx_lm: apply sigmoid(gate)
                // before the fused tail, whose input is core + residual.
                core.swap(attended);
                for (size_t h = 0; h < 24; ++h) {
                    for (size_t d = 0; d < 256; ++d) {
                        const size_t c = h * 256 + d;
                        const float gate = engine_half_to_float(q[h * 512 + 256 + d]);
                        const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
                        core[c] = engine_float_to_half(
                            engine_half_to_float(core[c]) * sigmoid);
                    }
                }
            } else if (!gdn_layers_[layer]->core_step(normalized.data(), 1, core, z)) {
                std::cerr << "[RindiEngine] GDN core failed at layer " << layer << std::endl;
                return false;
            } else {
                if (!gdn_layers_[layer]->gate_core(core, z, gated)) return false;
                core.swap(gated);
            }
        } else {
            if (next_projection.empty()) {
                std::cerr << "[RindiEngine] missing folded projection at layer " << layer << std::endl;
                return false;
            }
            if (attention) {
                constexpr size_t Q = 24 * 256;
                constexpr size_t K = 4 * 256;
                constexpr size_t QG = 2 * Q;
                if (next_projection.size() != QG + 2 * K) {
                    std::cerr << "[RindiEngine] attention folded projection shape at layer " << layer
                              << ": " << next_projection.size() << std::endl;
                    return false;
                }
                // q_proj is laid out per head as [q(256), gate(256)]. Read the
                // q/k/v slices straight out of the folded projection buffer.
                const uint16_t* np = next_projection.data();
                if (!attention_layers_[layer]->core_step(np, QG, np + QG, K,
                                                         np + QG + K, K, core)) {
                    std::cerr << "[RindiEngine] attention folded core failed at layer " << layer << std::endl;
                    return false;
                }
                for (size_t h = 0; h < 24; ++h) {
                    for (size_t d = 0; d < 256; ++d) {
                        const size_t c = h * 256 + d;
                        const float gate = engine_half_to_float(np[h * 512 + 256 + d]);
                        const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
                        core[c] = engine_float_to_half(
                            engine_half_to_float(core[c]) * sigmoid);
                    }
                }
            } else {
                constexpr size_t QKV = 10240;
                constexpr size_t Z = 6144;
                constexpr size_t G = 48;
                if (next_projection.size() != QKV + Z + 2 * G) {
                    std::cerr << "[RindiEngine] GDN folded projection shape at layer " << layer
                              << ": " << next_projection.size() << std::endl;
                    return false;
                }
                const uint16_t* np = next_projection.data();
                RindiGdnProjectionView projected{np, np + QKV, np + QKV + Z,
                                                 np + QKV + Z + G};
                if (!gdn_layers_[layer]->core_from_projected_view(projected, 1, core, z)) {
                    std::cerr << "[RindiEngine] GDN folded core failed at layer " << layer << std::endl;
                    return false;
                }
                if (!gdn_layers_[layer]->gate_core(core, z, gated)) return false;
                core.swap(gated);
            }
        }

        const bool has_next = layer + 1 < num_layers_;
        if (debug_layers && layer == 0) {
            float core_sq = 0.0f, residual_sq = 0.0f;
            for (uint16_t value : core) {
                const float x = engine_half_to_float(value); core_sq += x * x;
            }
            for (uint16_t value : residual) {
                const float x = engine_half_to_float(value); residual_sq += x * x;
            }
            std::cerr << "[RindiDebug] layer0 core_rms="
                      << std::sqrt(core_sq / core.size())
                      << " residual_rms="
                      << std::sqrt(residual_sq / residual.size()) << std::endl;
        }
        const auto layer_tail_start = std::chrono::high_resolution_clock::now();
        if (!chain_->evaluate_tail(static_cast<int>(layer), core.data(), core.size(),
                                   residual.data(), hidden, has_next ? &next_projection : nullptr)) {
            std::cerr << "[RindiEngine] fused tail failed at layer " << layer << std::endl;
            return false;
        }
        if (std::getenv("RINDI_DEBUG_TIMING") && this_forward_index == 0) {
            const auto layer_end = std::chrono::high_resolution_clock::now();
            const double core_ms = std::chrono::duration<double, std::milli>(
                layer_tail_start - layer_core_start).count();
            const double tail_ms = std::chrono::duration<double, std::milli>(
                layer_end - layer_tail_start).count();
            std::cerr << "[RindiTiming] layer=" << layer
                      << " core_ms=" << core_ms
                      << " tail_ms=" << tail_ms << std::endl;
        }
        if (debug_layers) {
            float sum = 0.0f;
            for (uint16_t value : hidden) {
                const float x = engine_half_to_float(value);
                sum += x * x;
            }
            std::cerr << "[RindiDebug] forward=" << this_forward_index
                      << " layer=" << layer << " type=" << (attention ? "attn" : "gdn")
                      << " hidden_rms=" << std::sqrt(sum / hidden.size()) << std::endl;
        }
        if (has_next) {
            // The fused tail has already applied the next block's RMSNorm and
            // input projection. It is therefore the next layer's core input,
            // with no host-side normalization or projection required.
            normalized.clear();
        }
    }
    output = std::move(hidden);
    if (std::getenv("RINDI_DEBUG_TIMING")) {
        const auto timing_end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            timing_end - timing_start).count();
        std::cerr << "[RindiTiming] forward=" << this_forward_index
                  << " ms=" << elapsed_ms << std::endl;
    }
    return true;
}

bool RindiEngine::forward_prompt_batch(const std::vector<uint16_t>& input,
                                       size_t lanes,
                                       std::vector<uint16_t>& output) {
    // GDN's causal convolution has three history columns in a 32-column ANE
    // surface, so at most 29 new prompt tokens can be submitted together.
    if (!scheduler_ready_ || lanes == 0 || lanes > 29 ||
        input.size() != hidden_dim_ * lanes ||
        attention_layers_.size() != num_layers_ ||
        gdn_layers_.size() != num_layers_) return false;

    constexpr size_t Q = 24 * 256;
    constexpr size_t K = 4 * 256;
    constexpr size_t QG = 2 * Q;
    constexpr size_t QKV = 10240;
    constexpr size_t Z = 6144;
    constexpr size_t G = 48;
    static thread_local size_t debug_batch_index = 0;
    const size_t this_batch_index = debug_batch_index++;
    const bool debug_batch_timing = std::getenv("RINDI_DEBUG_TIMING") &&
                                     this_batch_index == 0;

    std::vector<uint16_t> hidden = input;
    std::vector<uint16_t> normalized;
    std::vector<uint16_t>& next_projection = next_projection_;
    std::vector<uint16_t>& core = batch_core_scratch_;
    std::vector<uint16_t>& z = batch_z_scratch_;
    std::vector<uint16_t>& gated = batch_gated_scratch_;

    for (size_t layer = 0; layer < num_layers_; ++layer) {
        const auto layer_start = std::chrono::high_resolution_clock::now();
        residual_scratch_ = hidden;
        const std::vector<uint16_t>& residual = residual_scratch_;
        const bool attention = attention_layers_[layer] != nullptr;

        if (layer == 0) {
            normalized.resize(hidden_dim_ * lanes);
            std::vector<uint16_t> lane_in(hidden_dim_), lane_norm;
            for (size_t lane = 0; lane < lanes; ++lane) {
                for (size_t c = 0; c < hidden_dim_; ++c)
                    lane_in[c] = hidden[c * lanes + lane];
                if (!apply_rms_norm(lane_in, first_input_norm_, lane_norm) ||
                    lane_norm.size() != hidden_dim_) return false;
                for (size_t c = 0; c < hidden_dim_; ++c)
                    normalized[c * lanes + lane] = lane_norm[c];
            }
        }

        if (layer == 0) {
            if (attention) {
                std::vector<uint16_t> q, k, v, attended;
                if (!attention_layers_[layer]->project(normalized.data(), lanes, q, k, v) ||
                    !attention_layers_[layer]->core_step_batch(q, k, v, lanes, attended))
                    return false;
                core = std::move(attended);
                for (size_t h = 0; h < 24; ++h) {
                    for (size_t d = 0; d < 256; ++d) {
                        const size_t c = h * 256 + d;
                        const size_t qc = h * 512 + 256 + d;
                        for (size_t lane = 0; lane < lanes; ++lane) {
                            const float gate = engine_half_to_float(q[qc * lanes + lane]);
                            core[c * lanes + lane] = engine_float_to_half(
                                engine_half_to_float(core[c * lanes + lane]) /
                                (1.0f + std::exp(-gate)));
                        }
                    }
                }
            } else {
                std::vector<uint16_t> gated;
                if (!gdn_layers_[layer]->core_step(normalized.data(), lanes, core, z) ||
                    !gdn_layers_[layer]->gate_core_batch(core, z, lanes, gated))
                    return false;
                core = std::move(gated);
            }
        } else {
            if (next_projection.empty()) return false;
            if (attention) {
                if (next_projection.size() != (QG + 2 * K) * lanes) return false;
                const uint16_t* np = next_projection.data();
                if (!attention_layers_[layer]->core_step_batch(np, np + QG * lanes,
                                                               np + (QG + K) * lanes,
                                                               lanes, core))
                    return false;
                for (size_t h = 0; h < 24; ++h) {
                    for (size_t d = 0; d < 256; ++d) {
                        const size_t c = h * 256 + d;
                        const size_t qc = h * 512 + 256 + d;
                        for (size_t lane = 0; lane < lanes; ++lane) {
                            const float gate = engine_half_to_float(np[qc * lanes + lane]);
                            core[c * lanes + lane] = engine_float_to_half(
                                engine_half_to_float(core[c * lanes + lane]) /
                                (1.0f + std::exp(-gate)));
                        }
                    }
                }
            } else {
                if (next_projection.size() != (QKV + Z + 2 * G) * lanes) return false;
                const uint16_t* np = next_projection.data();
                RindiGdnProjectionView projected{np, np + QKV * lanes,
                                                 np + (QKV + Z) * lanes,
                                                 np + (QKV + Z + G) * lanes};
                if (!gdn_layers_[layer]->core_from_projected_view(projected, lanes, core, z) ||
                    !gdn_layers_[layer]->gate_core_batch(core, z, lanes, gated))
                    return false;
                core.swap(gated);
            }
        }

        const bool has_next = layer + 1 < num_layers_;
        const auto core_end = std::chrono::high_resolution_clock::now();
        if (!chain_->evaluate_tail_batch(static_cast<int>(layer), core.data(),
                                         core.size() / lanes, residual.data(), lanes,
                                         hidden, has_next ? &next_projection : nullptr))
            return false;
        if (debug_batch_timing) {
            const auto layer_end = std::chrono::high_resolution_clock::now();
            std::cerr << "[RindiTiming] batch_layer=" << layer
                      << " core_ms=" << std::chrono::duration<double, std::milli>(
                             core_end - layer_start).count()
                      << " tail_ms=" << std::chrono::duration<double, std::milli>(
                             layer_end - core_end).count() << std::endl;
        }
    }
    output = std::move(hidden);
    return true;
}

int RindiEngine::sample_next_token(const std::vector<uint16_t>& hidden,
                                   float temperature, std::mt19937& rng) {
    if (hidden.size() != hidden_dim_) return tokenizer_.eos_token_id();
    const size_t vocab = std::min(tokenizer_.vocab_size(), size_t(248320));
    std::vector<std::pair<float, int>> top;
    top.reserve(64);
    std::vector<uint16_t> row;

    if (lm_head_gpu_ready_ && vocab > 0) {
        std::memcpy(metal_buffer_get_contents(lm_head_input_gpu_),
                    hidden.data(), hidden_dim_ * sizeof(uint16_t));
        const bool greedy = temperature <= 0.0f;
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
        if (cmd) {
            metal_dispatch_gemm_bf16(metal_ctx_, cmd, lm_head_input_gpu_,
                                     lm_head_gpu_, lm_head_logits_gpu_,
                                     1, static_cast<int>(vocab),
                                     static_cast<int>(hidden_dim_));
            // Greedy sampling fuses the argmax into the same command buffer;
            // dispatches inside one compute encoder execute in order, so this
            // removes a second commit/wait round trip per token.
            bool fused_argmax = false;
            if (greedy && lm_head_token_gpu_) {
                metal_dispatch_argmax_fp16(metal_ctx_, cmd, lm_head_logits_gpu_,
                                           lm_head_token_gpu_, 1,
                                           static_cast<int>(vocab));
                fused_argmax = true;
            }
            metal_command_buffer_commit(cmd);
            metal_command_buffer_wait(cmd);
            if (fused_argmax) {
                const int32_t token = *static_cast<const int32_t*>(
                    metal_buffer_get_contents(lm_head_token_gpu_));
                if (token >= 0 && static_cast<size_t>(token) < vocab) {
                    if (std::getenv("RINDI_DEBUG_LOGITS")) {
                        std::cerr << "[RindiDebug] Metal LM-head argmax token="
                                  << token << "=\"" << tokenizer_.decode(token)
                                  << "\"" << std::endl;
                    }
                    return token;
                }
                fused_argmax = false;  // fall through to the top-k path
            }
            const uint16_t* logits = static_cast<const uint16_t*>(
                metal_buffer_get_contents(lm_head_logits_gpu_));
            if (logits) {
                for (size_t token = 0; token < vocab; ++token) {
                    const float score = engine_half_to_float(logits[token]);
                    if (top.size() < 64) top.emplace_back(score, static_cast<int>(token));
                    else {
                        auto worst = std::min_element(top.begin(), top.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });
                        if (score > worst->first) *worst = {score, static_cast<int>(token)};
                    }
                }
                // Continue through the common top-k temperature path below.
                row.clear();
            }
        }
    }

    if (!top.empty()) {
        std::sort(top.begin(), top.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (std::getenv("RINDI_DEBUG_LOGITS")) {
            std::cerr << "[RindiDebug] Metal top logits:";
            const size_t count = std::min<size_t>(10, top.size());
            for (size_t i = 0; i < count; ++i)
                std::cerr << " " << top[i].second << "=\""
                          << tokenizer_.decode(top[i].second) << "\"@" << top[i].first;
            std::cerr << std::endl;
        }
        if (temperature <= 0.0f) return top.front().second;
        const float max_score = top.front().first;
        float denom = 0.0f;
        std::vector<float> probs(top.size());
        for (size_t i = 0; i < top.size(); ++i) {
            probs[i] = std::exp((top[i].first - max_score) / temperature);
            denom += probs[i];
        }
        std::uniform_real_distribution<float> dist(0.0f, denom);
        float draw = dist(rng);
        for (size_t i = 0; i < probs.size(); ++i) {
            draw -= probs[i];
            if (draw <= 0.0f) return top[i].second;
        }
        return top.back().second;
    }

    // Fallback for packaged checkpoints without a separate BF16 LM head or
    // when Metal allocation/dispatch is unavailable.
    const SafeTensorsLoader* head_loader = has_separate_lm_head_
        ? &lm_head_loader_ : &safetensors_;
    const char* head_name = has_separate_lm_head_
        ? "lm_head.weight" : "embed_tokens.weight";
    for (size_t token = 0; token < vocab; ++token) {
        if (!head_loader->get_row_fp16(head_name, token, row) ||
            row.size() != hidden_dim_) continue;
        float score = 0.0f;
        for (size_t i = 0; i < hidden_dim_; ++i)
            score += engine_half_to_float(hidden[i]) * engine_half_to_float(row[i]);
        if (top.size() < 64) top.emplace_back(score, static_cast<int>(token));
        else {
            auto worst = std::min_element(top.begin(), top.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            if (score > worst->first) *worst = {score, static_cast<int>(token)};
        }
    }
    if (top.empty()) return tokenizer_.eos_token_id();
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    if (std::getenv("RINDI_DEBUG_LOGITS")) {
        std::cerr << "[RindiDebug] top logits:";
        const size_t count = std::min<size_t>(10, top.size());
        for (size_t i = 0; i < count; ++i) {
            std::cerr << " " << top[i].second << "=\""
                      << tokenizer_.decode(top[i].second) << "\"@" << top[i].first;
        }
        std::cerr << std::endl;
    }
    if (temperature <= 0.0f) return top.front().second;
    const float max_score = top.front().first;
    float denom = 0.0f;
    std::vector<float> probs(top.size());
    for (size_t i = 0; i < top.size(); ++i) {
        probs[i] = std::exp((top[i].first - max_score) / temperature);
        denom += probs[i];
    }
    std::uniform_real_distribution<float> dist(0.0f, denom);
    float draw = dist(rng);
    for (size_t i = 0; i < probs.size(); ++i) {
        draw -= probs[i];
        if (draw <= 0.0f) return top[i].second;
    }
    return top.back().second;
}

bool RindiEngine::debug_compare_logits(
    const std::vector<uint16_t>& reference_input,
    const std::vector<uint16_t>& metal_input) {
    if (reference_input.size() != hidden_dim_ || metal_input.size() != hidden_dim_ ||
        !lm_head_gpu_ready_ || !lm_head_gpu_ || !lm_head_logits_gpu_ || !metal_ctx_)
        return false;
    const size_t vocab = std::min(tokenizer_.vocab_size(), size_t(248320));
    std::vector<uint16_t> reference_logits(vocab), metal_logits(vocab);
    auto run_head = [&](const std::vector<uint16_t>& input,
                        std::vector<uint16_t>& logits) -> bool {
        std::memcpy(metal_buffer_get_contents(lm_head_input_gpu_), input.data(),
                    hidden_dim_ * sizeof(uint16_t));
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
        if (!cmd) return false;
        metal_dispatch_gemm_bf16(metal_ctx_, cmd, lm_head_input_gpu_,
                                 lm_head_gpu_, lm_head_logits_gpu_, 1,
                                 static_cast<int>(vocab),
                                 static_cast<int>(hidden_dim_));
        metal_command_buffer_commit(cmd);
        metal_command_buffer_wait(cmd);
        const auto* mapped = static_cast<const uint16_t*>(
            metal_buffer_get_contents(lm_head_logits_gpu_));
        if (!mapped) return false;
        std::memcpy(logits.data(), mapped, vocab * sizeof(uint16_t));
        return true;
    };
    if (!run_head(reference_input, reference_logits) ||
        !run_head(metal_input, metal_logits)) return false;

    double max_abs = 0.0, sum_abs = 0.0;
    int reference_argmax = -1, metal_argmax = -1;
    float reference_best = -std::numeric_limits<float>::infinity();
    float metal_best = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < vocab; ++i) {
        const float reference = engine_half_to_float(reference_logits[i]);
        const float metal = engine_half_to_float(metal_logits[i]);
        const double diff = std::abs(static_cast<double>(reference) -
                                     static_cast<double>(metal));
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
        if (reference > reference_best) {
            reference_best = reference;
            reference_argmax = static_cast<int>(i);
        }
        if (metal > metal_best) {
            metal_best = metal;
            metal_argmax = static_cast<int>(i);
        }
    }
    std::cerr << "[RindiLogitCompare] max_abs=" << max_abs
              << " mean_abs=" << (vocab ? sum_abs / vocab : 0.0)
              << " ref_argmax=" << reference_argmax
              << " metal_argmax=" << metal_argmax
              << " ref_best=" << reference_best
              << " metal_best=" << metal_best
              << " validation="
              << ((max_abs <= 0.05 && (vocab ? sum_abs / vocab : 0.0) <= 0.01 &&
                   reference_argmax == metal_argmax) ? "PASS" : "FAIL")
              << std::endl;
    return true;
}

std::string RindiEngine::generate(
    const std::string& prompt_text,
    int max_tokens,
    float temperature,
    std::function<void(const std::string& token_text)> stream_cb
) {
    if (!ready_ || !scheduler_ready_ || prompt_text.empty()) return "";

    std::vector<int> prompt_tokens = tokenizer_.encode(prompt_text);
    if (std::getenv("RINDI_DEBUG_PROMPT")) {
        std::cerr << "[RindiDebug] prompt_tokens=" << prompt_tokens.size() << std::endl;
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            std::cerr << "  " << i << ": " << prompt_tokens[i] << " \""
                      << tokenizer_.decode(prompt_tokens[i]) << "\"" << std::endl;
        }
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {tokenizer_.im_start_id(), tokenizer_.im_end_id()};
    }

    std::vector<uint16_t> embedding(hidden_dim_, 0);
    std::vector<uint16_t> hidden_state(hidden_dim_, 0);
    std::vector<uint16_t> metal_hidden_state;

    auto t0 = std::chrono::high_resolution_clock::now();

    reset_scheduler();
    // APC: if the prompt strictly extends the cached prefix, roll the model
    // back to end-of-prefix state and prefill only the suffix.
    size_t prefill_offset = 0;
    apc_last_hit_ = false;
    apc_last_saved_ = 0;
    const bool apc_enabled = !std::getenv("RINDI_DISABLE_APC");
    if (apc_enabled && apc_.valid &&
        prompt_tokens.size() >= apc_.tokens.size()) {
        bool prefix_match = true;
        for (size_t i = 0; i < apc_.tokens.size(); ++i)
            if (prompt_tokens[i] != apc_.tokens[i]) { prefix_match = false; break; }
        if (prefix_match && apc_restore()) {
            if (prompt_tokens.size() == apc_.tokens.size() &&
                apc_.last_hidden.size() == hidden_dim_) {
                // Full reuse: state already reflects every prompt token; the
                // cached end-of-prefill hidden feeds decode directly.
                prefill_offset = prompt_tokens.size();
                hidden_state = apc_.last_hidden;
                apc_last_hit_ = true;
                apc_last_saved_ = prompt_tokens.size();
                if (std::getenv("RINDI_DEBUG_TIMING"))
                    std::cerr << "[APC] full hit: " << prompt_tokens.size()
                              << " tokens reused" << std::endl;
            } else if (prompt_tokens.size() > apc_.tokens.size()) {
                prefill_offset = apc_.tokens.size();
                apc_last_hit_ = true;
                apc_last_saved_ = prefill_offset;
                if (std::getenv("RINDI_DEBUG_TIMING"))
                    std::cerr << "[APC] prefix hit: reuse=" << prefill_offset
                              << " of " << prompt_tokens.size() << std::endl;
            }
        }
    }

    const auto prefill_start = std::chrono::high_resolution_clock::now();

    // Prefill prompt chunks through the 32-column ANE tails. GDN's causal
    // convolution reserves three columns for history, leaving 29 live lanes.
    // Recurrent GDN and attention state are advanced in lane order inside the
    // batched core functions, while each fused tail is evaluated once per
    // chunk instead of once per token.
    const size_t kPrefillLanes = ane_width_ - 3;   // minus GDN history columns
    for (size_t offset = prefill_offset; offset < prompt_tokens.size(); offset += kPrefillLanes) {
        const size_t lanes = std::min(kPrefillLanes, prompt_tokens.size() - offset);
        std::vector<uint16_t> batch_input(hidden_dim_ * lanes);
        std::vector<uint16_t> row;
        for (size_t lane = 0; lane < lanes; ++lane) {
            if (!safetensors_.get_embedding_row_fp16(prompt_tokens[offset + lane],
                                                     hidden_dim_, row) ||
                row.size() != hidden_dim_) return "";
            for (size_t c = 0; c < hidden_dim_; ++c)
                batch_input[c * lanes + lane] = row[c];
        }
        std::vector<uint16_t> batch_hidden;
        if (!forward_prompt_batch(batch_input, lanes, batch_hidden) ||
            batch_hidden.size() != hidden_dim_ * lanes) {
            std::cerr << "[RindiEngine] prompt batch failed at offset " << offset
                      << " lanes=" << lanes << std::endl;
            return "";
        }
        if (std::getenv("RINDI_COMPARE_METAL_TAIL")) {
            const auto& metal_batch = chain_->last_metal_batch_output();
            if (metal_batch.size() == hidden_dim_ * lanes) {
                metal_hidden_state.resize(hidden_dim_);
                for (size_t c = 0; c < hidden_dim_; ++c)
                    metal_hidden_state[c] = metal_batch[c * lanes + (lanes - 1)];
            }
        }
        hidden_state.resize(hidden_dim_);
        for (size_t c = 0; c < hidden_dim_; ++c)
            hidden_state[c] = batch_hidden[c * lanes + (lanes - 1)];
    }
    const auto prefill_end = std::chrono::high_resolution_clock::now();
    if (std::getenv("RINDI_DEBUG_TIMING")) {
        const double prefill_ms = std::chrono::duration<double, std::milli>(
            prefill_end - prefill_start).count();
        std::cerr << "[RindiTiming] prefill_tokens=" << prompt_tokens.size()
                  << " ms=" << prefill_ms
                  << " tok_s=" << (prefill_ms > 0.0
                                      ? prompt_tokens.size() * 1000.0 / prefill_ms
                                      : 0.0)
                  << std::endl;
    }
    // Cache end-of-prefill state for exact-prefix reuse by later requests.
    if (apc_enabled && prompt_tokens.size() <= 8192)
        apc_store(prompt_tokens, hidden_state);

    // Decode from the final RMSNorm and tied embedding head. The head remains
    // on the mapped safetensors file, while all transformer blocks are native.
    std::string generated_text;
    std::mt19937 rng(0x523138u);
    const auto decode_start = std::chrono::high_resolution_clock::now();
    auto first_token_time = decode_start;
    size_t generated_tokens = 0;
    bool got_first_token = false;

    const bool spec = mtp_ready_ && mtp_depth_ > 0 && temperature <= 0.0f &&
                      prompt_tokens.size() > 0;
    const bool dbg_mtp = std::getenv("RINDI_DEBUG_MTP") != nullptr;
    size_t spec_steps = 0;
    // Acceptance-aware speculation guard: EMA of accepted drafts per round.
    // When drafting is persistently net-negative (EMA below the break-even
    // bound), cool off for 7 plain rounds, then re-probe once. Verified
    // replay keeps exactness regardless of this policy.
    float mtp_ema = 1.0f;
    int mtp_rounds = 0;
    int mtp_cool = 0;
    // Default OFF (0 disables cooling): the current draft head sits near
    // break-even on this workload; opt in via RINDI_MTP_EMA_MIN (e.g. 1.25).
    const float mtp_ema_min = std::getenv("RINDI_MTP_EMA_MIN")
        ? std::atof(std::getenv("RINDI_MTP_EMA_MIN")) : 0.0f;
    const auto emit = [&](int token_id) {
        std::string token_str = tokenizer_.decode(token_id);
        if (token_str.empty()) token_str = " ";
        generated_text += token_str;
        ++generated_tokens;
        if (dbg_mtp) std::fprintf(stderr, "[MTPTRACE] emit %d\n", token_id);
        if (stream_cb) stream_cb(token_str);
    };

    auto finalize_stats = [&](const std::chrono::high_resolution_clock::time_point& end) {
        last_stats_.prompt_tokens = prompt_tokens.size();
        last_stats_.generated_tokens = generated_tokens;
        last_stats_.prefill_ms = std::chrono::duration<double, std::milli>(
            prefill_end - prefill_start).count();
        last_stats_.ttft_ms = got_first_token
            ? std::chrono::duration<double, std::milli>(first_token_time - t0).count() : 0.0;
        last_stats_.decode_ms = got_first_token
            ? std::chrono::duration<double, std::milli>(end - first_token_time).count() : 0.0;
        last_stats_.total_ms = std::chrono::duration<double, std::milli>(end - t0).count();
    };

    if (spec) {
        // ---- Speculative (MTP) greedy decode -------------------------------
        // Invariant: hidden_state is the state at the position BEFORE cur;
        // cur is the last emitted but not yet consumed token (-1 = none).
        int cur = -1;
        size_t hot_streak = 0;   // consecutive full accepts -> deeper drafts
        auto round_start = std::chrono::high_resolution_clock::now();
        std::vector<uint16_t> logits_input;
        std::vector<int> drafts;
        std::vector<std::vector<uint16_t>> draft_hs;
        std::vector<uint16_t> batch_in, batch_hidden, replay_hidden;
        DecodeSnapshot snap;

        auto sample_cur = [&]() -> bool {
            if (!apply_rms_norm(hidden_state, final_norm_, logits_input)) return false;
            cur = argmax_token(logits_input);
            return cur >= 0;
        };
        // Emit at most `allowed` tokens from a confirmed batch; returns the
        // number actually emitted so the round can stop at the token cap.
        auto emit_batch = [&](const std::vector<int>& toks, size_t allowed) {
            const size_t n = std::min(toks.size(), allowed);
            for (size_t i = 0; i < n; ++i) emit(toks[i]);
            return n;
        };

        while (generated_tokens < max_tokens) {
            if (cur < 0) {
                if (!sample_cur()) break;
                if (cur == tokenizer_.eos_token_id()) break;
                emit(cur);
                if (got_first_token == false) { first_token_time = std::chrono::high_resolution_clock::now(); got_first_token = true; }
            }

            {
                const auto ts = std::chrono::high_resolution_clock::now();
                capture_snapshot(snap);
                if (dbg_mtp) std::fprintf(stderr, "[MTPPHASE] snap_ms=%.3f\n",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - ts).count());
            }
            snap.last_hidden = hidden_state;

            drafts.clear();
            draft_hs.clear();
            // Adaptive depth: tails are lane-invariant in COST (a K-lane
            // verify costs ~= a 1-lane step on this ANE backend), so when the
            // drafter is hot, deeper drafts propose more near-free tokens.
            // Retreat on any miss. Acceptance still goes through the proven
            // re-forward replay, so drafting depth does not affect exactness.
            ++mtp_rounds;
            bool speculate = true;
            if (mtp_cool > 0) { --mtp_cool; speculate = false; }
            else if (mtp_rounds > 8 && mtp_ema < mtp_ema_min) {
                mtp_cool = 7; speculate = false;   // re-probe on the 8th round
            }
            int depth = mtp_depth_;
            if (speculate) {
                if (hot_streak >= 2) depth = std::min<int>(depth + 2, 8);
                else if (hot_streak >= 1) depth = std::min<int>(depth + 1, 8);
            }
            int dtok = cur;
            std::vector<uint16_t> dh = hidden_state;
            std::vector<uint16_t> ho;
            for (int i = 0; speculate && i < depth; ++i) {
                int t = -1;
                if (!mtp_.draft(dtok, dh, ho, t)) break;
                if (t == tokenizer_.eos_token_id()) break;
                drafts.push_back(t);
                draft_hs.push_back(ho);
                dtok = t;
            }
            ++spec_steps;
            if (dbg_mtp) {
                const auto now = std::chrono::high_resolution_clock::now();
                std::fprintf(stderr, "[MTPPHASE] draft_ms=%.3f\n",
                    std::chrono::duration<double, std::milli>(now - round_start).count());
                round_start = now;
            }

            if (drafts.empty()) {
                // Nothing to verify: consume cur through the target stack,
                // keeping the draft cache aligned with every confirmed token.
                mtp_.advance(cur, snap.last_hidden);
                if (!safetensors_.get_embedding_row_fp16(cur, hidden_dim_, embedding) ||
                    !forward_token(embedding, hidden_state)) break;
                cur = -1;   // next round samples fresh
                continue;
            }

            const size_t lanes = drafts.size() + 1;
            batch_in.resize(hidden_dim_ * lanes);
            {
                std::vector<int> seq;
                seq.reserve(lanes);
                seq.push_back(cur);
                for (int d : drafts) seq.push_back(d);
                std::vector<uint16_t> row;
                for (size_t lane = 0; lane < lanes; ++lane) {
                    if (!safetensors_.get_embedding_row_fp16(seq[lane], hidden_dim_, row) ||
                        row.size() != hidden_dim_) {
                        finalize_stats(std::chrono::high_resolution_clock::now());
                        return generated_text;
                    }
                    for (size_t c = 0; c < hidden_dim_; ++c)
                        batch_in[c * lanes + lane] = row[c];
                }
            }
            if (!forward_prompt_batch(batch_in, lanes, batch_hidden) ||
                batch_hidden.size() != hidden_dim_ * lanes) {
                finalize_stats(std::chrono::high_resolution_clock::now());
                return generated_text;
            }
            if (dbg_mtp) {
                const auto now2 = std::chrono::high_resolution_clock::now();
                std::fprintf(stderr, "[MTPPHASE] verify_ms=%.3f\n",
                    std::chrono::duration<double, std::milli>(now2 - round_start).count());
                round_start = now2;
            }
            std::vector<int> preds;
            if (!argmax_over_hidden(batch_hidden, lanes, preds)) {
                finalize_stats(std::chrono::high_resolution_clock::now());
                return generated_text;
            }

            size_t n_ok = 0;
            while (n_ok < drafts.size() && preds[n_ok] == drafts[n_ok]) ++n_ok;
            mtp_ema = 0.75f * mtp_ema + 0.25f * (float)n_ok;
            if (dbg_mtp) {
                std::fprintf(stderr, "[MTPTRACE] round cur=%d drafts=", cur);
                for (int d : drafts) std::fprintf(stderr, "%d,", d);
                std::fprintf(stderr, " preds=");
                for (int q : preds) std::fprintf(stderr, "%d,", q);
                std::fprintf(stderr, " n_ok=%zu\n", n_ok);
            }

            if (n_ok == drafts.size()) {
                ++hot_streak;
                // Full acceptance: keep every draft plus the bonus token.
                // Slot the final accepted draft into the draft cache (the
                // drafting loop only consumed cur..d_{k-1}), paired with the
                // state at the position BEFORE it: lane L-2 of this verify
                // batch, which is d_{k-1}'s position for k >= 1, or cur's own
                // prior state when k == 1.
                {
                    const uint16_t* prev_base = lanes >= 2
                        ? batch_hidden.data() + (lanes - 2) * hidden_dim_
                        : snap.last_hidden.data();
                    std::vector<uint16_t> prev_lane(prev_base,
                                                    prev_base + hidden_dim_);
                    mtp_.advance(drafts.back(), prev_lane);
                }
                const int bonus = preds[lanes - 1];
                hidden_state.assign(batch_hidden.end() - hidden_dim_,
                                    batch_hidden.end());
                std::vector<int> confirmed(drafts);
                if (bonus != tokenizer_.eos_token_id()) confirmed.push_back(bonus);
                const size_t allowed = max_tokens - generated_tokens;
                const size_t took = emit_batch(confirmed, allowed);
                cur = confirmed[took - 1];
                if (took < confirmed.size() || bonus == tokenizer_.eos_token_id()) {
                    if (bonus == tokenizer_.eos_token_id() && took == confirmed.size())
                        cur = -1;
                    break;
                }
            } else {
                hot_streak = 0;   // a miss retreats the next round's draft depth
                // Partial: rewind, replay the proven prefix, take the fix.
                {
                    const auto tr = std::chrono::high_resolution_clock::now();
                    restore_snapshot(snap);
                    if (dbg_mtp) std::fprintf(stderr, "[MTPPHASE] restore_ms=%.3f\n",
                        std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - tr).count());
                }
                const size_t keep_len = n_ok + 1;   // [cur] + accepted drafts
                replay_hidden.clear();
                batch_in.resize(hidden_dim_ * keep_len);
                {
                    std::vector<int> seq;
                    seq.push_back(cur);
                    for (size_t i = 0; i < n_ok; ++i) seq.push_back(drafts[i]);
                    std::vector<uint16_t> row;
                    for (size_t lane = 0; lane < keep_len; ++lane) {
                        if (!safetensors_.get_embedding_row_fp16(seq[lane], hidden_dim_, row)) {
                            finalize_stats(std::chrono::high_resolution_clock::now());
                            return generated_text;
                        }
                        for (size_t c = 0; c < hidden_dim_; ++c)
                            batch_in[c * keep_len + lane] = row[c];
                    }
                }
                if (!forward_prompt_batch(batch_in, keep_len, replay_hidden) ||
                    replay_hidden.size() != hidden_dim_ * keep_len) {
                    finalize_stats(std::chrono::high_resolution_clock::now());
                    return generated_text;
                }
                hidden_state.assign(
                    replay_hidden.end() - hidden_dim_, replay_hidden.end());
                const int fix = preds[n_ok];
                std::vector<int> confirmed(drafts.begin(),
                                           drafts.begin() + n_ok);
                if (fix != tokenizer_.eos_token_id()) confirmed.push_back(fix);
                const size_t allowed = max_tokens - generated_tokens;
                const size_t took = emit_batch(confirmed, allowed);
                if (fix == tokenizer_.eos_token_id()) {
                    mtp_.restore_kv(snap.mtp_pos, {}, {});
                    break;
                }
                if (took < confirmed.size()) break;   // token cap reached
                cur = fix;
                // Re-align the draft cache over the confirmed suffix so the
                // next round's drafts continue at the right positions. Entry
                // i pairs draft i with the state BEFORE it: d_0 follows cur
                // (snap.last_hidden), d_i follows d_{i-1} (draft_hs[i-1]).
                mtp_.restore_kv(snap.mtp_pos, {}, {});
                {
                    std::vector<uint16_t> prev = snap.last_hidden;
                    for (size_t i = 0; i < n_ok; ++i) {
                        mtp_.advance(drafts[i], prev);
                        prev = draft_hs[i];
                    }
                    mtp_.advance(fix, hidden_state);
                }
            }
        }
        last_eval_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        finalize_stats(std::chrono::high_resolution_clock::now());
        last_stats_.spec_used = true;
        last_stats_.spec_steps = spec_steps;
        last_stats_.accepted_per_step = spec_steps > 0
            ? static_cast<double>(generated_tokens) / static_cast<double>(spec_steps)
            : 0.0;
        return generated_text;
    }

    for (int step = 0; step < max_tokens; ++step) {
        std::vector<uint16_t> logits_input;
        if (!apply_rms_norm(hidden_state, final_norm_, logits_input)) {
            finalize_stats(std::chrono::high_resolution_clock::now());
            return generated_text;
        }
        if (std::getenv("RINDI_DEBUG_HIDDEN")) {
            float raw_sq = 0.0f, norm_sq = 0.0f;
            for (size_t i = 0; i < hidden_dim_; ++i) {
                const float raw = engine_half_to_float(hidden_state[i]);
                const float norm = engine_half_to_float(logits_input[i]);
                raw_sq += raw * raw;
                norm_sq += norm * norm;
            }
            std::cerr << "[RindiDebug] hidden_rms=" << std::sqrt(raw_sq / hidden_dim_)
                      << " logits_input_rms=" << std::sqrt(norm_sq / hidden_dim_)
                      << std::endl;
        }
        if (step == 0 && !metal_hidden_state.empty() &&
            std::getenv("RINDI_COMPARE_METAL_TAIL")) {
            std::vector<uint16_t> metal_logits_input;
            if (apply_rms_norm(metal_hidden_state, final_norm_, metal_logits_input))
                debug_compare_logits(logits_input, metal_logits_input);
        }
        int next_token_id = sample_next_token(logits_input, temperature, rng);
        if (next_token_id == tokenizer_.eos_token_id()) break;
        if (std::getenv("RINDI_DEBUG_MTP"))
            std::fprintf(stderr, "[PLAINTRACE] emit %d\n", next_token_id);
        if (!got_first_token) {
            first_token_time = std::chrono::high_resolution_clock::now();
            got_first_token = true;
        }
        ++generated_tokens;
        (void)spec;

        std::string token_str = tokenizer_.decode(next_token_id);
        if (token_str.empty()) {
            token_str = " ";
        }

        generated_text += token_str;
        if (stream_cb) {
            stream_cb(token_str);
        }

        // The hidden state is only needed when another token will be sampled;
        // avoid an extra 64-layer ANE pass after the final requested token.
        if (step + 1 < max_tokens) {
            if (!safetensors_.get_embedding_row_fp16(next_token_id, hidden_dim_, embedding) ||
                !forward_token(embedding, hidden_state)) {
                finalize_stats(std::chrono::high_resolution_clock::now());
                return generated_text;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_eval_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    finalize_stats(t1);
    if (std::getenv("RINDI_DEBUG_TIMING")) {
        const double decode_ms = std::chrono::duration<double, std::milli>(
            t1 - decode_start).count();
        std::cerr << "[RindiTiming] decode_output_chars=" << generated_text.size()
                  << " ms=" << decode_ms << std::endl;
    }

    return generated_text;
}

std::string RindiEngine::chat_completion(
    const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& tools_json,
    int max_tokens,
    float temperature,
    std::function<void(const std::string& token_text)> stream_cb,
    bool enable_thinking,
    const std::string& reasoning_effort
) {
    std::string prompt = tokenizer_.apply_chat_template(
        messages, tools_json, enable_thinking, reasoning_effort);
    return generate(prompt, max_tokens, temperature, stream_cb);
}
