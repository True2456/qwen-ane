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
    chain_ = std::make_unique<RindiNativeChain>(hidden_dim_, 32);
    scheduler_ready_ = init_scheduler();
    if (!scheduler_ready_) {
        std::cerr << "[RindiEngine] Native transformer scheduler failed to initialize" << std::endl;
        return false;
    }

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
                                                        static_cast<int>(layer), 4096, 32)) {
                std::cerr << "[RindiEngine] Failed to compile attention core " << layer << std::endl;
                return false;
            }
        } else {
            gdn_layers_[layer] = std::make_unique<RindiGdnLayer>();
            if (!gdn_layers_[layer]->compile_core(chain_->ane_context(), safetensors_,
                                                  static_cast<int>(layer), 32)) {
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

    std::vector<uint16_t> next_projection;
    std::vector<uint16_t> residual;
    std::vector<uint16_t> core;
    std::vector<uint16_t> z;
    for (size_t layer = 0; layer < num_layers_; ++layer) {
        const auto layer_core_start = std::chrono::high_resolution_clock::now();
        residual = hidden;
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
                core = std::move(attended);
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
                std::vector<uint16_t> gated;
                if (!gdn_layers_[layer]->gate_core(core, z, gated)) return false;
                core = std::move(gated);
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
                // q_proj is laid out per head as [q(256), gate(256)]. Keep
                // the full projection for core_step(), which extracts q,
                // then apply the gate before the fused tail.
                std::vector<uint16_t> q(next_projection.begin(), next_projection.begin() + QG);
                std::vector<uint16_t> k(next_projection.begin() + QG,
                                        next_projection.begin() + QG + K);
                std::vector<uint16_t> v(next_projection.begin() + QG + K,
                                        next_projection.end());
                if (!attention_layers_[layer]->core_step(q, k, v, core)) {
                    std::cerr << "[RindiEngine] attention folded core failed at layer " << layer << std::endl;
                    return false;
                }
                for (size_t h = 0; h < 24; ++h) {
                    for (size_t d = 0; d < 256; ++d) {
                        const size_t c = h * 256 + d;
                        const float gate = engine_half_to_float(q[h * 512 + 256 + d]);
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
                RindiGdnProjectionOutput projected;
                projected.qkv.assign(next_projection.begin(), next_projection.begin() + QKV);
                projected.z.assign(next_projection.begin() + QKV,
                                   next_projection.begin() + QKV + Z);
                projected.beta.assign(next_projection.begin() + QKV + Z,
                                      next_projection.begin() + QKV + Z + G);
                projected.a.assign(next_projection.begin() + QKV + Z + G,
                                   next_projection.end());
                if (!gdn_layers_[layer]->core_from_projected(projected, 1, core, z)) {
                    std::cerr << "[RindiEngine] GDN folded core failed at layer " << layer << std::endl;
                    return false;
                }
                std::vector<uint16_t> gated;
                if (!gdn_layers_[layer]->gate_core(core, z, gated)) return false;
                core = std::move(gated);
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
    std::vector<uint16_t> next_projection;
    std::vector<uint16_t> residual;
    std::vector<uint16_t> core;
    std::vector<uint16_t> z;

    for (size_t layer = 0; layer < num_layers_; ++layer) {
        const auto layer_start = std::chrono::high_resolution_clock::now();
        residual = hidden;
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
                std::vector<uint16_t> q(next_projection.begin(),
                                        next_projection.begin() + QG * lanes);
                std::vector<uint16_t> k(next_projection.begin() + QG * lanes,
                                        next_projection.begin() + (QG + K) * lanes);
                std::vector<uint16_t> v(next_projection.begin() + (QG + K) * lanes,
                                        next_projection.end());
                if (!attention_layers_[layer]->core_step_batch(q, k, v, lanes, core))
                    return false;
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
                if (next_projection.size() != (QKV + Z + 2 * G) * lanes) return false;
                RindiGdnProjectionOutput projected;
                projected.qkv.assign(next_projection.begin(),
                                     next_projection.begin() + QKV * lanes);
                projected.z.assign(next_projection.begin() + QKV * lanes,
                                   next_projection.begin() + (QKV + Z) * lanes);
                projected.beta.assign(next_projection.begin() + (QKV + Z) * lanes,
                                      next_projection.begin() + (QKV + Z + G) * lanes);
                projected.a.assign(next_projection.begin() + (QKV + Z + G) * lanes,
                                   next_projection.end());
                std::vector<uint16_t> gated;
                if (!gdn_layers_[layer]->core_from_projected(projected, lanes, core, z) ||
                    !gdn_layers_[layer]->gate_core_batch(core, z, lanes, gated))
                    return false;
                core = std::move(gated);
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
        MetalCommandBufferHandle cmd = metal_command_buffer_create(metal_ctx_);
        if (cmd) {
            metal_dispatch_gemm_bf16(metal_ctx_, cmd, lm_head_input_gpu_,
                                     lm_head_gpu_, lm_head_logits_gpu_,
                                     1, static_cast<int>(vocab),
                                     static_cast<int>(hidden_dim_));
            metal_command_buffer_commit(cmd);
            metal_command_buffer_wait(cmd);
            const uint16_t* logits = static_cast<const uint16_t*>(
                metal_buffer_get_contents(lm_head_logits_gpu_));
            if (logits) {
                if (temperature <= 0.0f) {
                    MetalCommandBufferHandle argmax_cmd = metal_command_buffer_create(metal_ctx_);
                    if (argmax_cmd) {
                        metal_dispatch_argmax_fp16(
                            metal_ctx_, argmax_cmd, lm_head_logits_gpu_,
                            lm_head_token_gpu_, 1, static_cast<int>(vocab));
                        metal_command_buffer_commit(argmax_cmd);
                        metal_command_buffer_wait(argmax_cmd);
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
                    }
                }
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
    const auto prefill_start = std::chrono::high_resolution_clock::now();

    // Prefill prompt chunks through the 32-column ANE tails. GDN's causal
    // convolution reserves three columns for history, leaving 29 live lanes.
    // Recurrent GDN and attention state are advanced in lane order inside the
    // batched core functions, while each fused tail is evaluated once per
    // chunk instead of once per token.
    constexpr size_t kPrefillLanes = 29;
    for (size_t offset = 0; offset < prompt_tokens.size(); offset += kPrefillLanes) {
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
            batch_hidden.size() != hidden_dim_ * lanes) return "";
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

    // Decode from the final RMSNorm and tied embedding head. The head remains
    // on the mapped safetensors file, while all transformer blocks are native.
    std::string generated_text;
    std::mt19937 rng(0x523138u);
    const auto decode_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_tokens; ++step) {
        std::vector<uint16_t> logits_input;
        if (!apply_rms_norm(hidden_state, final_norm_, logits_input)) return generated_text;
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
                !forward_token(embedding, hidden_state)) return generated_text;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    last_eval_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
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
