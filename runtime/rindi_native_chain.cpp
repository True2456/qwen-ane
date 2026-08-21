/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_native_chain.cpp - Implementation of Pure C++ 64-Layer ANE Execution Chain.
 */

#include "rindi_native_chain.h"
#include <iostream>
#include <cstring>

RindiNativeChain::RindiNativeChain(size_t hidden_dim, size_t seq_len)
    : hidden_dim_(hidden_dim),
      seq_len_(seq_len),
      surface_bytes_(seq_len * hidden_dim * sizeof(uint16_t)),
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
    }
    layers_.clear();
    
    if (surf_a_) CFRelease(surf_a_);
    if (surf_b_) CFRelease(surf_b_);
    
    if (ane_ctx_) ane_context_destroy(ane_ctx_);
    if (metal_ctx_) metal_context_destroy(metal_ctx_);
}

bool RindiNativeChain::load_layer(int layer_idx, const std::string& package_path) {
    if (!ane_ctx_) return false;
    
    ANEModel* model = ane_model_load_compiled(ane_ctx_, package_path.c_str(), "q38_layer", 0);
    if (!model) {
        std::cerr << "Failed to load compiled ANE model for layer " << layer_idx << " from " << package_path << std::endl;
        return false;
    }
    
    ANERequest* req_ab = ane_request_create(ane_ctx_, model, surf_a_, surf_b_, 0);
    ANERequest* req_ba = ane_request_create(ane_ctx_, model, surf_b_, surf_a_, 0);
    
    if (layer_idx >= (int)layers_.size()) {
        layers_.resize(layer_idx + 1, {nullptr, nullptr, nullptr});
    }
    
    layers_[layer_idx] = {model, req_ab, req_ba};
    return true;
}

bool RindiNativeChain::evaluate_step(const void* input_fp16, void* output_fp16) {
    if (layers_.empty()) return false;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // 1. Copy input embeddings to initial IOSurface (surf_a_)
    IOSurfaceLock(surf_a_, 0, NULL);
    void* base_a = IOSurfaceGetBaseAddress(surf_a_);
    std::memcpy(base_a, input_fp16, surface_bytes_);
    IOSurfaceUnlock(surf_a_, 0, NULL);
    
    // 2. Sequential 64-Layer ANE Execution via pre-created requests (0 dispatch overhead)
    bool use_a_as_input = true;
    for (size_t l = 0; l < layers_.size(); l++) {
        const auto& entry = layers_[l];
        if (!entry.model) continue;
        
        ANERequest* req = use_a_as_input ? entry.req_a_to_b : entry.req_b_to_a;
        if (!ane_request_evaluate(ane_ctx_, entry.model, req, NULL, 0, NULL, 0)) {
            std::cerr << "Evaluation failed at layer " << l << std::endl;
            return false;
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
