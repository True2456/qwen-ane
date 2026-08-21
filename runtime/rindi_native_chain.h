/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_native_chain.h - Pure C++ 64-Layer ANE Execution Chain with Zero Dispatch Overhead.
 */

#ifndef RINDI_NATIVE_CHAIN_H
#define RINDI_NATIVE_CHAIN_H

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include "ane_c_bridge.h"
#include "metal_engine.h"

class RindiNativeChain {
public:
    RindiNativeChain(size_t hidden_dim = 5120, size_t seq_len = 32);
    ~RindiNativeChain();

    // Load compiled .hwx / package blobs for all layers
    bool load_layer(int layer_idx, const std::string& package_path);
    
    // Execute a full 64-layer forward pass on ANE
    bool evaluate_step(const void* input_fp16, void* output_fp16);

    // Fast ping-pong pointer swapping
    size_t get_num_layers() const { return layers_.size(); }
    double get_last_eval_ms() const { return last_eval_ms_; }

private:
    size_t hidden_dim_;
    size_t seq_len_;
    size_t surface_bytes_;
    
    ANEContext* ane_ctx_;
    MetalContext* metal_ctx_;
    
    IOSurfaceRef surf_a_;
    IOSurfaceRef surf_b_;
    
    struct LayerEntry {
        ANEModel* model;
        ANERequest* req_a_to_b;
        ANERequest* req_b_to_a;
    };
    
    std::vector<LayerEntry> layers_;
    double last_eval_ms_;
};

#endif // RINDI_NATIVE_CHAIN_H
