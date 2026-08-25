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
#include "rindi_ane_swift.h"
#include "metal_engine.h"
#include "rindi_ane_projection.h"
#include "safetensors_loader.h"

class RindiNativeChain {
public:
    RindiNativeChain(size_t hidden_dim = 5120, size_t seq_len = 32);
    ~RindiNativeChain();

    // Load compiled .hwx / package blobs for all layers
    bool load_layer(int layer_idx, const std::string& package_path);

    // Reconstruct one Python-exported fused tail from its quantized cache
    // blobs.  The cache stores weights, not a directly loadable Espresso
    // package, so the MIL descriptor must be rebuilt before the ANE can load
    // the content-addressed compiled artifact.
    bool compile_layer(int layer_idx, const std::string& package_path,
                       const SafeTensorsLoader& loader);
    bool compile_metal_tails(const SafeTensorsLoader& loader,
                             const std::string& package_path);

    // Execute the fused tail for one transformer block. `core` is the
    // pre-out-projection attention/GDN output and `residual` is the block
    // input. When present, `next_projection` is the folded head for the next
    // block; its slices are consumed by the scheduler.
    bool evaluate_tail(int layer_idx, const uint16_t* core, size_t core_dim,
                       const uint16_t* residual, std::vector<uint16_t>& output,
                       std::vector<uint16_t>* next_projection = nullptr);
    bool evaluate_tail_batch(int layer_idx, const uint16_t* core, size_t core_dim,
                             const uint16_t* residual, size_t lanes,
                             std::vector<uint16_t>& output,
                             std::vector<uint16_t>* next_projection = nullptr);
    
    // Execute a full 64-layer forward pass on ANE
    bool evaluate_step(const void* input_fp16, void* output_fp16);

    // Fast ping-pong pointer swapping
    size_t get_num_layers() const { return layers_.size(); }
    double get_last_eval_ms() const { return last_eval_ms_; }
    size_t get_hidden_dim() const { return hidden_dim_; }
    size_t get_seq_len() const { return seq_len_; }
    ANEContext* ane_context() const { return ane_ctx_; }
    // In compare mode, this is the last full-Metal tail output captured for
    // the most recent batch. It is used by the engine to validate final logits
    // against the ANE reference output from the same forward pass.
    const std::vector<uint16_t>& last_metal_batch_output() const {
        return last_metal_batch_output_;
    }

private:
    struct MetalTail {
        std::unique_ptr<RindiAneProjection> out_proj;
        std::unique_ptr<RindiAneProjection> gate_proj;
        std::unique_ptr<RindiAneProjection> up_proj;
        std::unique_ptr<RindiAneProjection> down_proj;
        std::vector<std::unique_ptr<RindiAneProjection>> next_proj;
        std::vector<size_t> next_offsets;
        MetalBufferHandle post_norm{nullptr};
        MetalBufferHandle input_norm{nullptr};
        size_t core_dim{0};
        size_t intermediate{0};
        size_t next_projection{0};
        bool ready{false};

        ~MetalTail() {
            if (post_norm) metal_buffer_release(post_norm);
            if (input_norm) metal_buffer_release(input_norm);
        }
    };

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
        IOSurfaceRef input_surface;
        IOSurfaceRef output_surface;
        IOSurfaceRef projection_surface;
        size_t input_channels;
        size_t projection_channels;
        size_t core_dim{0};
        size_t intermediate{0};
        size_t written_lanes{0};   // lanes currently valid in input_surface
        bool attention{false};
        std::unique_ptr<MetalTail> metal_tail;
        // CoreAI backend (macOS 27): official-runtime bundle handle. When
        // non-null, evaluate_tail_batch routes to the CoreAI path.
        void* coreai_model{nullptr};
        bool coreai_proj_queried{false};   // output-width discovery done
    };
    
    std::vector<LayerEntry> layers_;
    MetalBufferHandle metal_tail_core_{nullptr};
    MetalBufferHandle metal_tail_residual_{nullptr};
    MetalBufferHandle metal_tail_work_{nullptr};
    MetalBufferHandle metal_tail_norm_{nullptr};
    MetalBufferHandle metal_tail_ff_{nullptr};
    MetalBufferHandle metal_tail_activation_{nullptr};
    MetalBufferHandle metal_tail_next_{nullptr};
    size_t metal_tail_core_capacity_{0};
    size_t metal_tail_intermediate_{0};
    size_t metal_tail_next_capacity_{0};
    bool metal_tail_ready_{false};
    double last_eval_ms_;
    std::vector<uint16_t> last_metal_batch_output_;

    bool compile_metal_tail(int layer_idx, const SafeTensorsLoader& loader,
                            const std::string& package_path,
                            size_t core_dim, size_t intermediate,
                            size_t next_projection, bool attention);
    bool evaluate_tail_batch_metal(int layer_idx, const uint16_t* core,
                                   size_t core_dim, const uint16_t* residual,
                                   size_t lanes, std::vector<uint16_t>& output,
                                   std::vector<uint16_t>* next_projection);
    // Official CoreAI runtime backend (macOS 27). Loads exported INT4 tail
    // bundles instead of hand-written MIL; layout conversion happens here.
    bool evaluate_tail_batch_coreai(int layer_idx, const uint16_t* core,
                                    size_t core_dim, const uint16_t* residual,
                                    size_t lanes, std::vector<uint16_t>& output,
                                    std::vector<uint16_t>* next_projection);
    std::vector<uint16_t> coreai_xin_;      // token-major staging
    std::vector<uint16_t> coreai_out_;      // concatenated outputs staging
    size_t coreai_min_lanes_{16};           // lanes below this -> Metal tail
};

#endif // RINDI_NATIVE_CHAIN_H
