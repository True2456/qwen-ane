// SPDX-License-Identifier: Apache-2.0
#ifndef RINDI_ANE_PROJECTION_H
#define RINDI_ANE_PROJECTION_H

#include "ane_c_bridge.h"
#include "metal_engine.h"
#include "safetensors_loader.h"
#include <IOSurface/IOSurfaceRef.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class RindiAneProjection {
public:
    RindiAneProjection() = default;
    ~RindiAneProjection();
    RindiAneProjection(const RindiAneProjection&) = delete;
    RindiAneProjection& operator=(const RindiAneProjection&) = delete;

    bool compile_fp16(ANEContext* ctx, const uint16_t* weights,
                      size_t input_dim, size_t output_dim, size_t width,
                      const std::string& tag = "projection");
    bool compile_int4(ANEContext* ctx, const SafeTensorsLoader& loader,
                      const std::string& tensor_name, size_t width,
                      const std::string& tag = "projection_int4");
    // Host-side quantized projection used by the full native scheduler when
    // the ANE is already occupied by the 64 fused block tails. It preserves
    // the same row-wise int4 contract without allocating another ANE model.
    bool compile_int4_host(const SafeTensorsLoader& loader,
                           const std::string& tensor_name);
    bool compile_chain_int4(MetalContext* ctx, const std::string& weights_path,
                            const std::string& scales_path, size_t input_dim,
                            size_t output_dim);
    // Direct BF16 projection on Metal (gemm_bf16) for unquantized tensors
    // such as the MTP draft block. evaluate() runs one command buffer.
    bool compile_bf16_metal(MetalContext* ctx, const SafeTensorsLoader& loader,
                            const std::string& tensor_name);
    bool bf16_ready() const { return bf16_ready_; }
    // Quantize an unquantized BF16 checkpoint tensor to the groupwise int4
    // format at load time and run it on the fast groupwise Metal kernel.
    // Per 64-column group: scale=(max-min)/15, bias=min, q in [0,15].
    bool compile_int4_from_bf16(MetalContext* ctx,
                                const SafeTensorsLoader& loader,
                                const std::string& tensor_name);
    bool compile_int4_from_bf16_rows(MetalContext* ctx, const void* data,
                                     size_t rows, size_t cols);
    // Row-concatenated variant: quantizes each named tensor independently and
    // packs the rows into one projection (e.g. fused q/k/v or gate/up).
    bool compile_int4_from_bf16_fused(
        MetalContext* ctx, const SafeTensorsLoader& loader,
        const std::vector<std::pair<std::string, size_t>>& tensors);
    bool evaluate(const uint16_t* input, size_t lanes,
                  std::vector<uint16_t>& output);
    bool metal_dispatch(MetalContext* ctx, MetalCommandBufferHandle cmd,
                        MetalBufferHandle input, MetalBufferHandle output,
                        size_t lanes, size_t input_offset = 0,
                        size_t output_offset = 0) const;
    // Lane-1 signed-rowwise Q4 projection through the shared SME2 backend.
    // Deliberately explicit: model planners select it per shape after an
    // energy/latency A/B; evaluate() and metal_dispatch() keep their behavior.
    bool sme2_evaluate_rowwise(const uint16_t* input, uint16_t* output) const;
    bool sme2_evaluate_rowwise_range(const uint16_t* input, uint16_t* output,
                                     size_t row_start, size_t row_count) const;
    bool metal_dispatch_rowwise_range(MetalContext* ctx,
                                      MetalCommandBufferHandle cmd,
                                      MetalBufferHandle input,
                                      MetalBufferHandle output,
                                      size_t row_start,
                                      size_t row_count) const;
    bool sme2_rowwise_ready() const { return metal_ready_ && metal_rowwise_; }
    bool metal_ready() const { return metal_ready_; }
    // Scoped by RindiEngine around true prompt ingestion. Keeping this
    // thread-local prevents the faster reduction order from leaking into MTP
    // verification/rebuild batches that require lane-1-identical numerics.
    static bool prefill_batch_reductions_enabled();
    static void set_prefill_batch_reductions(bool enabled);
    size_t input_dim() const { return input_dim_; }
    size_t output_dim() const { return output_dim_; }
    size_t width() const { return width_; }
    bool ready() const { return host_ready_ || bf16_ready_ || metal_ready_ || (model_ && request_); }

private:
    bool init_metal_int4(const SafeTensorsLoader& loader,
                         const std::string& tensor_name);

    ANEContext* ctx_{nullptr};
    ANEModel* model_{nullptr};
    ANERequest* request_{nullptr};
    IOSurfaceRef input_surface_{nullptr};
    IOSurfaceRef output_surface_{nullptr};
    size_t input_dim_{0};
    size_t output_dim_{0};
    size_t width_{0};
    bool host_ready_{false};
    std::vector<uint8_t> host_packed_;
    std::vector<float> host_scales_;
    const SafeTensorsLoader* host_loader_{nullptr};
    std::string host_tensor_name_;
    bool host_groupwise_{false};
    bool metal_ready_{false};
    bool metal_rowwise_{false};
    bool bf16_ready_{false};
    MetalContext* metal_ctx_{nullptr};
    MetalBufferHandle metal_weights_{nullptr};
    MetalBufferHandle metal_scales_{nullptr};
    MetalBufferHandle metal_biases_{nullptr};
    MetalBufferHandle metal_input_{nullptr};
    MetalBufferHandle metal_output_{nullptr};
    MetalBufferHandle metal_qmv_partials_{nullptr};
    size_t metal_packed_cols_{0};
    size_t metal_groups_{0};
};

#endif
