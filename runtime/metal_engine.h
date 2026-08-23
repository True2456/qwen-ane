/*
 * SPDX-License-Identifier: Apache-2.0
 * metal_engine.h - Ultra-low-latency Metal C runtime for heterogeneous Metal + ANE compute.
 */

#ifndef METAL_ENGINE_H
#define METAL_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <IOSurface/IOSurfaceRef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MetalContext MetalContext;
typedef void* MetalBufferHandle;
typedef void* MetalSharedEventHandle;
typedef void* MetalPipelineHandle;
typedef void* MetalCommandBufferHandle;

/**
 * Initialize a Metal Context using the system default device.
 */
MetalContext* metal_context_create(void);

/**
 * Destroy a Metal Context and release resources.
 */
void metal_context_destroy(MetalContext* ctx);

/**
 * Get device name for diagnostic reporting.
 */
const char* metal_get_device_name(MetalContext* ctx);

/**
 * Allocate a new IOSurface of specified size (in bytes) suitable for ANE + Metal zero-copy.
 */
IOSurfaceRef metal_create_iosurface(size_t nbytes);

/**
 * Wrap an existing IOSurfaceRef into an MTLBuffer with zero copying.
 */
MetalBufferHandle metal_buffer_from_iosurface(MetalContext* ctx, IOSurfaceRef surface);

/**
 * Allocate standard MTLBuffer.
 */
MetalBufferHandle metal_buffer_create(MetalContext* ctx, size_t nbytes);

/**
 * Get direct base memory address of an IOSurfaceRef.
 */
void* metal_iosurface_get_base_address(IOSurfaceRef surface);

/**
 * Lock/unlock IOSurface for CPU access if needed.
 */
void metal_iosurface_lock(IOSurfaceRef surface, uint32_t options);
void metal_iosurface_unlock(IOSurfaceRef surface, uint32_t options);

/**
 * Free/release an MTLBuffer handle.
 */
void metal_buffer_release(MetalBufferHandle buf);

/**
 * Get raw memory pointer from an MTLBuffer.
 */
void* metal_buffer_get_contents(MetalBufferHandle buf);

/**
 * Get length in bytes of an MTLBuffer.
 */
size_t metal_buffer_get_length(MetalBufferHandle buf);

/**
 * Create a new MTLSharedEvent for CPU-free hardware synchronization between GPU and ANE.
 */
MetalSharedEventHandle metal_shared_event_create(MetalContext* ctx);

/**
 * Release an MTLSharedEvent.
 */
void metal_shared_event_release(MetalSharedEventHandle event);

/**
 * Read current signaled value of an MTLSharedEvent.
 */
uint64_t metal_shared_event_get_value(MetalSharedEventHandle event);

/**
 * Set signaled value of an MTLSharedEvent from CPU.
 */
void metal_shared_event_set_value(MetalSharedEventHandle event, uint64_t value);

/**
 * Compile Metal Shading Language source code into the context library cache.
 * Returns true on success. If false, out_error will contain the compiler diagnostics.
 */
bool metal_load_library_source(MetalContext* ctx, const char* source_code, const char** out_error);

/**
 * Retrieve a compiled compute pipeline state by kernel function name.
 */
MetalPipelineHandle metal_get_pipeline(MetalContext* ctx, const char* kernel_name);

/**
 * Create a new command buffer from the context's command queue.
 */
MetalCommandBufferHandle metal_command_buffer_create(MetalContext* ctx);

/**
 * Encode a signal on an MTLSharedEvent into the command buffer.
 */
void metal_encode_signal_event(MetalCommandBufferHandle cmd_buf, MetalSharedEventHandle event, uint64_t value);

/**
 * Encode a wait on an MTLSharedEvent into the command buffer before subsequent GPU work.
 */
void metal_encode_wait_event(MetalCommandBufferHandle cmd_buf, MetalSharedEventHandle event, uint64_t value);

/**
 * Commit a command buffer for asynchronous GPU execution.
 */
void metal_command_buffer_commit(MetalCommandBufferHandle cmd_buf);

/**
 * Synchronously wait for a command buffer to complete execution.
 */
void metal_command_buffer_wait(MetalCommandBufferHandle cmd_buf);

/**
 * Check if command buffer has completed.
 */
bool metal_command_buffer_is_completed(MetalCommandBufferHandle cmd_buf);

/* ========================================================================= */
/* Specialized Accelerated Compute Dispatches                                */
/* ========================================================================= */

/**
 * Dispatches RMSNorm over input buffer: y = x / sqrt(mean(x^2) + eps) * weight.
 * Operates on FP16 data.
 */
void metal_dispatch_rmsnorm_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle out_buf,
    int S,
    int C,
    float eps
);

/**
 * Dispatches dynamic MoE expert gathering.
 * Gathers active expert activations into contiguous buffer for ANE submission.
 */
void metal_dispatch_moe_gather_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle src_activations,   /* [num_tokens, hidden_dim] */
    MetalBufferHandle expert_indices,    /* [num_tokens, top_k] int32 */
    MetalBufferHandle dst_gathered,      /* [num_tokens * top_k, hidden_dim] */
    int num_tokens,
    int top_k,
    int hidden_dim
);

/**
 * Dispatches dynamic MoE expert output scattering and weighted accumulation.
 */
void metal_dispatch_moe_scatter_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle expert_outputs,    /* [num_tokens * top_k, hidden_dim] */
    MetalBufferHandle expert_weights,    /* [num_tokens, top_k] fp32 */
    MetalBufferHandle expert_indices,    /* [num_tokens, top_k] int32 */
    MetalBufferHandle dst_combined,      /* [num_tokens, hidden_dim] */
    int num_tokens,
    int top_k,
    int hidden_dim
);

/**
 * Zero-overhead ANE Layout Packing / Unpacking.
 * Transforms between standard contiguous row-major [S, C] and ANE spatial channel [1, C, 1, S]
 * or handles channel alignment padding.
 */
void metal_dispatch_layout_transform_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle out_buf,
    int C,
    int S,
    int mode /* 0: linear to ANE [1, C, 1, S], 1: ANE to linear [S, C] */
);

/**
 * Dispatches GEMM FP16: out_buf [M, N] = in_buf [M, K] @ weight_buf.T [N, K].
 */
void metal_dispatch_gemm_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle out_buf,
    int M,
    int N,
    int K
);

/**
 * Dispatches GEMM with a BF16 weight matrix:
 * out_buf [M, N] = in_buf [M, K] @ weight_buf.T [N, K].
 * The BF16 weights are converted on the GPU during the dot product.
 */
void metal_dispatch_gemm_bf16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle out_buf,
    int M,
    int N,
    int K
);

/**
 * Groupwise-int4 matrix projection. Weights are row-major packed as eight
 * signed 4-bit values per uint32; scales/biases are BF16 per 64 columns.
 * Input/output use channel-major [feature, lane] layout.
 */
void metal_dispatch_gemm_int4_groupwise(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle bias_buf,
    MetalBufferHandle output_buf,
    int rows,
    int logical_cols,
    int packed_cols,
    int groups,
    int lanes
);

/* Batched (lanes>1) K-parallel groupwise GEMM, one threadgroup per row,
 * threads tiled [kpar, lane] sharing the W word and splitting K. Fast path
 * for prefill where lanes>1 (the naive per-thread-K kernel is ~16 ms/call). */
void metal_dispatch_gemm_int4_groupwise_batch(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle bias_buf,
    MetalBufferHandle output_buf,
    int rows,
    int logical_cols,
    int packed_cols,
    int groups,
    int lanes
);

/* Register-blocked simdgroup-MMA int4 groupwise GEMM (MLX steel-gemm
 * structure). BM=32 x BN=32 x BK=64 tiles, 128-thread threadgroups.
 * fp16-dequant rounding (~1e-3 rel) - NOT bit-exact vs the scalar kernel;
 * validated in probes/test_metal_simd_gemm.mm. 2.4x faster than batch. */
void metal_dispatch_gemm_int4_simd(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle bias_buf,
    MetalBufferHandle output_buf,
    int rows,
    int logical_cols,
    int packed_cols,
    int groups,
    int lanes
);

/* Same projection with byte offsets into the input/output channel-major
 * buffers. This lets a fused tail write several folded projections into one
 * contiguous next-layer buffer without a CPU concat. */
void metal_dispatch_gemm_int4_groupwise_offset(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    size_t input_offset,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle bias_buf,
    MetalBufferHandle output_buf,
    size_t output_offset,
    int rows,
    int logical_cols,
    int packed_cols,
    int groups,
    int lanes
);

void metal_dispatch_gemm_int4_rowwise_offset(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    size_t input_offset,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle output_buf,
    size_t output_offset,
    int rows,
    int logical_cols,
    int packed_cols,
    int lanes
);

/* Tiled variant for exact ANE chain blobs (one FP16 scale per output row). */
void metal_dispatch_gemm_int4_rowwise_tiled_offset(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    size_t input_offset,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle output_buf,
    size_t output_offset,
    int rows,
    int logical_cols,
    int packed_cols,
    int lanes
);

/* K-parallel batched rowwise GEMM, one threadgroup per output row. */
void metal_dispatch_gemm_int4_rowwise_batched(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle output_buf,
    int rows,
    int logical_cols,
    int packed_cols,
    int lanes
);

/* Register-blocked simdgroup-MMA ROWWISE int4 GEMM (signed nibbles, per-row
 * fp16 scales). Validated in probes/test_metal_rw_simd.mm: 1.7-3.3x faster
 * than gemm_int4_rowwise_batched. fp16-dequant rounding (~1e-3 rel). */
void metal_dispatch_gemm_int4_rw_simd(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,
    MetalBufferHandle output_buf,
    int rows,
    int logical_cols,
    int packed_cols,
    int lanes
);

void metal_dispatch_add_channel_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle a_buf,
    MetalBufferHandle b_buf,
    MetalBufferHandle out_buf,
    int channels,
    int lanes
);

void metal_dispatch_rmsnorm_channel_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle output_buf,
    int channels,
    int lanes,
    float eps
);

void metal_dispatch_swiglu_channel_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle gate_up_buf,
    MetalBufferHandle output_buf,
    int intermediate,
    int lanes
);

/** Advance a GDN state for a batch of causal lanes on Metal. */
void metal_dispatch_gdn_recurrence(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle state_buf,
    MetalBufferHandle decay_buf,
    MetalBufferHandle key_buf,
    MetalBufferHandle query_buf,
    MetalBufferHandle value_buf,
    MetalBufferHandle beta_buf,
    MetalBufferHandle output_buf,
    int heads,
    int key_dim,
    int value_dim,
    int lanes
);

/**
 * Dispatches GPU Argmax FP16: out_tokens [B] = argmax(logits [B, V], axis=-1).
 */
void metal_dispatch_argmax_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle logits_buf,
    MetalBufferHandle out_tokens_buf,
    int B,
    int V
);

/* ========================================================================= */
/* Attention core (scores -> softmax -> P*V) over a resident KV cache.        */
/* Layouts: q [lanes*HQ*D] fp16, k/v caches [cap*HK*D] fp16,                  */
/*          scores/probs [lanes*HQ*cap] fp32, out [lanes*HQ*D] fp16.          */
/* ========================================================================= */

void metal_dispatch_gemv_int4_groupwise(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle w_buf, MetalBufferHandle scales_buf,
    MetalBufferHandle biases_buf, MetalBufferHandle x_buf,
    MetalBufferHandle y_buf,
    uint32_t rows, uint32_t packed_cols, uint32_t K, uint32_t groups);

void metal_dispatch_attn_scores_fp16(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle q_buf, MetalBufferHandle k_cache,
    MetalBufferHandle scores_buf,
    uint32_t base_pos, uint32_t kv_len, uint32_t lanes, uint32_t cap,
    uint32_t heads_q, uint32_t heads_kv, uint32_t dim);

void metal_dispatch_attn_softmax_fp16(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle scores_buf, MetalBufferHandle probs_buf,
    uint32_t rows /* lanes*HQ */, uint32_t base_pos, uint32_t cap,
    uint32_t heads_q);

void metal_dispatch_attn_pv_fp16(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle probs_buf, MetalBufferHandle v_cache,
    MetalBufferHandle out_buf,
    uint32_t base_pos, uint32_t lanes, uint32_t cap,
    uint32_t heads_q, uint32_t heads_kv, uint32_t dim);

#ifdef __cplusplus
}
#endif

#endif /* METAL_ENGINE_H */
