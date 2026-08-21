/*
 * SPDX-License-Identifier: Apache-2.0
 * shaders.metal - Metal compute shaders for high-throughput LLM operations and ANE layout bridging.
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RMSNorm FP16 with FP32 Accumulation
// ============================================================================
kernel void rmsnorm_fp16(
    device const half*  in           [[buffer(0)]],
    device const half*  weight       [[buffer(1)]],
    device half*        out          [[buffer(2)]],
    constant uint&      dim          [[buffer(3)]],
    constant float&     eps          [[buffer(4)]],
    uint2               pos          [[thread_position_in_grid]],
    uint                tid          [[thread_index_in_threadgroup]],
    uint                simd_lane_id [[thread_index_in_simdgroup]],
    uint                simd_group_id[[simdgroup_index_in_threadgroup]],
    threadgroup float*  shared_sum   [[threadgroup(0)]]
) {
    uint seq_idx = pos.y;
    device const half* in_row = in + seq_idx * dim;
    device half* out_row = out + seq_idx * dim;

    // Accumulate sum of squares in FP32
    float local_sq = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        float val = float(in_row[i]);
        local_sq += val * val;
    }

    // SIMD reduction
    local_sq = simd_sum(local_sq);

    if (simd_lane_id == 0) {
        shared_sum[simd_group_id] = local_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total_sq = 0.0f;
        for (uint g = 0; g < 8; ++g) {
            total_sq += shared_sum[g];
        }
        shared_sum[0] = rsqrt(total_sq / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_sum[0];

    // Apply normalization and scale
    for (uint i = tid; i < dim; i += 256) {
        float normalized = float(in_row[i]) * inv_rms;
        float scaled = normalized * float(weight[i]);
        out_row[i] = half(scaled);
    }
}

// ============================================================================
// Layout Conversion: Linear FP16 [S, C] -> ANE Spatial FP16 [1, C, 1, S]
// ============================================================================
kernel void layout_linear_to_ane_fp16(
    device const half* in_linear [[buffer(0)]], // [S, C]
    device half*       out_ane   [[buffer(1)]], // [1, C, 1, S]
    constant uint&     C         [[buffer(2)]],
    constant uint&     S         [[buffer(3)]],
    uint2              pos       [[thread_position_in_grid]]
) {
    uint c = pos.x;
    uint s = pos.y;
    if (c >= C || s >= S) return;

    out_ane[c * S + s] = in_linear[s * C + c];
}

// ============================================================================
// Layout Conversion: ANE Spatial FP16 [1, C, 1, S] -> Linear FP16 [S, C]
// ============================================================================
kernel void layout_ane_to_linear_fp16(
    device const half* in_ane     [[buffer(0)]], // [1, C, 1, S]
    device half*       out_linear [[buffer(1)]], // [S, C]
    constant uint&     C          [[buffer(2)]],
    constant uint&     S          [[buffer(3)]],
    uint2              pos        [[thread_position_in_grid]]
) {
    uint c = pos.x;
    uint s = pos.y;
    if (c >= C || s >= S) return;

    out_linear[s * C + c] = in_ane[c * S + s];
}

// ============================================================================
// Fused SwiGLU FP16: y = (gate * sigmoid(gate)) * up
// ============================================================================
kernel void fused_swiglu_fp16(
    device const half* gate       [[buffer(0)]],
    device const half* up         [[buffer(1)]],
    device half*       out        [[buffer(2)]],
    constant uint&     total_elem [[buffer(3)]],
    uint               idx        [[thread_position_in_grid]]
) {
    if (idx >= total_elem) return;

    float g = float(gate[idx]);
    float u = float(up[idx]);
    float silu_g = g / (1.0f + exp(-g));
    out[idx] = half(silu_g * u);
}

// ============================================================================
// GEMM FP16 with FP32 Accumulation: C = A @ B.T
// A: [M, K], B: [N, K], C: [M, N]
// ============================================================================
kernel void gemm_fp16(
    device const half*  A           [[buffer(0)]], // [M, K]
    device const half*  B           [[buffer(1)]], // [N, K] (transposed weight)
    device half*        C           [[buffer(2)]], // [M, N]
    constant uint&      M           [[buffer(3)]],
    constant uint&      N           [[buffer(4)]],
    constant uint&      K           [[buffer(5)]],
    uint2               pos         [[thread_position_in_grid]]
) {
    uint col = pos.x; // [0 .. N-1]
    uint row = pos.y; // [0 .. M-1]
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    uint a_offset = row * K;
    uint b_offset = col * K;

    for (uint k = 0; k < K; ++k) {
        sum += float(A[a_offset + k]) * float(B[b_offset + k]);
    }
    C[row * N + col] = half(sum);
}

// ============================================================================
// Direct ANE Spatial Tiled 4-Bit GEMM on Metal GPU
// Reads ANE (K/64, N/64, 64, 64) 4-bit packed weights directly from 12.1 GB pool!
// ============================================================================
kernel void gemm_ane_tiled_4bit_fp16(
    device const half*  A           [[buffer(0)]], // Input: [M, K]
    device const uchar* B_ane_tiled [[buffer(1)]], // ANE 4-bit Packed Spatial Weights
    device const half*  scales      [[buffer(2)]], // Per-channel / group scales
    device half*        C           [[buffer(3)]], // Output: [M, N]
    constant uint&      M           [[buffer(4)]], // Sequence length (e.g. 10,500)
    constant uint&      N           [[buffer(5)]], // Output channels
    constant uint&      K           [[buffer(6)]], // Input channels
    uint2               pos         [[thread_position_in_grid]]
) {
    uint col = pos.x; // Output feature index [0 .. N-1]
    uint row = pos.y; // Sequence token index [0 .. M-1]
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    uint n_block = col / 64;
    uint in_n = col % 64;

    for (uint k_block = 0; k_block < K / 64; ++k_block) {
        // Compute base offset for 64x64 ANE spatial tile
        uint tile_idx = n_block * (K / 64) + k_block;
        uint tile_byte_offset = tile_idx * (64 * 64 / 2); // 4-bit packed

        for (uint in_k = 0; in_k < 64; in_k += 2) {
            uint byte_pos = tile_byte_offset + (in_k * 64 + in_n) / 2;
            uchar packed_val = B_ane_tiled[byte_pos];

            int w0 = int(packed_val & 0x0F) - 8;
            int w1 = int((packed_val >> 4) & 0x0F) - 8;

            uint global_k0 = k_block * 64 + in_k;
            uint global_k1 = global_k0 + 1;

            float s = float(scales[col]);
            sum += float(A[row * K + global_k0]) * (float(w0) * s);
            sum += float(A[row * K + global_k1]) * (float(w1) * s);
        }
    }
    C[row * N + col] = half(sum);
}

// ============================================================================
// Fast GPU Argmax for Speculative Drafting
// ============================================================================
kernel void argmax_fp16(
    device const half*  logits      [[buffer(0)]], // [B, V]
    device int*         out_tokens  [[buffer(1)]], // [B]
    constant uint&      V           [[buffer(2)]],
    constant uint&      B           [[buffer(3)]],
    uint2               pos         [[thread_position_in_grid]]
) {
    uint seq_idx = pos.x;
    if (seq_idx >= B) return;

    device const half* row = logits + seq_idx * V;
    half max_val = row[0];
    int max_idx = 0;

    for (uint i = 1; i < V; ++i) {
        half val = row[i];
        if (val > max_val) {
            max_val = val;
            max_idx = int(i);
        }
    }
    out_tokens[seq_idx] = max_idx;
}
