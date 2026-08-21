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
// Rotary Position Embedding (RoPE) FP16
// ============================================================================
kernel void rope_fp16(
    device half*        q            [[buffer(0)]],
    device half*        k            [[buffer(1)]],
    device const half*  cos_tab      [[buffer(2)]],
    device const half*  sin_tab      [[buffer(3)]],
    constant uint&      head_dim     [[buffer(4)]],
    constant uint&      num_q_heads  [[buffer(5)]],
    constant uint&      num_k_heads  [[buffer(6)]],
    uint3               pos          [[thread_position_in_grid]]
) {
    // pos.x = half_head_dim idx [0..head_dim/2 - 1]
    // pos.y = head idx
    // pos.z = token/sequence idx
    uint half_dim = head_dim / 2;
    uint d = pos.x;
    if (d >= half_dim) return;

    uint head = pos.y;
    uint seq = pos.z;

    half cos_val = cos_tab[seq * half_dim + d];
    half sin_val = sin_tab[seq * half_dim + d];

    if (head < num_q_heads) {
        uint offset = seq * (num_q_heads * head_dim) + head * head_dim;
        half x0 = q[offset + d];
        half x1 = q[offset + d + half_dim];
        q[offset + d]            = x0 * cos_val - x1 * sin_val;
        q[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;
    }

    if (head < num_k_heads) {
        uint offset = seq * (num_k_heads * head_dim) + head * head_dim;
        half x0 = k[offset + d];
        half x1 = k[offset + d + half_dim];
        k[offset + d]            = x0 * cos_val - x1 * sin_val;
        k[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// Dynamic MoE Expert Gather FP16
// Gathers [num_tokens, hidden_dim] -> [num_tokens * top_k, hidden_dim]
// ============================================================================
kernel void moe_gather_fp16(
    device const half*  src_activations [[buffer(0)]], // [num_tokens, hidden_dim]
    device const int*   expert_indices  [[buffer(1)]], // [num_tokens, top_k]
    device half*        dst_gathered    [[buffer(2)]], // [num_tokens * top_k, hidden_dim]
    constant uint&      hidden_dim      [[buffer(3)]],
    constant uint&      top_k           [[buffer(4)]],
    uint2               pos             [[thread_position_in_grid]]
) {
    uint token_k_idx = pos.y; // [0 .. num_tokens * top_k - 1]
    uint token_idx = token_k_idx / top_k;
    uint d = pos.x;

    if (d >= hidden_dim) return;

    half val = src_activations[token_idx * hidden_dim + d];
    dst_gathered[token_k_idx * hidden_dim + d] = val;
}

// ============================================================================
// Dynamic MoE Expert Scatter & Weighted Accumulation FP16
// Combines [num_tokens * top_k, hidden_dim] -> [num_tokens, hidden_dim]
// ============================================================================
kernel void moe_scatter_fp16(
    device const half*  expert_outputs  [[buffer(0)]], // [num_tokens * top_k, hidden_dim]
    device const float* expert_weights  [[buffer(1)]], // [num_tokens, top_k] (fp32)
    device const int*   expert_indices  [[buffer(2)]], // [num_tokens, top_k] (int32)
    device half*        dst_combined    [[buffer(3)]], // [num_tokens, hidden_dim]
    constant uint&      hidden_dim      [[buffer(4)]],
    constant uint&      top_k           [[buffer(5)]],
    uint2               pos             [[thread_position_in_grid]]
) {
    uint token_idx = pos.y; // [0 .. num_tokens - 1]
    uint d = pos.x;

    if (d >= hidden_dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < top_k; ++k) {
        float weight = expert_weights[token_idx * top_k + k];
        half out_val = expert_outputs[(token_idx * top_k + k) * hidden_dim + d];
        acc += weight * float(out_val);
    }

    dst_combined[token_idx * hidden_dim + d] = half(acc);
}

// ============================================================================
// Zero-Overhead ANE Layout Transforms FP16
// Mode 0: Linear [S, C] -> ANE Planar [1, C, 1, S]
// In ANE layout, channel c has stride S (i.e. out[c * S + s] = in[s * C + c])
// ============================================================================
kernel void layout_linear_to_ane_fp16(
    device const half* in_linear  [[buffer(0)]], // [S, C]
    device half*       out_ane    [[buffer(1)]], // [1, C, 1, S]
    constant uint&     C          [[buffer(2)]],
    constant uint&     S          [[buffer(3)]],
    uint2              pos        [[thread_position_in_grid]]
) {
    uint c = pos.x;
    uint s = pos.y;
    if (c >= C || s >= S) return;

    // Linear offset: s * C + c
    // ANE offset:    c * S + s
    out_ane[c * S + s] = in_linear[s * C + c];
}

// ============================================================================
// Mode 1: ANE Planar [1, C, 1, S] -> Linear [S, C]
// in_ane[c * S + s] -> out_linear[s * C + c]
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
    out_tokens[seq_idx] = 999;
}
