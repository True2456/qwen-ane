/*
 * SPDX-License-Identifier: Apache-2.0
 * metal_engine.m - High-performance Objective-C Metal engine for ANE/GPU heterogeneous compute.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurfaceRef.h>
#import <IOSurface/IOSurface.h>
#include "metal_engine.h"

// Explicit declaration for IOSurface-backed MTLBuffer creation
@protocol MTLDeviceIOSurface <MTLDevice>
- (id<MTLBuffer>)newBufferWithIOSurface:(IOSurfaceRef)iosurface;
@end

struct MetalContext {
    id<MTLDeviceIOSurface> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> defaultLibrary;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelineCache;
    char deviceName[256];
};

static const char* kDefaultShadersSource = 
"#include <metal_stdlib>\n"
"#include <metal_simdgroup_matrix>\n"
"using namespace metal;\n"
"\n"
"inline float bf16_to_float(ushort bits) {\n"
"    return as_type<float>(uint(bits) << 16);\n"
"}\n"
"\n"
"kernel void rmsnorm_fp16(\n"
"    device const half*  in           [[buffer(0)]],\n"
"    device const half*  weight       [[buffer(1)]],\n"
"    device half*        out          [[buffer(2)]],\n"
"    constant uint&      dim          [[buffer(3)]],\n"
"    constant float&     eps          [[buffer(4)]],\n"
"    uint2               pos          [[thread_position_in_grid]],\n"
"    uint                tid          [[thread_index_in_threadgroup]],\n"
"    uint                simd_lane_id [[thread_index_in_simdgroup]],\n"
"    uint                simd_group_id[[simdgroup_index_in_threadgroup]],\n"
"    threadgroup float*  shared_sum   [[threadgroup(0)]]\n"
") {\n"
"    uint seq_idx = pos.y;\n"
"    device const half* in_row = in + seq_idx * dim;\n"
"    device half* out_row = out + seq_idx * dim;\n"
"    float local_sq = 0.0f;\n"
"    for (uint i = tid; i < dim; i += 256) {\n"
"        float val = float(in_row[i]);\n"
"        local_sq += val * val;\n"
"    }\n"
"    local_sq = simd_sum(local_sq);\n"
"    if (simd_lane_id == 0) {\n"
"        shared_sum[simd_group_id] = local_sq;\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float total_sq = 0.0f;\n"
"        for (uint g = 0; g < 8; ++g) {\n"
"            total_sq += shared_sum[g];\n"
"        }\n"
"        shared_sum[0] = rsqrt(total_sq / float(dim) + eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float inv_rms = shared_sum[0];\n"
"    for (uint i = tid; i < dim; i += 256) {\n"
"        float normalized = float(in_row[i]) * inv_rms;\n"
"        float scaled = normalized * float(weight[i]);\n"
"        out_row[i] = half(scaled);\n"
"    }\n"
"}\n"
"\n"
"kernel void rope_fp16(\n"
"    device half*        q            [[buffer(0)]],\n"
"    device half*        k            [[buffer(1)]],\n"
"    device const half*  cos_tab      [[buffer(2)]],\n"
"    device const half*  sin_tab      [[buffer(3)]],\n"
"    constant uint&      head_dim     [[buffer(4)]],\n"
"    constant uint&      num_q_heads  [[buffer(5)]],\n"
"    constant uint&      num_k_heads  [[buffer(6)]],\n"
"    uint3               pos          [[thread_position_in_grid]]\n"
") {\n"
"    uint half_dim = head_dim / 2;\n"
"    uint d = pos.x;\n"
"    if (d >= half_dim) return;\n"
"    uint head = pos.y;\n"
"    uint seq = pos.z;\n"
"    half cos_val = cos_tab[seq * half_dim + d];\n"
"    half sin_val = sin_tab[seq * half_dim + d];\n"
"    if (head < num_q_heads) {\n"
"        uint offset = seq * (num_q_heads * head_dim) + head * head_dim;\n"
"        half x0 = q[offset + d];\n"
"        half x1 = q[offset + d + half_dim];\n"
"        q[offset + d]            = x0 * cos_val - x1 * sin_val;\n"
"        q[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;\n"
"    }\n"
"    if (head < num_k_heads) {\n"
"        uint offset = seq * (num_k_heads * head_dim) + head * head_dim;\n"
"        half x0 = k[offset + d];\n"
"        half x1 = k[offset + d + half_dim];\n"
"        k[offset + d]            = x0 * cos_val - x1 * sin_val;\n"
"        k[offset + d + half_dim] = x0 * sin_val + x1 * cos_val;\n"
"    }\n"
"}\n"
"\n"
"kernel void moe_gather_fp16(\n"
"    device const half*  src_activations [[buffer(0)]],\n"
"    device const int*   expert_indices  [[buffer(1)]],\n"
"    device half*        dst_gathered    [[buffer(2)]],\n"
"    constant uint&      hidden_dim      [[buffer(3)]],\n"
"    constant uint&      top_k           [[buffer(4)]],\n"
"    uint2               pos             [[thread_position_in_grid]]\n"
") {\n"
"    uint token_k_idx = pos.y;\n"
"    uint token_idx = token_k_idx / top_k;\n"
"    uint d = pos.x;\n"
"    if (d >= hidden_dim) return;\n"
"    half val = src_activations[token_idx * hidden_dim + d];\n"
"    dst_gathered[token_k_idx * hidden_dim + d] = val;\n"
"}\n"
"\n"
"kernel void moe_scatter_fp16(\n"
"    device const half*  expert_outputs  [[buffer(0)]],\n"
"    device const float* expert_weights  [[buffer(1)]],\n"
"    device const int*   expert_indices  [[buffer(2)]],\n"
"    device half*        dst_combined    [[buffer(3)]],\n"
"    constant uint&      hidden_dim      [[buffer(4)]],\n"
"    constant uint&      top_k           [[buffer(5)]],\n"
"    uint2               pos             [[thread_position_in_grid]]\n"
") {\n"
"    uint token_idx = pos.y;\n"
"    uint d = pos.x;\n"
"    if (d >= hidden_dim) return;\n"
"    float acc = 0.0f;\n"
"    for (uint k = 0; k < top_k; ++k) {\n"
"        float weight = expert_weights[token_idx * top_k + k];\n"
"        half out_val = expert_outputs[(token_idx * top_k + k) * hidden_dim + d];\n"
"        acc += weight * float(out_val);\n"
"    }\n"
"    dst_combined[token_idx * hidden_dim + d] = half(acc);\n"
"}\n"
"\n"
"kernel void layout_linear_to_ane_fp16(\n"
"    device const half* in_linear  [[buffer(0)]],\n"
"    device half*       out_ane    [[buffer(1)]],\n"
"    constant uint&     C          [[buffer(2)]],\n"
"    constant uint&     S          [[buffer(3)]],\n"
"    uint2              pos        [[thread_position_in_grid]]\n"
") {\n"
"    uint c = pos.x;\n"
"    uint s = pos.y;\n"
"    if (c >= C || s >= S) return;\n"
"    out_ane[c * S + s] = in_linear[s * C + c];\n"
"}\n"
"\n"
"kernel void layout_ane_to_linear_fp16(\n"
"    device const half* in_ane     [[buffer(0)]],\n"
"    device half*       out_linear [[buffer(1)]],\n"
"    constant uint&     C          [[buffer(2)]],\n"
"    constant uint&     S          [[buffer(3)]],\n"
"    uint2              pos        [[thread_position_in_grid]]\n"
") {\n"
"    uint c = pos.x;\n"
"    uint s = pos.y;\n"
"    if (c >= C || s >= S) return;\n"
"    out_linear[s * C + c] = in_ane[c * S + s];\n"
"}\n"
"\n"
"kernel void fused_swiglu_fp16(\n"
"    device const half* gate       [[buffer(0)]],\n"
"    device const half* up         [[buffer(1)]],\n"
"    device half*       out        [[buffer(2)]],\n"
"    constant uint&     total_elem [[buffer(3)]],\n"
"    uint               idx        [[thread_position_in_grid]]\n"
") {\n"
"    if (idx >= total_elem) return;\n"
"    float g = float(gate[idx]);\n"
"    float u = float(up[idx]);\n"
"    float silu_g = g / (1.0f + exp(-g));\n"
"    out[idx] = half(silu_g * u);\n"
"}\n"
"\n"
"kernel void gemm_fp16(\n"
"    device const half*  A           [[buffer(0)]],\n"
"    device const half*  B           [[buffer(1)]],\n"
"    device half*        C           [[buffer(2)]],\n"
"    constant uint&      M           [[buffer(3)]],\n"
"    constant uint&      N           [[buffer(4)]],\n"
"    constant uint&      K           [[buffer(5)]],\n"
"    uint2               pos         [[thread_position_in_grid]]\n"
") {\n"
"    uint col = pos.x;\n"
"    uint row = pos.y;\n"
"    if (row >= M || col >= N) return;\n"
"    float sum = 0.0f;\n"
"    uint a_offset = row * K;\n"
"    uint b_offset = col * K;\n"
"    for (uint k = 0; k < K; ++k) {\n"
"        sum += float(A[a_offset + k]) * float(B[b_offset + k]);\n"
"    }\n"
"    C[row * N + col] = half(sum);\n"
"}\n"
"\n"
"kernel void gemm_bf16(\n"
"    device const half* A           [[buffer(0)]],\n"
"    device const ushort* B         [[buffer(1)]],\n"
"    device half* C                 [[buffer(2)]],\n"
"    constant uint& M               [[buffer(3)]],\n"
"    constant uint& N               [[buffer(4)]],\n"
"    constant uint& K               [[buffer(5)]],\n"
"    uint2 pos                      [[thread_position_in_grid]]\n"
") {\n"
"    uint col = pos.x;\n"
"    uint row = pos.y;\n"
"    if (row >= M || col >= N) return;\n"
"    float sum = 0.0f;\n"
"    uint a_offset = row * K;\n"
"    uint b_offset = col * K;\n"
"    for (uint k = 0; k < K; ++k) {\n"
"        float b_bits = as_type<float>(uint(B[b_offset + k]) << 16);\n"
"        sum += float(A[a_offset + k]) * b_bits;\n"
"    }\n"
"    C[row * N + col] = half(sum);\n"
"}\n"
"\n"
"kernel void gemm_int4_groupwise(\n"
"    device const half* A       [[buffer(0)]],\n"
"    device const uint* W       [[buffer(1)]],\n"
"    device const ushort* S     [[buffer(2)]],\n"
"    device const ushort* Bias  [[buffer(3)]],\n"
"    device half* C             [[buffer(4)]],\n"
"    constant uint& rows        [[buffer(5)]],\n"
"    constant uint& cols        [[buffer(6)]],\n"
"    constant uint& packed_cols [[buffer(7)]],\n"
"    constant uint& groups      [[buffer(8)]],\n"
"    constant uint& lanes       [[buffer(9)]],\n"
"    uint2 pos                  [[thread_position_in_grid]]\n"
") {\n"
"    uint lane = pos.x;\n"
"    uint row = pos.y;\n"
"    if (lane >= lanes || row >= rows) return;\n"
"    float sum = 0.0f;\n"
"    uint wbase = row * packed_cols;\n"
"    uint sbase = row * groups;\n"
"    for (uint c = 0; c < cols; ++c) {\n"
"        uint word = W[wbase + (c >> 3)];\n"
"        int q = int((word >> ((c & 7u) * 4u)) & 0xfu);\n"
"        uint sbits = uint(S[sbase + (c >> 6)]) << 16;\n"
"        uint bbits = uint(Bias[sbase + (c >> 6)]) << 16;\n"
"        float scale = as_type<float>(sbits);\n"
"        float bias = as_type<float>(bbits);\n"
"        sum += float(A[c * lanes + lane]) * (float(q) * scale + bias);\n"
"    }\n"
"    C[row * lanes + lane] = half(sum);\n"
"}\n"
"\n"
"kernel void gemm_int4_rowwise(\n"
"    device const half* A       [[buffer(0)]],\n"
"    device const uchar* W      [[buffer(1)]],\n"
"    device const half* S       [[buffer(2)]],\n"
"    device half* C             [[buffer(3)]],\n"
"    constant uint& rows        [[buffer(4)]],\n"
"    constant uint& cols        [[buffer(5)]],\n"
"    constant uint& packed_cols [[buffer(6)]],\n"
"    constant uint& lanes       [[buffer(7)]],\n"
"    uint2 pos                  [[thread_position_in_grid]]\n"
") {\n"
"    uint lane = pos.x;\n"
"    uint row = pos.y;\n"
"    if (lane >= lanes || row >= rows) return;\n"
"    float sum = 0.0f;\n"
"    for (uint c = 0; c < cols; ++c) {\n"
"        uchar packed = W[row * packed_cols + (c >> 1)];\n"
"        int q = int((packed >> ((c & 1u) * 4u)) & 0xfu);\n"
"        if (q >= 8) q -= 16;\n"
"        sum += float(A[c * lanes + lane]) * float(q) * float(S[row]);\n"
"    }\n"
"    C[row * lanes + lane] = half(sum);\n"
"}\n"
"\n"
"kernel void gemm_int4_rowwise_batch(\n"
"    device const half*   A     [[buffer(0)]],   // [K, lanes] channel-major\n"
"    device const uchar*  W     [[buffer(1)]],   // [rows, packed_cols] 2 nibbles/byte\n"
"    device const half*   S     [[buffer(2)]],   // [rows] per-row scale\n"
"    device half*         C     [[buffer(3)]],   // [rows, lanes]\n"
"    constant uint& rows         [[buffer(4)]],\n"
"    constant uint& cols         [[buffer(5)]],\n"
"    constant uint& packed_cols  [[buffer(6)]],\n"
"    constant uint& lanes        [[buffer(7)]],\n"
"    uint row [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tg  [[threads_per_threadgroup]],\n"
"    threadgroup float* shared [[threadgroup(0)]]) {\n"
"    if (row >= rows) return;\n"
"    uint lane = tid % lanes;\n"
"    uint kpar = tid / lanes;\n"
"    uint lane_count = min(lanes, 32u);\n"
"    if (lane >= lane_count) { shared[tid] = 0.0f; return; }\n"
"    device const uchar* wrow = W + (size_t)row * packed_cols;\n"
"    uint nlane_tg = tg / lane_count;    // kpar slices per row\n"
"    float acc = 0.0f;\n"
"    float scale = float(S[row]);\n"
"    for (uint kp = kpar; kp < nlane_tg; kp += nlane_tg) {\n"
"        uint c2_lo = (packed_cols * kp) / nlane_tg;\n"
"        uint c2_hi = (packed_cols * (kp + 1u)) / nlane_tg;\n"
"        for (uint c = c2_lo; c < c2_hi; ++c) {\n"
"            uchar packed = wrow[c];\n"
"            int q0 = int(packed & 0x0fu); if (q0 >= 8) q0 -= 16;\n"
"            int q1 = int((packed >> 4u) & 0xfu); if (q1 >= 8) q1 -= 16;\n"
"            acc += float(A[(2*c) * lanes + lane]) * float(q0) * scale;\n"
"            acc += float(A[(2*c+1) * lanes + lane]) * float(q1) * scale;\n"
"        }\n"
"    }\n"
"    shared[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint off = nlane_tg / 2u; off > 0u; off >>= 1u) {\n"
"        if (kpar < off)\n"
"            shared[tid] += shared[(kpar + off) * lane_count + lane];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (kpar == 0u) C[row * lanes + lane] = half(shared[lane]);\n"
"}\n"
"\n"
"kernel void gemm_int4_rowwise_tiled(\n"
"    device const half* A       [[buffer(0)]],\n"
"    device const uchar* W      [[buffer(1)]],\n"
"    device const half* S       [[buffer(2)]],\n"
"    device half* C             [[buffer(3)]],\n"
"    constant uint& rows        [[buffer(4)]],\n"
"    constant uint& cols        [[buffer(5)]],\n"
"    constant uint& packed_cols [[buffer(6)]],\n"
"    constant uint& lanes       [[buffer(7)]],\n"
"    uint2 global_pos           [[thread_position_in_grid]],\n"
"    uint2 local_pos            [[thread_position_in_threadgroup]],\n"
"    uint2 group_pos            [[threadgroup_position_in_grid]],\n"
"    uint tid                   [[thread_index_in_threadgroup]],\n"
"    threadgroup half a_tile[64 * 32],\n"
"    threadgroup uchar q_tile[16 * 64],\n"
"    threadgroup half scale_tile[16]\n"
") {\n"
"    const uint tile_row = group_pos.y * 16u;\n"
"    const uint lane = global_pos.x;\n"
"    const uint row = global_pos.y;\n"
"    const bool active = lane < lanes && row < rows;\n"
"    float sum = 0.0f;\n"
"    for (uint k0 = 0; k0 < cols; k0 += 64u) {\n"
"        for (uint idx = tid; idx < 64u * 32u; idx += 512u) {\n"
"            const uint k = idx / 32u;\n"
"            const uint l = idx & 31u;\n"
"            a_tile[idx] = (k0 + k < cols && l < lanes)\n"
"                ? A[(k0 + k) * lanes + l] : half(0.0f);\n"
"        }\n"
"        for (uint idx = tid; idx < 16u * 64u; idx += 512u) {\n"
"            const uint r = idx / 64u;\n"
"            const uint k = idx & 63u;\n"
"            const uint actual_row = tile_row + r;\n"
"            const uint actual_k = k0 + k;\n"
"            uchar q = 0;\n"
"            if (actual_row < rows && actual_k < cols) {\n"
"                const uchar packed = W[actual_row * packed_cols + (actual_k >> 1)];\n"
"                q = (packed >> ((actual_k & 1u) * 4u)) & 0xfu;\n"
"            }\n"
"            q_tile[idx] = q;\n"
"        }\n"
"        if (tid < 16u) {\n"
"            const uint actual_row = tile_row + tid;\n"
"            scale_tile[tid] = actual_row < rows ? S[actual_row] : half(0.0f);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint count = min(64u, cols - k0);\n"
"        for (uint k = 0; k < count; ++k) {\n"
"            int q = int(q_tile[local_pos.y * 64u + k] & 0xfu);\n"
"            if (q >= 8) q -= 16;\n"
"            sum += float(a_tile[k * 32u + local_pos.x]) *\n"
"                   float(q) * float(scale_tile[local_pos.y]);\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (active) C[row * lanes + lane] = half(sum);\n"
"}\n"
"\n"
"kernel void add_channel_fp16(\n"
"    device const half* a [[buffer(0)]],\n"
"    device const half* b [[buffer(1)]],\n"
"    device half* out [[buffer(2)]],\n"
"    constant uint& channels [[buffer(3)]],\n"
"    constant uint& lanes [[buffer(4)]],\n"
"    uint2 pos [[thread_position_in_grid]]\n"
") {\n"
"    uint lane = pos.x;\n"
"    uint c = pos.y;\n"
"    if (lane >= lanes || c >= channels) return;\n"
"    out[c * lanes + lane] = half(float(a[c * lanes + lane]) +\n"
"                                  float(b[c * lanes + lane]));\n"
"}\n"
"\n"
"kernel void gemm_int4_groupwise_tiled(\n"
"    device const half* A       [[buffer(0)]],\n"
"    device const uint* W       [[buffer(1)]],\n"
"    device const ushort* S     [[buffer(2)]],\n"
"    device const ushort* Bias  [[buffer(3)]],\n"
"    device half* C             [[buffer(4)]],\n"
"    constant uint& rows        [[buffer(5)]],\n"
"    constant uint& cols        [[buffer(6)]],\n"
"    constant uint& packed_cols [[buffer(7)]],\n"
"    constant uint& groups      [[buffer(8)]],\n"
"    constant uint& lanes       [[buffer(9)]],\n"
"    uint2 global_pos           [[thread_position_in_grid]],\n"
"    uint2 local_pos            [[thread_position_in_threadgroup]],\n"
"    uint tid                   [[thread_index_in_threadgroup]],\n"
"    threadgroup half a_tile[64 * 32],\n"
"    threadgroup uchar q_tile[16 * 64],\n"
"    threadgroup half scale_tile[16],\n"
"    threadgroup half bias_tile[16]\n"
") {\n"
"    uint lane = global_pos.x;\n"
"    uint row = global_pos.y;\n"
"    uint local_lane = local_pos.x;\n"
"    uint local_row = local_pos.y;\n"
"    bool active = lane < lanes && row < rows;\n"
"    float sum = 0.0f;\n"
"    uint row_base = row * packed_cols;\n"
"    uint row_group = row * groups;\n"
"    for (uint k0 = 0; k0 < cols; k0 += 64) {\n"
"        for (uint idx = tid; idx < 64 * 32; idx += 512) {\n"
"            uint k = idx / 32;\n"
"            uint l = idx & 31u;\n"
"            a_tile[idx] = (k0 + k < cols && l < lanes) ? A[(k0 + k) * lanes + l] : half(0.0f);\n"
"        }\n"
"        for (uint idx = tid; idx < 16 * 64; idx += 512) {\n"
"            uint r = idx / 64;\n"
"            uint k = idx & 63u;\n"
"            uint actual_row = (global_pos.y / 16u) * 16u + r;\n"
"            uint actual_k = k0 + k;\n"
"            uchar q = 0;\n"
"            if (actual_row < rows && actual_k < cols) {\n"
"                uint word = W[actual_row * packed_cols + (actual_k >> 3)];\n"
"                q = uchar((word >> ((actual_k & 7u) * 4u)) & 0xfu);\n"
"            }\n"
"            q_tile[idx] = q;\n"
"        }\n"
"        if (tid < 16) {\n"
"            uint actual_row = (global_pos.y / 16u) * 16u + tid;\n"
"            uint group_idx = k0 >> 6;\n"
"            if (actual_row < rows && group_idx < groups) {\n"
"                scale_tile[tid] = half(bf16_to_float(S[actual_row * groups + group_idx]));\n"
"                bias_tile[tid] = half(bf16_to_float(Bias[actual_row * groups + group_idx]));\n"
"            } else {\n"
"                scale_tile[tid] = half(0.0f);\n"
"                bias_tile[tid] = half(0.0f);\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        uint count = min(64u, cols - k0);\n"
"        for (uint k = 0; k < count; ++k)\n"
"            sum += float(a_tile[k * 32 + local_lane]) *\n"
"                   (float(q_tile[local_row * 64 + k]) * float(scale_tile[local_row]) +\n"
"                    float(bias_tile[local_row]));\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (active) C[row * lanes + lane] = half(sum);\n"
"}\n"
"\n"
"kernel void rmsnorm_channel_fp16(\n"
"    device const half* input [[buffer(0)]],\n"
"    device const half* weight [[buffer(1)]],\n"
"    device half* output [[buffer(2)]],\n"
"    constant uint& channels [[buffer(3)]],\n"
"    constant uint& lanes [[buffer(4)]],\n"
"    constant float& eps [[buffer(5)]],\n"
"    uint2 pos [[thread_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint simd_lane_id [[thread_index_in_simdgroup]],\n"
"    uint simd_group_id [[simdgroup_index_in_threadgroup]],\n"
"    threadgroup float* shared_sum [[threadgroup(0)]]\n"
") {\n"
"    uint lane = pos.y;\n"
"    if (lane >= lanes) return;\n"
"    float local_sq = 0.0f;\n"
"    for (uint c = tid; c < channels; c += 256) {\n"
"        float x = float(input[c * lanes + lane]);\n"
"        local_sq += x * x;\n"
"    }\n"
"    local_sq = simd_sum(local_sq);\n"
"    if (simd_lane_id == 0) shared_sum[simd_group_id] = local_sq;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float total = 0.0f;\n"
"        for (uint g = 0; g < 8; ++g) total += shared_sum[g];\n"
"        shared_sum[0] = rsqrt(total / float(channels) + eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float inv = shared_sum[0];\n"
"    for (uint c = tid; c < channels; c += 256)\n"
"        output[c * lanes + lane] = half(float(input[c * lanes + lane]) * inv *\n"
"                                        float(weight[c]));\n"
"}\n"
"\n"
"kernel void swiglu_channel_fp16(\n"
"    device const half* gate_up [[buffer(0)]],\n"
"    device half* output [[buffer(1)]],\n"
"    constant uint& intermediate [[buffer(2)]],\n"
"    constant uint& lanes [[buffer(3)]],\n"
"    uint2 pos [[thread_position_in_grid]]\n"
") {\n"
"    uint lane = pos.x;\n"
"    uint c = pos.y;\n"
"    if (lane >= lanes || c >= intermediate) return;\n"
"    float g = float(gate_up[c * lanes + lane]);\n"
"    float u = float(gate_up[(intermediate + c) * lanes + lane]);\n"
"    output[c * lanes + lane] = half((g / (1.0f + exp(-g))) * u);\n"
"}\n"
"\n"
"kernel void gdn_recurrence(\n"
"    device half* state       [[buffer(0)]],\n"
"    device const half* decay [[buffer(1)]],\n"
"    device const half* key   [[buffer(2)]],\n"
"    device const half* query [[buffer(3)]],\n"
"    device const half* value [[buffer(4)]],\n"
"    device const half* beta  [[buffer(5)]],\n"
"    device half* output      [[buffer(6)]],\n"
"    constant uint& heads     [[buffer(7)]],\n"
"    constant uint& key_dim   [[buffer(8)]],\n"
"    constant uint& value_dim [[buffer(9)]],\n"
"    constant uint& lanes     [[buffer(10)]],\n"
"    uint2 pos                [[thread_position_in_grid]]\n"
") {\n"
"    uint v = pos.x;\n"
"    uint h = pos.y;\n"
"    if (h >= heads || v >= value_dim) return;\n"
"    uint state_head = h * key_dim * value_dim;\n"
"    uint key_head = h * key_dim * lanes;\n"
"    uint out_head = h * value_dim * lanes;\n"
"    for (uint lane = 0; lane < lanes; ++lane) {\n"
"        float dcy = float(decay[h * lanes + lane]);\n"
"        float b = float(beta[h * lanes + lane]);\n"
"        float kvm = 0.0f;\n"
"        for (uint d = 0; d < key_dim; ++d) {\n"
"            kvm += float(state[state_head + d * value_dim + v]) * dcy *\n"
"                   float(key[key_head + d * lanes + lane]);\n"
"        }\n"
"        float y = 0.0f;\n"
"        for (uint d = 0; d < key_dim; ++d) {\n"
"            uint si = state_head + d * value_dim + v;\n"
"            float s1 = float(state[si]) * dcy;\n"
"            float kval = float(key[key_head + d * lanes + lane]);\n"
"            float s2 = s1 + (float(value[out_head + v * lanes + lane]) - kvm) * b * kval;\n"
"            state[si] = half(s2);\n"
"            y += s2 * float(query[key_head + d * lanes + lane]);\n"
"        }\n"
"        output[out_head + v * lanes + lane] = half(y);\n"
"    }\n"
"}\n"
"\n"
"kernel void argmax_fp16(\n"
"    device const half*  logits      [[buffer(0)]],\n"
"    device int*         out_tokens  [[buffer(1)]],\n"
"    constant uint&      V           [[buffer(2)]],\n"
"    constant uint&      B           [[buffer(3)]],\n"
"    uint2               pos         [[thread_position_in_grid]]\n"
") {\n"
"    uint seq_idx = pos.x;\n"
"    if (seq_idx >= B) return;\n"
"    device const half* row = logits + seq_idx * V;\n"
"    float max_val = float(row[0]);\n"
"    int best_tok = 0;\n"
"    for (uint v = 1; v < V; ++v) {\n"
"        float val = float(row[v]);\n"
"        if (val > max_val) {\n"
"            max_val = val;\n"
"            best_tok = int(v);\n"
"        }\n"
"    }\n"
"    out_tokens[seq_idx] = best_tok;\n"
"}\n"
"\n"
"// ============================================================================\n"
"// Decode/prefill attention core: scores -> softmax -> P*V over a resident\n"
"// KV cache. Q rows are pre-normalized/RoPE'd by the host; K/V caches are\n"
"// [capacity, HK*D] fp16, shared by decode (lanes=1) and prefill batches.\n"
"// ============================================================================\n"
"// Lane-1 (GEMV) groupwise int4 matvec: one threadgroup per output row,\n"
"// threads stride the packed K dimension 8 nibbles at a time, then a tree\n"
"// reduction. Coalesced uint loads, weights read exactly once.\n"
"kernel void gemv_int4_groupwise(\n"
"    device const uint*   W     [[buffer(0)]],   // [rows, packed_cols]\n"
"    device const ushort* S     [[buffer(1)]],   // bf16 scales [rows*groups]\n"
"    device const ushort* B     [[buffer(2)]],   // bf16 biases\n"
"    device const half*   x     [[buffer(3)]],   // [K]\n"
"    device half*         y     [[buffer(4)]],   // [rows]\n"
"    constant uint& packed_cols [[buffer(5)]],\n"
"    constant uint& K           [[buffer(6)]],\n"
"    constant uint& groups      [[buffer(7)]],\n"
"    uint row [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tg  [[threads_per_threadgroup]],\n"
"    threadgroup float* shared [[threadgroup(0)]]) {\n"
"    device const uint* wrow = W + (size_t)row * packed_cols;\n"
"    float acc = 0.0f;\n"
"    for (uint c8 = tid; c8 < packed_cols; c8 += tg) {\n"
"        const uint word = wrow[c8];\n"
"        const uint base = c8 * 8u;\n"
"        const float s = float(as_type<float>(uint(S[row * groups + base / 64u]) << 16));\n"
"        const float b = float(as_type<float>(uint(B[row * groups + base / 64u]) << 16));\n"
"        #pragma unroll\n"
"        for (uint j = 0; j < 8u; ++j) {\n"
"            const uint q = (word >> (j * 4u)) & 0xfu;\n"
"            acc += float(x[base + j]) * (float(q) * s + b);\n"
"        }\n"
"    }\n"
"    shared[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint off = tg / 2u; off > 0u; off >>= 1u) {\n"
"        if (tid < off) shared[tid] += shared[tid + off];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (tid == 0u) y[row] = half(shared[0]);\n"
"}\n"
"\n"
"// Batched (lanes>1) groupwise int4 GEMM, K-parallel + lane-shared weight\n"
"// loads. One threadgroup per output row; threads tile (lane,kpar); each\n"
"// thread reduces a K-strided slice; tree-sum over kpar per lane. Each W row\n"
"// is read once and shared across all lanes, and K is split across threads.\n"
"kernel void gemm_int4_groupwise_batch(\n"
"    device const uint*   W     [[buffer(0)]],   // [rows, packed_cols]\n"
"    device const ushort* S     [[buffer(1)]],   // bf16 scales [rows*groups]\n"
"    device const ushort* B     [[buffer(2)]],   // bf16 biases\n"
"    device const half*   A     [[buffer(3)]],   // [K, lanes] channel-major\n"
"    device half*         C     [[buffer(4)]],   // [rows, lanes]\n"
"    constant uint& rows         [[buffer(5)]],\n"
"    constant uint& packed_cols  [[buffer(6)]],\n"
"    constant uint& groups       [[buffer(7)]],\n"
"    constant uint& lanes        [[buffer(8)]],\n"
"    uint row [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tg  [[threads_per_threadgroup]],\n"
"    threadgroup float* shared [[threadgroup(0)]]) {\n"
"    if (row >= rows) return;\n"
"    // Threads laid out as [kpar, lane] to coalesce the shared W word\n"
"    // across lanes: tid = kpar*lanes + lane, so lane is contiguous-quick.\n"
"    uint lane = tid % lanes;\n"
"    uint kpar = tid / lanes;\n"
"    uint lane_count = min(lanes, 32u);\n"
"    if (lane >= lane_count) { shared[tid] = 0.0f; return; }\n"
"    device const uint* wrow = W + (size_t)row * packed_cols;\n"
"    uint nlane_tg = tg / lane_count;    // kpar slices per threadgroup\n"
"    float acc = 0.0f;\n"
"    // Grid over K in nlane_tg equal-sized slices.\n"
"    for (uint kp = kpar; kp < nlane_tg; kp += nlane_tg) {\n"
"        uint c8_lo = (packed_cols * kp) / nlane_tg;\n"
"        uint c8_hi = (packed_cols * (kp + 1u)) / nlane_tg;\n"
"        for (uint c8 = c8_lo; c8 < c8_hi; ++c8) {\n"
"            uint word = wrow[c8];\n"
"            uint base = c8 * 8u;\n"
"            float s = float(as_type<float>(uint(S[row * groups + base / 64u]) << 16));\n"
"            float b = float(as_type<float>(uint(B[row * groups + base / 64u]) << 16));\n"
"            #pragma unroll\n"
"            for (uint j = 0; j < 8u; ++j) {\n"
"                uint q = (word >> (j * 4u)) & 0xfu;\n"
"                acc += float(A[(base + j) * lanes + lane]) * (float(q) * s + b);\n"
"            }\n"
"        }\n"
"    }\n"
"    shared[tid] = acc;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    // Tree-reduce the kpar dimension while leaving lanes distinct.\n"
"    for (uint off = nlane_tg / 2u; off > 0u; off >>= 1u) {\n"
"        if (kpar < off)\n"
"            shared[tid] += shared[(kpar + off) * lane_count + lane];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    if (kpar == 0u) C[row * lanes + lane] = half(shared[lane]);\n"
"}\n"

"\n"
"// Register-blocked int4 groupwise GEMM:\n"
"//   C[M x N] = deq(W)[M x K] @ A[K x N]\n"
"// Tile BM=32, BN=32, BK=64. Threadgroup = 128 threads = 4 simdgroups.\n"
"// Simdgroup sg owns output rows [sg*8, sg*8+8) x all 32 cols of the tile,\n"
"// held as 4 x (8x8) fp32 fragments accumulated via simdgroup MMA.\n"
"kernel void gemm_int4_simd(\n"
"    device const uint*   W     [[buffer(0)]],\n"
"    device const ushort* S     [[buffer(1)]],\n"
"    device const ushort* Bias  [[buffer(2)]],\n"
"    device const half*   A     [[buffer(3)]],\n"
"    device half*         C     [[buffer(4)]],\n"
"    constant uint& rows         [[buffer(5)]],\n"
"    constant uint& cols         [[buffer(6)]],\n"
"    constant uint& packed_cols  [[buffer(7)]],\n"
"    constant uint& groups       [[buffer(8)]],\n"
"    constant uint& lanes        [[buffer(9)]],\n"
"    uint2 tg_pos [[threadgroup_position_in_grid]],\n"
"    uint  tid    [[thread_index_in_threadgroup]],\n"
"    uint  sg_id  [[simdgroup_index_in_threadgroup]],\n"
"    threadgroup half*  w_tile  [[threadgroup(0)]],  // dequantized W tile [32][64], ld 72\n"
"    threadgroup half*  a_tile  [[threadgroup(1)]],  // staged A tile   [64][32], ld 40\n"
"    threadgroup float* c_stage [[threadgroup(2)]]   // fp32 out stage [32][32] -> fp16 C\n"
") {\n"
"    const uint c_row = tg_pos.x * 32u;\n"
"    const uint c_col = tg_pos.y * 32u;\n"
"\n"
"    simdgroup_matrix<float, 8, 8> cf0 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    simdgroup_matrix<float, 8, 8> cf1 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    simdgroup_matrix<float, 8, 8> cf2 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    simdgroup_matrix<float, 8, 8> cf3 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"\n"
"    for (uint k0 = 0u; k0 < cols; k0 += 64u) {\n"
"        // ---- Phase 1a: dequantize W block [32 x 64] into threadgroup fp16.\n"
"        //      Per-group scale/bias; out-of-range entries are exact zeros so\n"
"        //      they contribute nothing to any dot product.\n"
"        for (uint idx = tid; idx < 32u * 64u; idx += 128u) {\n"
"            uint r = idx / 64u;\n"
"            uint k = idx - r * 64u;\n"
"            float v = 0.0f;\n"
"            uint gr = c_row + r;\n"
"            uint gk = k0 + k;\n"
"            if (gr < rows && gk < cols) {\n"
"                uint g = gk >> 6;\n"
"                if (g < groups) {\n"
"                    uint word = W[gr * packed_cols + (gk >> 3)];\n"
"                    uint q = (word >> ((gk & 7u) * 4u)) & 0xfu;\n"
"                    float s = bf16_to_float(S[gr * groups + g]);\n"
"                    float b = bf16_to_float(Bias[gr * groups + g]);\n"
"                    v = float(q) * s + b;\n"
"                }\n"
"            }\n"
"            w_tile[r * 72u + k] = half(v);\n"
"        }\n"
"        // ---- Phase 1b: stage A block [64 x 32], zero-padded outside range.\n"
"        for (uint idx = tid; idx < 64u * 32u; idx += 128u) {\n"
"            uint k = idx / 32u;\n"
"            uint l = idx - k * 32u;\n"
"            half val = half(0.0f);\n"
"            if (k0 + k < cols && l < lanes)\n"
"                val = A[(k0 + k) * lanes + l];\n"
"            a_tile[k * 40u + l] = val;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"        // ---- Phase 2: register-blocked MMA. K-dim advances in 8-wide steps.\n"
"        for (uint kk = 0u; kk < 64u; kk += 8u) {\n"
"            simdgroup_matrix<half, 8, 8> At;\n"
"            simdgroup_load(At, &w_tile[(sg_id * 8u) * 72u + kk], 72ul, ulong2(0, 0), false);\n"
"            #pragma unroll\n"
"            for (uint n = 0u; n < 4u; ++n) {\n"
"                simdgroup_matrix<half, 8, 8> Bt;\n"
"                simdgroup_load(Bt, &a_tile[kk * 40u + n * 8u], 40ul, ulong2(0, 0), false);\n"
"                if (n == 0u) simdgroup_multiply_accumulate(cf0, At, Bt, cf0);\n"
"                if (n == 1u) simdgroup_multiply_accumulate(cf1, At, Bt, cf1);\n"
"                if (n == 2u) simdgroup_multiply_accumulate(cf2, At, Bt, cf2);\n"
"                if (n == 3u) simdgroup_multiply_accumulate(cf3, At, Bt, cf3);\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"\n"
"    // ---- Store: fragments -> fp32 threadgroup stage -> bounds-checked fp16 C.\n"
"    //      Zero-padded dequant/staging makes any tile overhang contribute exact\n"
"    //      zeros, so no rows/lanes divisibility is required.\n"
"    simdgroup_store(cf0, &c_stage[(sg_id * 8u) * 32u + 0u],  32ul, ulong2(0, 0), false);\n"
"    simdgroup_store(cf1, &c_stage[(sg_id * 8u) * 32u + 8u],  32ul, ulong2(0, 0), false);\n"
"    simdgroup_store(cf2, &c_stage[(sg_id * 8u) * 32u + 16u], 32ul, ulong2(0, 0), false);\n"
"    simdgroup_store(cf3, &c_stage[(sg_id * 8u) * 32u + 24u], 32ul, ulong2(0, 0), false);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint idx = tid; idx < 32u * 32u; idx += 128u) {\n"
"        uint r  = idx / 32u;\n"
"        uint l  = idx - r * 32u;\n"
"        uint gr = c_row + r;\n"
"        uint gc = c_col + l;\n"
"        if (gr < rows && gc < lanes)\n"
"            C[((size_t)gr * lanes) + gc] = half(c_stage[idx]);\n"
"    }\n"
"}\n"
"kernel void gemm_int4_rw_simd(\n"
"    device const uchar*  W     [[buffer(0)]],\n"
"    device const half*   S     [[buffer(1)]],\n"
"    device const half*   A     [[buffer(2)]],\n"
"    device half*         C     [[buffer(3)]],\n"
"    constant uint& rows         [[buffer(4)]],\n"
"    constant uint& cols         [[buffer(5)]],\n"
"    constant uint& packed_cols  [[buffer(6)]],\n"
"    constant uint& lanes        [[buffer(7)]],\n"
"    uint2 tg_pos [[threadgroup_position_in_grid]],\n"
"    uint  tid    [[thread_index_in_threadgroup]],\n"
"    uint  sg_id  [[simdgroup_index_in_threadgroup]],\n"
"    threadgroup half*  w_tile  [[threadgroup(0)]],  // [32][64] ld 72\n"
"    threadgroup half*  a_tile  [[threadgroup(1)]],  // [64][32] ld 40\n"
"    threadgroup float* c_stage [[threadgroup(2)]],  // [32][32] fp32\n"
"    threadgroup half*  scale_sh[[threadgroup(3)]]    // [32] row scales\n"
") {\n"
"    const uint c_row = tg_pos.x * 32u;\n"
"    const uint c_col = tg_pos.y * 32u;\n"
"\n"
"    // Per-row fp16 scales: loaded once, reused across all K steps.\n"
"    if (tid < 32u)\n"
"        scale_sh[tid] = (c_row + tid < rows) ? S[c_row + tid] : half(0.0f);\n"
"    // readers of scale_sh sit in every simdgroup - publish before first use\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    simdgroup_matrix<float, 8, 8> cf0 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    simdgroup_matrix<float, 8, 8> cf1 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    simdgroup_matrix<float, 8, 8> cf2 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"    simdgroup_matrix<float, 8, 8> cf3 = simdgroup_matrix<float, 8, 8>(0.0f);\n"
"\n"
"    for (uint k0 = 0u; k0 < cols; k0 += 64u) {\n"
"        // ---- Dequant W block [32 x 64]: two nibbles per byte, per-row scale.\n"
"        for (uint idx = tid; idx < 32u * 64u; idx += 128u) {\n"
"            uint r = idx / 64u;\n"
"            uint k = idx - r * 64u;\n"
"            half v = half(0.0f);\n"
"            uint gr = c_row + r;\n"
"            uint gk = k0 + k;\n"
"            if (gr < rows && gk < cols) {\n"
"                uint byte = W[gr * packed_cols + (gk >> 1)];\n"
"                int q = int((gk & 1u) ? ((byte >> 4) & 0xfu) : (byte & 0xfu));\n"
"                if (q >= 8) q -= 16;   // signed int4 nibble\n"
"                v = half(float(q) * float(scale_sh[r]));\n"
"            }\n"
"            w_tile[r * 72u + k] = v;\n"
"        }\n"
"        // ---- Stage A block [64 x 32], zero-padded.\n"
"        for (uint idx = tid; idx < 64u * 32u; idx += 128u) {\n"
"            uint k = idx / 32u;\n"
"            uint l = idx - k * 32u;\n"
"            half val = half(0.0f);\n"
"            if (k0 + k < cols && l < lanes)\n"
"                val = A[(k0 + k) * lanes + l];\n"
"            a_tile[k * 40u + l] = val;\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"        for (uint kk = 0u; kk < 64u; kk += 8u) {\n"
"            simdgroup_matrix<half, 8, 8> At;\n"
"            simdgroup_load(At, &w_tile[(sg_id * 8u) * 72u + kk], 72ul, ulong2(0, 0), false);\n"
"            #pragma unroll\n"
"            for (uint n = 0u; n < 4u; ++n) {\n"
"                simdgroup_matrix<half, 8, 8> Bt;\n"
"                simdgroup_load(Bt, &a_tile[kk * 40u + n * 8u], 40ul, ulong2(0, 0), false);\n"
"                if (n == 0u) simdgroup_multiply_accumulate(cf0, At, Bt, cf0);\n"
"                if (n == 1u) simdgroup_multiply_accumulate(cf1, At, Bt, cf1);\n"
"                if (n == 2u) simdgroup_multiply_accumulate(cf2, At, Bt, cf2);\n"
"                if (n == 3u) simdgroup_multiply_accumulate(cf3, At, Bt, cf3);\n"
"            }\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"\n"
"    // ---- Store: fp32 stage -> bounds-checked fp16 C.\n"
"    simdgroup_store(cf0, &c_stage[(sg_id * 8u) * 32u + 0u],  32ul, ulong2(0, 0), false);\n"
"    simdgroup_store(cf1, &c_stage[(sg_id * 8u) * 32u + 8u],  32ul, ulong2(0, 0), false);\n"
"    simdgroup_store(cf2, &c_stage[(sg_id * 8u) * 32u + 16u], 32ul, ulong2(0, 0), false);\n"
"    simdgroup_store(cf3, &c_stage[(sg_id * 8u) * 32u + 24u], 32ul, ulong2(0, 0), false);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint idx = tid; idx < 32u * 32u; idx += 128u) {\n"
"        uint r  = idx / 32u;\n"
"        uint l  = idx - r * 32u;\n"
"        uint gr = c_row + r;\n"
"        uint gc = c_col + l;\n"
"        if (gr < rows && gc < lanes)\n"
"            C[((size_t)gr * lanes) + gc] = half(c_stage[idx]);\n"
"    }\n"
"}\n"



"kernel void attn_scores_fp16(\n"
"    device const half*  q         [[buffer(0)]],  // [lanes*HQ*D]\n"
"    device const half*  k_cache   [[buffer(1)]],  // [cap*HK*D]\n"
"    device float*       scores    [[buffer(2)]],  // [lanes*HQ*cap]\n"
"    constant uint&      base_pos  [[buffer(3)]],\n"
"    constant uint&      kv_len    [[buffer(4)]],  // base_pos + lanes\n"
"    constant uint&      cap       [[buffer(5)]],  // cache capacity == score stride\n"
"    constant uint&      heads_q   [[buffer(6)]],\n"
"    constant uint&      heads_kv  [[buffer(7)]],\n"
"    constant uint&      dim       [[buffer(8)]],\n"
"    uint3               pos       [[thread_position_in_grid]] // x: t, y: head, z: lane\n"
") {\n"
"    uint t = pos.x;\n"
"    uint h = pos.y;\n"
"    uint lane = pos.z;\n"
"    if (t >= kv_len || t > base_pos + lane) return; // causal mask\n"
"    uint kh = h / (heads_q / heads_kv);\n"
"    device const half* qp = q + ((size_t)lane * heads_q + h) * dim;\n"
"    device const half* kp = k_cache + ((size_t)t * heads_kv + kh) * dim;\n"
"    float acc = 0.0f;\n"
"    for (uint d = 0; d < dim; ++d) acc += float(qp[d]) * float(kp[d]);\n"
"    scores[((size_t)lane * heads_q + h) * cap + t] = acc * rsqrt(float(dim));\n"
"}\n"
"\n"
"kernel void attn_softmax_fp16(\n"
"    device const float* scores   [[buffer(0)]],  // [rows*cap], rows = lanes*HQ\n"
"    device float*       probs    [[buffer(1)]],\n"
"    constant uint&      base_pos [[buffer(2)]],\n"
"    constant uint&      cap      [[buffer(3)]],\n"
"    constant uint&      heads_q  [[buffer(4)]],\n"
"    uint2               pos      [[thread_position_in_grid]],  // x: tid, y: row\n"
"    uint                tid      [[thread_index_in_threadgroup]],\n"
"    uint2               tg       [[threads_per_threadgroup]],\n"
"    threadgroup float*  shared   [[threadgroup(0)]]\n"
") {\n"
"    const uint row = pos.y;\n"
"    const uint lane = row / heads_q;\n"
"    const uint limit = base_pos + lane + 1;\n"
"    device const float* s = scores + (size_t)row * cap;\n"
"    device float* p = probs + (size_t)row * cap;\n"
"    float local_max = -1e30f;\n"
"    for (uint t = tid; t < limit; t += tg.x) {\n"
"        local_max = max(local_max, s[t]);\n"
"    }\n"
"    shared[tid] = local_max;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint off = tg.x / 2; off > 0; off >>= 1) {\n"
"        if (tid < off) shared[tid] = max(shared[tid], shared[tid + off]);\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float m = shared[0];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float local_sum = 0.0f;\n"
"    for (uint t = tid; t < limit; t += tg.x) {\n"
"        float e = exp(s[t] - m);\n"
"        p[t] = e;\n"
"        local_sum += e;\n"
"    }\n"
"    shared[tid] = local_sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint off = tg.x / 2; off > 0; off >>= 1) {\n"
"        if (tid < off) shared[tid] += shared[tid + off];\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    const float inv_total = 1.0f / shared[0];\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint t = tid; t < limit; t += tg.x) {\n"
"        p[t] *= inv_total;\n"
"    }\n"
"}\n"
"\n"
"kernel void attn_pv_fp16(\n"
"    device const float* probs   [[buffer(0)]],  // [lanes*HQ*cap]\n"
"    device const half*  v_cache [[buffer(1)]],  // [cap*HK*D]\n"
"    device half*        out     [[buffer(2)]],  // [lanes*HQ*D]\n"
"    constant uint&      base_pos [[buffer(3)]],\n"
"    constant uint&      cap     [[buffer(4)]],\n"
"    constant uint&      heads_q [[buffer(5)]],\n"
"    constant uint&      heads_kv [[buffer(6)]],\n"
"    constant uint&      dim     [[buffer(7)]],\n"
"    uint3               pos     [[thread_position_in_grid]] // x: dtile(64), y: head, z: lane\n"
") {\n"
"    const uint d0 = pos.x * 64;\n"
"    const uint h = pos.y;\n"
"    const uint lane = pos.z;\n"
"    if (d0 >= dim) return;\n"
"    const uint limit = base_pos + lane + 1;\n"
"    const uint kh = h / (heads_q / heads_kv);\n"
"    float acc[64];\n"
"    for (uint i = 0; i < 64; ++i) acc[i] = 0.0f;\n"
"    device const float* p = probs + ((size_t)lane * heads_q + h) * cap;\n"
"    for (uint t = 0; t < limit; ++t) {\n"
"        const float w = p[t];\n"
"        device const half* vp = v_cache + ((size_t)t * heads_kv + kh) * dim + d0;\n"
"        for (uint i = 0; i < 64; ++i) acc[i] += w * float(vp[i]);\n"
"    }\n"
"    device half* op = out + ((size_t)lane * heads_q + h) * dim + d0;\n"
"    for (uint i = 0; i < 64; ++i) op[i] = half(acc[i]);\n"
"}\n";

MetalContext* metal_context_create(void) {
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        [pool release];
        return NULL;
    }

    MetalContext* ctx = (MetalContext*)calloc(1, sizeof(MetalContext));
    if (!ctx) {
        [device release];
        [pool release];
        return NULL;
    }

    ctx->device = [(id<MTLDeviceIOSurface>)device retain];
    ctx->commandQueue = [[device newCommandQueue] retain];
    ctx->pipelineCache = [[NSMutableDictionary alloc] init];

    const char* name = [[device name] UTF8String];
    if (name) {
        strncpy(ctx->deviceName, name, sizeof(ctx->deviceName) - 1);
    }

    const char* error_str = NULL;
    metal_load_library_source(ctx, kDefaultShadersSource, &error_str);

    [pool release];
    return ctx;
}

void metal_context_destroy(MetalContext* ctx) {
    if (!ctx) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    if (ctx->pipelineCache) {
        [ctx->pipelineCache release];
    }
    if (ctx->defaultLibrary) {
        [ctx->defaultLibrary release];
    }
    if (ctx->commandQueue) {
        [ctx->commandQueue release];
    }
    if (ctx->device) {
        [ctx->device release];
    }
    free(ctx);
    [pool release];
}

const char* metal_get_device_name(MetalContext* ctx) {
    if (!ctx) return "Unknown";
    return ctx->deviceName;
}

IOSurfaceRef metal_create_iosurface(size_t nbytes) {
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSDictionary* properties = [NSDictionary dictionaryWithObjectsAndKeys:
        [NSNumber numberWithUnsignedLong:nbytes], (id)kIOSurfaceWidth,
        [NSNumber numberWithInt:1], (id)kIOSurfaceHeight,
        [NSNumber numberWithInt:1], (id)kIOSurfaceBytesPerElement,
        [NSNumber numberWithUnsignedLong:nbytes], (id)kIOSurfaceBytesPerRow,
        [NSNumber numberWithUnsignedLong:nbytes], (id)kIOSurfaceAllocSize,
        [NSNumber numberWithInt:0], (id)kIOSurfacePixelFormat,
        nil];
    IOSurfaceRef surf = IOSurfaceCreate((CFDictionaryRef)properties);
    [pool release];
    return surf;
}

MetalBufferHandle metal_buffer_from_iosurface(MetalContext* ctx, IOSurfaceRef surface) {
    if (!ctx || !surface) return NULL;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLBuffer> buffer = [ctx->device newBufferWithIOSurface:surface];
    [buffer retain];
    [pool release];
    return (MetalBufferHandle)buffer;
}

MetalBufferHandle metal_buffer_create(MetalContext* ctx, size_t nbytes) {
    if (!ctx || nbytes == 0) return NULL;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:nbytes
                                                    options:MTLResourceStorageModeShared];
    [buffer retain];
    [pool release];
    return (MetalBufferHandle)buffer;
}

void metal_buffer_release(MetalBufferHandle buf) {
    if (!buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLBuffer> buffer = (id<MTLBuffer>)buf;
    [buffer release];
    [pool release];
}

void* metal_iosurface_get_base_address(IOSurfaceRef surface) {
    if (!surface) return NULL;
    return IOSurfaceGetBaseAddress(surface);
}

void metal_iosurface_lock(IOSurfaceRef surface, uint32_t options) {
    if (surface) {
        IOSurfaceLock(surface, options, NULL);
    }
}

void metal_iosurface_unlock(IOSurfaceRef surface, uint32_t options) {
    if (surface) {
        IOSurfaceUnlock(surface, options, NULL);
    }
}

void* metal_buffer_get_contents(MetalBufferHandle buf) {
    if (!buf) return NULL;
    id<MTLBuffer> buffer = (id<MTLBuffer>)buf;
    void* ptr = [buffer contents];
    if (!ptr && [buffer respondsToSelector:@selector(iosurface)]) {
        IOSurfaceRef surf = [(id)buffer iosurface];
        if (surf) {
            ptr = IOSurfaceGetBaseAddress(surf);
        }
    }
    return ptr;
}

size_t metal_buffer_get_length(MetalBufferHandle buf) {
    if (!buf) return 0;
    id<MTLBuffer> buffer = (id<MTLBuffer>)buf;
    return [buffer length];
}

MetalSharedEventHandle metal_shared_event_create(MetalContext* ctx) {
    if (!ctx) return NULL;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLSharedEvent> event = [ctx->device newSharedEvent];
    [event retain];
    [pool release];
    return (MetalSharedEventHandle)event;
}

void metal_shared_event_release(MetalSharedEventHandle event) {
    if (!event) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLSharedEvent> ev = (id<MTLSharedEvent>)event;
    [ev release];
    [pool release];
}

uint64_t metal_shared_event_get_value(MetalSharedEventHandle event) {
    if (!event) return 0;
    id<MTLSharedEvent> ev = (id<MTLSharedEvent>)event;
    return [ev signaledValue];
}

void metal_shared_event_set_value(MetalSharedEventHandle event, uint64_t value) {
    if (!event) return;
    id<MTLSharedEvent> ev = (id<MTLSharedEvent>)event;
    [ev setSignaledValue:value];
}

bool metal_load_library_source(MetalContext* ctx, const char* source_code, const char** out_error) {
    if (!ctx || !source_code) return false;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:source_code];
    MTLCompileOptions* options = [[[MTLCompileOptions alloc] init] autorelease];
    options.fastMathEnabled = YES;

    id<MTLLibrary> library = [ctx->device newLibraryWithSource:src
                                                       options:options
                                                         error:&error];

    if (!library) {
        if (out_error && error) {
            *out_error = strdup([[error localizedDescription] UTF8String]);
        }
        [pool release];
        return false;
    }

    if (ctx->defaultLibrary) {
        [ctx->defaultLibrary release];
    }
    ctx->defaultLibrary = [library retain];
    [pool release];
    return true;
}

MetalPipelineHandle metal_get_pipeline(MetalContext* ctx, const char* kernel_name) {
    if (!ctx || !kernel_name) return NULL;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSString* name = [NSString stringWithUTF8String:kernel_name];
    id<MTLComputePipelineState> cached = [ctx->pipelineCache objectForKey:name];
    if (cached) {
        [pool release];
        return (MetalPipelineHandle)cached;
    }

    if (!ctx->defaultLibrary) {
        NSLog(@"[ERROR] metal_get_pipeline: defaultLibrary is NULL for kernel %@", name);
        [pool release];
        return NULL;
    }

    id<MTLFunction> func = [ctx->defaultLibrary newFunctionWithName:name];
    if (!func) {
        NSLog(@"[ERROR] metal_get_pipeline: Function not found in library: %@", name);
        [pool release];
        return NULL;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [ctx->device newComputePipelineStateWithFunction:func error:&error];
    [func release];

    if (!pipeline) {
        NSLog(@"[ERROR] metal_get_pipeline: Failed to create pipeline for %@: %@", name, error);
        [pool release];
        return NULL;
    }

    [ctx->pipelineCache setObject:pipeline forKey:name];
    [pool release];
    return (MetalPipelineHandle)pipeline;
}

MetalCommandBufferHandle metal_command_buffer_create(MetalContext* ctx) {
    if (!ctx) return NULL;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBufferWithUnretainedReferences];
    if (!cmdBuf) {
        cmdBuf = [ctx->commandQueue commandBuffer];
    }
    [cmdBuf retain];
    [pool release];
    return (MetalCommandBufferHandle)cmdBuf;
}

void metal_encode_signal_event(MetalCommandBufferHandle cmd_buf, MetalSharedEventHandle event, uint64_t value) {
    if (!cmd_buf || !event) return;
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLSharedEvent> ev = (id<MTLSharedEvent>)event;
    [cmd encodeSignalEvent:ev value:value];
}

void metal_encode_wait_event(MetalCommandBufferHandle cmd_buf, MetalSharedEventHandle event, uint64_t value) {
    if (!cmd_buf || !event) return;
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLSharedEvent> ev = (id<MTLSharedEvent>)event;
    [cmd encodeWaitForEvent:ev value:value];
}

void metal_command_buffer_commit(MetalCommandBufferHandle cmd_buf) {
    if (!cmd_buf) return;
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    [cmd commit];
}

void metal_command_buffer_wait(MetalCommandBufferHandle cmd_buf) {
    if (!cmd_buf) return;
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    [cmd waitUntilCompleted];
    [cmd release];
}

bool metal_command_buffer_is_completed(MetalCommandBufferHandle cmd_buf) {
    if (!cmd_buf) return true;
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    return [cmd status] == MTLCommandBufferStatusCompleted;
}

/* ========================================================================= */
/* Accelerated Dispatches Implementation                                     */
/* ========================================================================= */

void metal_dispatch_rmsnorm_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle out_buf,
    int S,
    int C,
    float eps
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "rmsnorm_fp16");
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)in_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buf offset:0 atIndex:2];

    uint32_t dim_val = (uint32_t)C;
    [encoder setBytes:&dim_val length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&eps length:sizeof(float) atIndex:4];

    [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];

    MTLSize threadsPerGrid = MTLSizeMake(256, S, 1);
    MTLSize threadsPerGroup = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_moe_gather_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle src_activations,
    MetalBufferHandle expert_indices,
    MetalBufferHandle dst_gathered,
    int num_tokens,
    int top_k,
    int hidden_dim
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "moe_gather_fp16");
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)src_activations offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)expert_indices offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)dst_gathered offset:0 atIndex:2];

    uint32_t hdim = (uint32_t)hidden_dim;
    uint32_t tk = (uint32_t)top_k;
    [encoder setBytes:&hdim length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&tk length:sizeof(uint32_t) atIndex:4];

    int total_gather_rows = num_tokens * top_k;
    MTLSize threadsPerGrid = MTLSizeMake(hidden_dim, total_gather_rows, 1);
    MTLSize threadsPerGroup = MTLSizeMake(MIN(hidden_dim, 256), 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_moe_scatter_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle expert_outputs,
    MetalBufferHandle expert_weights,
    MetalBufferHandle expert_indices,
    MetalBufferHandle dst_combined,
    int num_tokens,
    int top_k,
    int hidden_dim
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "moe_scatter_fp16");
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)expert_outputs offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)expert_weights offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)expert_indices offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)dst_combined offset:0 atIndex:3];

    uint32_t hdim = (uint32_t)hidden_dim;
    uint32_t tk = (uint32_t)top_k;
    [encoder setBytes:&hdim length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&tk length:sizeof(uint32_t) atIndex:5];

    MTLSize threadsPerGrid = MTLSizeMake(hidden_dim, num_tokens, 1);
    MTLSize threadsPerGroup = MTLSizeMake(MIN(hidden_dim, 256), 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_layout_transform_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle out_buf,
    int C,
    int S,
    int mode
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    const char* kernel_name = (mode == 0) ? "layout_linear_to_ane_fp16" : "layout_ane_to_linear_fp16";
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)metal_get_pipeline(ctx, kernel_name);
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)in_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)out_buf offset:0 atIndex:1];

    uint32_t c_val = (uint32_t)C;
    uint32_t s_val = (uint32_t)S;
    [encoder setBytes:&c_val length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&s_val length:sizeof(uint32_t) atIndex:3];

    MTLSize threadsPerGrid = MTLSizeMake(C, S, 1);
    MTLSize threadsPerGroup = MTLSizeMake(MIN(C, 32), MIN(S, 8), 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_gemm_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle out_buf,
    int M,
    int N,
    int K
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_fp16");
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)in_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buf offset:0 atIndex:2];

    uint32_t m_val = (uint32_t)M;
    uint32_t n_val = (uint32_t)N;
    uint32_t k_val = (uint32_t)K;
    [encoder setBytes:&m_val length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&n_val length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&k_val length:sizeof(uint32_t) atIndex:5];

    MTLSize threadsPerGrid = MTLSizeMake(N, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(MIN(N, 16), MIN(M, 16), 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_gemm_bf16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle in_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle out_buf,
    int M,
    int N,
    int K
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_bf16");
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)in_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buf offset:0 atIndex:2];
    uint32_t m_val = (uint32_t)M;
    uint32_t n_val = (uint32_t)N;
    uint32_t k_val = (uint32_t)K;
    [encoder setBytes:&m_val length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&n_val length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&k_val length:sizeof(uint32_t) atIndex:5];
    MTLSize threadsPerGrid = MTLSizeMake(N, M, 1);
    MTLSize threadsPerGroup = MTLSizeMake(MIN(N, 16), MIN(M, 16), 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}

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
) {
    metal_dispatch_gemm_int4_groupwise_offset(
        ctx, cmd_buf, input_buf, 0, weight_buf, scale_buf, bias_buf,
        output_buf, 0, rows, logical_cols, packed_cols, groups, lanes);
}

/* NEW: K-parallel batched groupwise GEMM. One threadgroup per output row;
 * threads tile [kpar, lane] so the shared W word is loaded once and kept in
 * threadgroup memory, and K is split across kpar threads (power-of-two, so
 * the per-lane tree reduction is exact). Input A is [K, lanes] channel-major,
 * output C is [rows, lanes]. This is the fast path for prefill lanes>1. */
void metal_dispatch_gemm_int4_groupwise_batch(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf, MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf, MetalBufferHandle bias_buf,
    MetalBufferHandle output_buf,
    int rows, int logical_cols, int packed_cols, int groups, int lanes) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf ||
        !bias_buf || !output_buf || rows <= 0 || logical_cols <= 0 || lanes <= 0) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_groupwise_batch");
    if (!pipeline) { [pool release]; return; }
    int lc = MIN(lanes, 32);                        // lanes handled per group
    int knar = 1;
    while (knar * 2 <= (int)(logical_cols / 64) && knar < 32) knar *= 2;  // power-of-two slices
    if (knar < 1) knar = 1;
    int tg = lc * knar;                            // threads per group
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)bias_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:0 atIndex:4];
    uint32_t values[] = {(uint32_t)rows, (uint32_t)packed_cols,
                        (uint32_t)groups, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 4; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:5 + i];
    [encoder setThreadgroupMemoryLength:tg * sizeof(float) atIndex:0];
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

/* Register-blocked int4-groupwise GEMM using simdgroup MMA (MLX steel-gemm
 * structure): BM=32 x BN=32 tiles, BK=64 == quant group width; 128-thread
 * threadgroups = 4 simdgroups, each accumulating four 8x8 fp32 fragments over
 * rows [sg*8,+8). W tile dequantized to fp16 in threadgroup memory per K-step;
 * activations staged alongside. Fully bounds-checked: any rows/lanes valid.
 * Numerics: fp16-dequant rounding (~1e-3 rel), same class as MLX quantized
 * GEMMs. Validated in probes/test_metal_simd_gemm.mm: bad=0 vs scalar ref,
 * 2.25 ms vs 5.39 ms batch on 34816x5120 lanes=32 (2.4x, 5.08 TFLOPS). */
void metal_dispatch_gemm_int4_simd(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf, MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf, MetalBufferHandle bias_buf,
    MetalBufferHandle output_buf,
    int rows, int logical_cols, int packed_cols, int groups, int lanes) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf ||
        !bias_buf || !output_buf || rows <= 0 || logical_cols <= 0 || lanes <= 0) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_simd");
    if (!pipeline) { [pool release]; return; }
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)bias_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:0 atIndex:4];
    uint32_t values[] = {(uint32_t)rows, (uint32_t)logical_cols,
                         (uint32_t)packed_cols, (uint32_t)groups, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 5; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:5 + i];
    [encoder setThreadgroupMemoryLength:32 * 72 * 2 atIndex:0];
    [encoder setThreadgroupMemoryLength:64 * 40 * 2 atIndex:1];
    [encoder setThreadgroupMemoryLength:32 * 32 * sizeof(float) atIndex:2];
    MTLSize grid = MTLSizeMake((NSUInteger)((rows + 31) / 32),
                               (NSUInteger)((lanes + 31) / 32), 1);
    MTLSize group = MTLSizeMake(128, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

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
) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf ||
        !bias_buf || !output_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_groupwise");
    if (!pipeline) {
        [pool release];
        return;
    }
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:input_offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)bias_buf offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:output_offset atIndex:4];
    uint32_t values[] = {
        (uint32_t)rows, (uint32_t)logical_cols, (uint32_t)packed_cols,
        (uint32_t)groups, (uint32_t)lanes
    };
    for (NSUInteger i = 0; i < 5; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:5 + i];
    MTLSize grid = MTLSizeMake(lanes, rows, 1);
    MTLSize group = MTLSizeMake(MIN(lanes, 32), MIN(rows, 8), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

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
) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf || !output_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_rowwise");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:input_offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:output_offset atIndex:3];
    uint32_t values[] = {(uint32_t)rows, (uint32_t)logical_cols,
                         (uint32_t)packed_cols, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 4; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:4 + i];
    MTLSize grid = MTLSizeMake(lanes, rows, 1);
    MTLSize group = MTLSizeMake(MIN(lanes, 32), MIN(rows, 8), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

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
) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf || !output_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_rowwise_tiled");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:input_offset atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:output_offset atIndex:3];
    uint32_t values[] = {(uint32_t)rows, (uint32_t)logical_cols,
                         (uint32_t)packed_cols, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 4; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:4 + i];
    MTLSize grid = MTLSizeMake((lanes + 31) / 32, (rows + 15) / 16, 1);
    MTLSize group = MTLSizeMake(32, 16, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

/* NEW: K-parallel batched rowwise GEMM, one threadgroup per output row. */
/* Register-blocked ROWWISE int4 GEMM via simdgroup MMA (MLX steel structure):
 * BM=32 x BN=32 x BK=64 tiles, 128-thread threadgroups = 4 simdgroups x four
 * 8x8 fp32 fragments. Signed-int4 nibbles (2/byte), per-row fp16 scales.
 * Validated probes/test_metal_rw_simd.mm: bad=0 vs scalar ref; 1.7-3.3x
 * faster than gemm_int4_rowwise_batched on all production tail shapes. */
void metal_dispatch_gemm_int4_rw_simd(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf, MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf, MetalBufferHandle output_buf,
    int rows, int logical_cols, int packed_cols, int lanes) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf || !output_buf ||
        rows <= 0 || logical_cols <= 0 || lanes <= 0) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_rw_simd");
    if (!pipeline) { [pool release]; return; }
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:0 atIndex:3];
    uint32_t values[] = {(uint32_t)rows, (uint32_t)logical_cols,
                         (uint32_t)packed_cols, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 4; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:4 + i];
    [encoder setThreadgroupMemoryLength:32 * 72 * 2 atIndex:0];
    [encoder setThreadgroupMemoryLength:64 * 40 * 2 atIndex:1];
    [encoder setThreadgroupMemoryLength:32 * 32 * 4 atIndex:2];
    [encoder setThreadgroupMemoryLength:32 * 2 atIndex:3];
    MTLSize grid = MTLSizeMake((NSUInteger)((rows + 31) / 32),
                               (NSUInteger)((lanes + 31) / 32), 1);
    MTLSize group = MTLSizeMake(128, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_gemm_int4_rowwise_batched(
    MetalContext* ctx,   MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,  MetalBufferHandle weight_buf,
    MetalBufferHandle scale_buf,  MetalBufferHandle output_buf,
    int rows, int logical_cols, int packed_cols, int lanes) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !scale_buf || !output_buf ||
        rows <= 0 || logical_cols <= 0 || lanes <= 0) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemm_int4_rowwise_batch");
    if (!pipeline) { [pool release]; return; }
    int lc = MIN(lanes, 32);
    int knar = 1;
    while (knar * 2 <= (int)(logical_cols / 2) && knar < 32) knar *= 2;
    if (knar < 1) knar = 1;
    int tg = lc * knar;
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scale_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:0 atIndex:3];
    uint32_t values[] = {(uint32_t)rows, (uint32_t)logical_cols,
                        (uint32_t)packed_cols, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 4; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:4 + i];
    [encoder setThreadgroupMemoryLength:tg * sizeof(float) atIndex:0];
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(tg, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_add_channel_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle a_buf,
    MetalBufferHandle b_buf,
    MetalBufferHandle out_buf,
    int channels,
    int lanes
) {
    if (!ctx || !cmd_buf || !a_buf || !b_buf || !out_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "add_channel_fp16");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)a_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)b_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buf offset:0 atIndex:2];
    uint32_t values[] = {(uint32_t)channels, (uint32_t)lanes};
    [encoder setBytes:&values[0] length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&values[1] length:sizeof(uint32_t) atIndex:4];
    MTLSize grid = MTLSizeMake(lanes, channels, 1);
    MTLSize group = MTLSizeMake(MIN(lanes, 32), MIN(channels, 8), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_rmsnorm_channel_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle input_buf,
    MetalBufferHandle weight_buf,
    MetalBufferHandle output_buf,
    int channels,
    int lanes,
    float eps
) {
    if (!ctx || !cmd_buf || !input_buf || !weight_buf || !output_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "rmsnorm_channel_fp16");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)input_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)weight_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:0 atIndex:2];
    uint32_t values[] = {(uint32_t)channels, (uint32_t)lanes};
    [encoder setBytes:&values[0] length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&values[1] length:sizeof(uint32_t) atIndex:4];
    [encoder setBytes:&eps length:sizeof(float) atIndex:5];
    [encoder setThreadgroupMemoryLength:8 * sizeof(float) atIndex:0];
    MTLSize grid = MTLSizeMake(256, lanes, 1);
    MTLSize group = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_swiglu_channel_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle gate_up_buf,
    MetalBufferHandle output_buf,
    int intermediate,
    int lanes
) {
    if (!ctx || !cmd_buf || !gate_up_buf || !output_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "swiglu_channel_fp16");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)gate_up_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)output_buf offset:0 atIndex:1];
    uint32_t values[] = {(uint32_t)intermediate, (uint32_t)lanes};
    [encoder setBytes:&values[0] length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&values[1] length:sizeof(uint32_t) atIndex:3];
    MTLSize grid = MTLSizeMake(lanes, intermediate, 1);
    MTLSize group = MTLSizeMake(MIN(lanes, 32), MIN(intermediate, 8), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

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
) {
    if (!ctx || !cmd_buf || !state_buf || !decay_buf || !key_buf ||
        !query_buf || !value_buf || !beta_buf || !output_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gdn_recurrence");
    if (!pipeline) {
        [pool release];
        return;
    }
    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    MetalBufferHandle buffers[] = {
        state_buf, decay_buf, key_buf, query_buf, value_buf, beta_buf, output_buf
    };
    for (NSUInteger i = 0; i < 7; ++i)
        [encoder setBuffer:(id<MTLBuffer>)buffers[i] offset:0 atIndex:i];
    uint32_t values[] = {(uint32_t)heads, (uint32_t)key_dim,
                         (uint32_t)value_dim, (uint32_t)lanes};
    for (NSUInteger i = 0; i < 4; ++i)
        [encoder setBytes:&values[i] length:sizeof(uint32_t) atIndex:7 + i];
    MTLSize grid = MTLSizeMake(value_dim, heads, 1);
    MTLSize group = MTLSizeMake(MIN(value_dim, 32), MIN(heads, 8), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_argmax_fp16(
    MetalContext* ctx,
    MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle logits_buf,
    MetalBufferHandle out_tokens_buf,
    int B,
    int V
) {
    if (!ctx || !cmd_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline = (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "argmax_fp16");
    if (!pipeline) {
        [pool release];
        return;
    }

    id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:(id<MTLBuffer>)logits_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)out_tokens_buf offset:0 atIndex:1];

    uint32_t v_val = (uint32_t)V;
    uint32_t b_val = (uint32_t)B;
    [encoder setBytes:&v_val length:sizeof(uint32_t) atIndex:2];
    [encoder setBytes:&b_val length:sizeof(uint32_t) atIndex:3];

    MTLSize threadsPerGrid = MTLSizeMake((B + 31) / 32 * 32, 1, 1);
    MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [pool release];
}


/* ========================================================================= */
/* Attention core dispatches (scores -> softmax -> P*V)                       */
/* ========================================================================= */

void metal_dispatch_attn_scores_fp16(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle q_buf, MetalBufferHandle k_cache,
    MetalBufferHandle scores_buf,
    uint32_t base_pos, uint32_t kv_len, uint32_t lanes, uint32_t cap,
    uint32_t heads_q, uint32_t heads_kv, uint32_t dim) {
    if (!ctx || !cmd_buf || !q_buf || !k_cache || !scores_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "attn_scores_fp16");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)q_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)k_cache offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)scores_buf offset:0 atIndex:2];
    const uint32_t vals[] = {base_pos, kv_len, cap, heads_q, heads_kv, dim};
    for (NSUInteger i = 0; i < 6; ++i)
        [encoder setBytes:&vals[i] length:sizeof(uint32_t) atIndex:3 + i];
    MTLSize grid = MTLSizeMake(kv_len, heads_q, lanes);
    MTLSize group = MTLSizeMake(MIN(kv_len, 256u), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_attn_softmax_fp16(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle scores_buf, MetalBufferHandle probs_buf,
    uint32_t rows, uint32_t base_pos, uint32_t cap, uint32_t heads_q) {
    if (!ctx || !cmd_buf || !scores_buf || !probs_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "attn_softmax_fp16");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)scores_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)probs_buf offset:0 atIndex:1];
    const uint32_t vals[] = {base_pos, cap, heads_q};
    for (NSUInteger i = 0; i < 3; ++i)
        [encoder setBytes:&vals[i] length:sizeof(uint32_t) atIndex:2 + i];
    [encoder setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    MTLSize grid = MTLSizeMake(256, rows, 1);
    MTLSize group = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_attn_pv_fp16(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle probs_buf, MetalBufferHandle v_cache,
    MetalBufferHandle out_buf,
    uint32_t base_pos, uint32_t lanes, uint32_t cap,
    uint32_t heads_q, uint32_t heads_kv, uint32_t dim) {
    if (!ctx || !cmd_buf || !probs_buf || !v_cache || !out_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "attn_pv_fp16");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)probs_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)v_cache offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)out_buf offset:0 atIndex:2];
    const uint32_t vals[] = {base_pos, cap, heads_q, heads_kv, dim};
    for (NSUInteger i = 0; i < 5; ++i)
        [encoder setBytes:&vals[i] length:sizeof(uint32_t) atIndex:3 + i];
    const uint32_t dtiles = dim / 64;
    MTLSize grid = MTLSizeMake(dtiles, heads_q, lanes);
    MTLSize group = MTLSizeMake(MIN(dtiles, 4u), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}

void metal_dispatch_gemv_int4_groupwise(
    MetalContext* ctx, MetalCommandBufferHandle cmd_buf,
    MetalBufferHandle w_buf, MetalBufferHandle scales_buf,
    MetalBufferHandle biases_buf, MetalBufferHandle x_buf,
    MetalBufferHandle y_buf,
    uint32_t rows, uint32_t packed_cols, uint32_t K, uint32_t groups) {
    if (!ctx || !cmd_buf || !w_buf || !scales_buf || !biases_buf ||
        !x_buf || !y_buf) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    id<MTLComputePipelineState> pipeline =
        (id<MTLComputePipelineState>)metal_get_pipeline(ctx, "gemv_int4_groupwise");
    if (!pipeline) { [pool release]; return; }
    id<MTLComputeCommandEncoder> encoder =
        [(id<MTLCommandBuffer>)cmd_buf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)w_buf offset:0 atIndex:0];
    [encoder setBuffer:(id<MTLBuffer>)scales_buf offset:0 atIndex:1];
    [encoder setBuffer:(id<MTLBuffer>)biases_buf offset:0 atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)x_buf offset:0 atIndex:3];
    [encoder setBuffer:(id<MTLBuffer>)y_buf offset:0 atIndex:4];
    const uint32_t vals[] = {packed_cols, K, groups};
    for (NSUInteger i = 0; i < 3; ++i)
        [encoder setBytes:&vals[i] length:sizeof(uint32_t) atIndex:5 + i];
    [encoder setThreadgroupMemoryLength:256 * sizeof(float) atIndex:0];
    MTLSize grid = MTLSizeMake(rows, 1, 1);
    MTLSize group = MTLSizeMake(256, 1, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [pool release];
}
