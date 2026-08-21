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
"using namespace metal;\n"
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

