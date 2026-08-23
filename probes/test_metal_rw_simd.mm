// Standalone harness for the register-blocked simdgroup MMA ROWWISE int4 GEMM.
// Rowwise layout (compile_chain_int4): W = uchar[rows, packed_cols] with two
// nibbles per byte (low = col 2c, high = col 2c+1), S = fp16 half[rows],
// A = half[K, lanes], C = half[rows, lanes]; dequant v = q * scale[row].
//
// Gates before wiring:
//   1. bad=0 vs CPU scalar reference AND vs the engine rowwise_batched kernel
//      (fp16-dequant tolerance, same class as the groupwise simd kernel).
//   2. Strictly faster than gemm_int4_rowwise_batched on production shapes
//      (gate_proj is 34816 x 5120 - the single hottest tail GEMM).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

extern "C" {
#include "runtime/metal_engine.h"
}

static const char* kShader = R"MTL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Register-blocked ROWWISE int4 GEMM: C[M,N] = deq(W)[M,K] @ A[K,N].
// BM=32 x BN=32 x BK=64, 128 threads = 4 simdgroups, 4 fp32 8x8 frags each.
kernel void gemm_int4_rw_simd(
    device const uchar*  W     [[buffer(0)]],
    device const half*   S     [[buffer(1)]],
    device const half*   A     [[buffer(2)]],
    device half*         C     [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& cols         [[buffer(5)]],
    constant uint& packed_cols  [[buffer(6)]],
    constant uint& lanes        [[buffer(7)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  sg_id  [[simdgroup_index_in_threadgroup]],
    threadgroup half*  w_tile  [[threadgroup(0)]],  // [32][64] ld 72
    threadgroup half*  a_tile  [[threadgroup(1)]],  // [64][32] ld 40
    threadgroup float* c_stage [[threadgroup(2)]],  // [32][32] fp32
    threadgroup half*  scale_sh[[threadgroup(3)]]    // [32] row scales
) {
    const uint c_row = tg_pos.x * 32u;
    const uint c_col = tg_pos.y * 32u;

    // Per-row fp16 scales: loaded once, reused across all K steps.
    if (tid < 32u)
        scale_sh[tid] = (c_row + tid < rows) ? S[c_row + tid] : half(0.0f);
    // readers of scale_sh sit in every simdgroup - publish before first use
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<float, 8, 8> cf0 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> cf1 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> cf2 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> cf3 = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < cols; k0 += 64u) {
        // ---- Dequant W block [32 x 64]: two nibbles per byte, per-row scale.
        for (uint idx = tid; idx < 32u * 64u; idx += 128u) {
            uint r = idx / 64u;
            uint k = idx - r * 64u;
            half v = half(0.0f);
            uint gr = c_row + r;
            uint gk = k0 + k;
            if (gr < rows && gk < cols) {
                uint byte = W[gr * packed_cols + (gk >> 1)];
                int q = int((gk & 1u) ? ((byte >> 4) & 0xfu) : (byte & 0xfu));
                if (q >= 8) q -= 16;   // signed int4 nibble
                v = half(float(q) * float(scale_sh[r]));
            }
            w_tile[r * 72u + k] = v;
        }
        // ---- Stage A block [64 x 32], zero-padded.
        for (uint idx = tid; idx < 64u * 32u; idx += 128u) {
            uint k = idx / 32u;
            uint l = idx - k * 32u;
            half val = half(0.0f);
            if (k0 + k < cols && l < lanes)
                val = A[(k0 + k) * lanes + l];
            a_tile[k * 40u + l] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < 64u; kk += 8u) {
            simdgroup_matrix<half, 8, 8> At;
            simdgroup_load(At, &w_tile[(sg_id * 8u) * 72u + kk], 72ul, ulong2(0, 0), false);
            #pragma unroll
            for (uint n = 0u; n < 4u; ++n) {
                simdgroup_matrix<half, 8, 8> Bt;
                simdgroup_load(Bt, &a_tile[kk * 40u + n * 8u], 40ul, ulong2(0, 0), false);
                if (n == 0u) simdgroup_multiply_accumulate(cf0, At, Bt, cf0);
                if (n == 1u) simdgroup_multiply_accumulate(cf1, At, Bt, cf1);
                if (n == 2u) simdgroup_multiply_accumulate(cf2, At, Bt, cf2);
                if (n == 3u) simdgroup_multiply_accumulate(cf3, At, Bt, cf3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Store: fp32 stage -> bounds-checked fp16 C.
    simdgroup_store(cf0, &c_stage[(sg_id * 8u) * 32u + 0u],  32ul, ulong2(0, 0), false);
    simdgroup_store(cf1, &c_stage[(sg_id * 8u) * 32u + 8u],  32ul, ulong2(0, 0), false);
    simdgroup_store(cf2, &c_stage[(sg_id * 8u) * 32u + 16u], 32ul, ulong2(0, 0), false);
    simdgroup_store(cf3, &c_stage[(sg_id * 8u) * 32u + 24u], 32ul, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = tid; idx < 32u * 32u; idx += 128u) {
        uint r  = idx / 32u;
        uint l  = idx - r * 32u;
        uint gr = c_row + r;
        uint gc = c_col + l;
        if (gr < rows && gc < lanes)
            C[((size_t)gr * lanes) + gc] = half(c_stage[idx]);
    }
}
)MTL";

static uint32_t g_rng = 0xfeedbeefu;
static uint32_t rnd() { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

static uint16_t to_fp16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t e = int32_t((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = (u >> 13) & 0x3ff;
    if ((u & 0x7fffffff) <= 0x387fffff) return (uint16_t)sign;  // ~0
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7bff);
    return (uint16_t)(sign | (uint32_t)(e << 10) | m);
}
static float from_half(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t ex = (h >> 10) & 0x1f, ma = h & 0x3ff;
    uint32_t bits;
    if (ex == 0) bits = sign;
    else if (ex == 31) bits = sign | 0x7f800000u;
    else bits = sign | ((ex - 15 + 127) << 23) | (ma << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> cq = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kShader]
                                               options:nil error:&err];
        if (!lib) { std::fprintf(stderr, "SHADER COMPILE FAILED:\n%s\n",
                                 err.localizedDescription.UTF8String ?: "?"); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_int4_rw_simd"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { std::fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 2; }
        NSLog(@"shader compiled OK");

        MetalContext* ectx = metal_context_create();
        if (!ectx) { std::fprintf(stderr, "engine ctx failed\n"); return 2; }

        auto mkbuf = [&](const void* d, size_t n) -> id<MTLBuffer> {
            id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
            if (d) std::memcpy(b.contents, d, n);
            return b;
        };

        auto check_shape = [&](int rows, int cols, int lanes) -> bool {
            const int packed = cols / 2;
            std::vector<uint8_t> W((size_t)rows * packed);
            std::vector<uint16_t> S(rows);
            std::vector<uint16_t> A((size_t)cols * lanes);
            for (auto& w : W) w = (uint8_t)(rnd() & 0xff);
            for (auto& s : S) s = to_fp16(((float)(rnd() % 2000) - 1000.0f) / 40000.0f);
            for (auto& a : A) a = to_fp16(((float)(rnd() % 4000) - 2000.0f) / 1000.0f);

            id<MTLBuffer> bw = mkbuf(W.data(), W.size());
            id<MTLBuffer> bs = mkbuf(S.data(), S.size() * 2);
            id<MTLBuffer> ba = mkbuf(A.data(), A.size() * 2);
            id<MTLBuffer> bc  = [dev newBufferWithLength:(size_t)rows * lanes * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc2 = [dev newBufferWithLength:(size_t)rows * lanes * 2 options:MTLResourceStorageModeShared];
            std::memset(bc.contents, 0xAB, (size_t)rows * lanes * 2);
            std::memset(bc2.contents, 0xAB, (size_t)rows * lanes * 2);

            {   // simd kernel
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bw offset:0 atIndex:0];
                [enc setBuffer:bs offset:0 atIndex:1];
                [enc setBuffer:ba offset:0 atIndex:2];
                [enc setBuffer:bc offset:0 atIndex:3];
                uint32_t v[4] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)packed, (uint32_t)lanes};
                for (int i = 0; i < 4; ++i) [enc setBytes:&v[i] length:4 atIndex:4 + i];
                [enc setThreadgroupMemoryLength:32 * 72 * 2 atIndex:0];
                [enc setThreadgroupMemoryLength:64 * 40 * 2 atIndex:1];
                [enc setThreadgroupMemoryLength:32 * 32 * 4 atIndex:2];
                [enc setThreadgroupMemoryLength:32 * 2 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((rows + 31) / 32, (lanes + 31) / 32, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
                if (cb.status == MTLCommandBufferStatusError) {
                    std::fprintf(stderr, "[%dx%dx%d] cmd ERROR\n", rows, cols, lanes); return false;
                }
            }
            {   // engine rowwise batched (production reference)
                MetalCommandBufferHandle c = metal_command_buffer_create(ectx);
                metal_dispatch_gemm_int4_rowwise_batched(ectx, c, ba, bw, bs, bc2,
                                                         rows, cols, packed, lanes);
                metal_command_buffer_commit(c); metal_command_buffer_wait(c);
            }

            // CPU scalar reference
            std::vector<float> ref((size_t)rows * lanes);
            for (int r = 0; r < rows; ++r) {
                float sc = from_half(S[r]);
                for (int l = 0; l < lanes; ++l) {
                    float acc = 0.0f;
                    for (int k = 0; k < cols; ++k) {
                        uint8_t byte = W[(size_t)r * packed + (k >> 1)];
                        int qi = (k & 1) ? ((byte >> 4) & 0xf) : (byte & 0xf);
                        if (qi >= 8) qi -= 16;   // signed int4 nibble
                        float q = float(qi);
                        float av; uint16_t ah = A[(size_t)k * lanes + l];
                        uint32_t sign = uint32_t(ah & 0x8000) << 16;
                        uint32_t ex = (ah >> 10) & 0x1f, ma = ah & 0x3ff;
                        uint32_t abits = (ex == 0) ? sign : ((ex == 31) ? (sign | 0x7f800000u)
                                            : (sign | ((ex - 15 + 127) << 23) | (ma << 13)));
                        std::memcpy(&av, &abits, 4);
                        acc += av * q * sc;
                    }
                    ref[(size_t)r * lanes + l] = acc;
                }
            }

            auto* cs = (uint16_t*)bc.contents;
            auto* cb2p = (uint16_t*)bc2.contents;
            size_t bad = 0; double maxabs = 0, maxref = 0, batchdiff = 0;
            for (size_t i = 0; i < (size_t)rows * lanes; ++i) {
                float rv = ref[i], sv = from_half(cs[i]);
                maxref = std::max(maxref, (double)std::fabs(rv));
                maxabs = std::max(maxabs, (double)std::fabs(sv - rv));
                batchdiff = std::max(batchdiff, (double)std::fabs(from_half(cb2p[i]) - rv));
                if (std::fabs(sv - rv) > 0.02 * std::max(1.0, maxref)) ++bad;
            }
            std::fprintf(stderr,
                "[%6d x %d x %2d] |ref|max=%.3g simd-vs-ref maxabs=%.3g bad=%zu | engine-vs-ref maxabs=%.3g\n",
                rows, cols, lanes, maxref, maxabs, bad, batchdiff);
            bool ok = bad == 0 && maxabs < 0.02 * std::max(1.0, maxref);
            if (!ok) std::fprintf(stderr, "  ^^ RW-SIMD KERNEL FAIL\n");
            return ok;
        };

        bool ok = true;
        ok &= check_shape(64, 64, 8);
        ok &= check_shape(96, 128, 16);
        ok &= check_shape(128, 256, 24);
        ok &= check_shape(160, 512, 32);
        ok &= check_shape(256, 1024, 7);
        if (!ok) { std::fprintf(stderr, "CORRECTNESS GATE FAILED\n"); return 3; }

        // ---- performance on production tail shapes ----
        struct PShape { int rows, cols; const char* name; };
        PShape shapes[] = {
            {34816, 5120, "gate/up (h->2i)"},
            {17408, 5120, "down (i->h)"},
            {5120, 5120, "out/o_proj"},
            {5120, 2048, "qkv-ish"},
        };
        const int N = 10;
        for (auto& ps : shapes) {
            const int rows = ps.rows, cols = ps.cols, lanes = 32, packed = cols / 2;
            id<MTLBuffer> bw = mkbuf(nullptr, (size_t)rows * packed);
            id<MTLBuffer> bs = mkbuf(nullptr, (size_t)rows * 2);
            id<MTLBuffer> ba = mkbuf(nullptr, (size_t)cols * lanes * 2);
            id<MTLBuffer> bc = mkbuf(nullptr, (size_t)rows * lanes * 2);
            unsigned int rng = 7;
            uint8_t* wp = (uint8_t*)bw.contents;
            for (size_t i = 0; i < (size_t)rows * packed; ++i) { rng = rng * 1664525 + 1013904223; wp[i] = (uint8_t)(rng >> 16); }
            uint16_t* sp = (uint16_t*)bs.contents;
            for (int r = 0; r < rows; ++r) sp[r] = to_fp16(0.01f);
            uint16_t* ap = (uint16_t*)ba.contents;
            for (size_t i = 0; i < (size_t)cols * lanes; ++i) ap[i] = 0x3c00;

            for (int i = 0; i < 3; ++i) {
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bs offset:0 atIndex:1];
                [enc setBuffer:ba offset:0 atIndex:2]; [enc setBuffer:bc offset:0 atIndex:3];
                uint32_t v[4] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)packed, (uint32_t)lanes};
                for (int t = 0; t < 4; ++t) [enc setBytes:&v[t] length:4 atIndex:4 + t];
                [enc setThreadgroupMemoryLength:32 * 72 * 2 atIndex:0];
                [enc setThreadgroupMemoryLength:64 * 40 * 2 atIndex:1];
                [enc setThreadgroupMemoryLength:32 * 32 * 4 atIndex:2];
                [enc setThreadgroupMemoryLength:32 * 2 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(rows / 32, lanes / 32, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; ++i) {
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bs offset:0 atIndex:1];
                [enc setBuffer:ba offset:0 atIndex:2]; [enc setBuffer:bc offset:0 atIndex:3];
                uint32_t v[4] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)packed, (uint32_t)lanes};
                for (int t = 0; t < 4; ++t) [enc setBytes:&v[t] length:4 atIndex:4 + t];
                [enc setThreadgroupMemoryLength:32 * 72 * 2 atIndex:0];
                [enc setThreadgroupMemoryLength:64 * 40 * 2 atIndex:1];
                [enc setThreadgroupMemoryLength:32 * 32 * 4 atIndex:2];
                [enc setThreadgroupMemoryLength:32 * 2 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(rows / 32, lanes / 32, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            double dt = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

            for (int i = 0; i < 3; ++i) {
                MetalCommandBufferHandle c = metal_command_buffer_create(ectx);
                metal_dispatch_gemm_int4_rowwise_batched(ectx, c, ba, bw, bs, bc, rows, cols, packed, lanes);
                metal_command_buffer_commit(c); metal_command_buffer_wait(c);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; ++i) {
                MetalCommandBufferHandle c = metal_command_buffer_create(ectx);
                metal_dispatch_gemm_int4_rowwise_batched(ectx, c, ba, bw, bs, bc, rows, cols, packed, lanes);
                metal_command_buffer_commit(c); metal_command_buffer_wait(c);
            }
            double dt2 = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t1).count();
            double tf = 2.0 * rows * cols * lanes;
            std::fprintf(stderr,
                "%-16s %6dx%5d: rw-simd %.3f ms (%.2f TFLOPS) | engine rw-batched %.3f ms (%.2f TFLOPS) | %.2fx\n",
                ps.name, rows, cols, dt / N, tf / (dt / N) / 1e9,
                dt2 / N, tf / (dt2 / N) / 1e9, dt2 / dt);
        }
        return 0;
    }
}
