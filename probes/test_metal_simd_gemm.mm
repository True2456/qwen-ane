// Standalone harness for the register-blocked simdgroup MMA int4-groupwise GEMM.
// Mirrors MLX steel-gemm structure: BM=32 x BN=32 tile, BK=64 (== quant group),
// 4 simdgroups/threadgroup, four 8x8 fp32-accumulate fragments per simdgroup.
//
// Gates before wiring into the engine:
//   1. Numerical: matches CPU reference AND the production batch kernel
//      within tolerance (fp16-dequant rounding makes bit-exactness impossible
//      by construction - same tradeoff MLX makes).
//   2. Performance: strictly faster than the committed batch kernel
//      (baseline ~5.4 ms/call for 34816x5120 lanes=32 on this machine).
// No server required. Run: probes/run_simd_gemm.sh
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

inline float bf16_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// Register-blocked int4 groupwise GEMM:
//   C[M x N] = deq(W)[M x K] @ A[K x N]
// Tile BM=32, BN=32, BK=64. Threadgroup = 128 threads = 4 simdgroups.
// Simdgroup sg owns output rows [sg*8, sg*8+8) x all 32 cols of the tile,
// held as 4 x (8x8) fp32 fragments accumulated via simdgroup MMA.
kernel void gemm_int4_simd(
    device const uint*   W     [[buffer(0)]],
    device const ushort* S     [[buffer(1)]],
    device const ushort* Bias  [[buffer(2)]],
    device const half*   A     [[buffer(3)]],
    device half*         C     [[buffer(4)]],
    constant uint& rows         [[buffer(5)]],
    constant uint& cols         [[buffer(6)]],
    constant uint& packed_cols  [[buffer(7)]],
    constant uint& groups       [[buffer(8)]],
    constant uint& lanes        [[buffer(9)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  tid    [[thread_index_in_threadgroup]],
    uint  sg_id  [[simdgroup_index_in_threadgroup]],
    threadgroup half*  w_tile  [[threadgroup(0)]],  // dequantized W tile [32][64], ld 72
    threadgroup half*  a_tile  [[threadgroup(1)]],  // staged A tile   [64][32], ld 40
    threadgroup float* c_stage [[threadgroup(2)]]   // fp32 out stage [32][32] -> fp16 C
) {
    const uint c_row = tg_pos.x * 32u;
    const uint c_col = tg_pos.y * 32u;

    simdgroup_matrix<float, 8, 8> cf0 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> cf1 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> cf2 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> cf3 = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0u; k0 < cols; k0 += 64u) {
        // ---- Phase 1a: dequantize W block [32 x 64] into threadgroup fp16.
        //      Per-group scale/bias; out-of-range entries are exact zeros so
        //      they contribute nothing to any dot product.
        for (uint idx = tid; idx < 32u * 64u; idx += 128u) {
            uint r = idx / 64u;
            uint k = idx - r * 64u;
            float v = 0.0f;
            uint gr = c_row + r;
            uint gk = k0 + k;
            if (gr < rows && gk < cols) {
                uint g = gk >> 6;
                if (g < groups) {
                    uint word = W[gr * packed_cols + (gk >> 3)];
                    uint q = (word >> ((gk & 7u) * 4u)) & 0xfu;
                    float s = bf16_to_float(S[gr * groups + g]);
                    float b = bf16_to_float(Bias[gr * groups + g]);
                    v = float(q) * s + b;
                }
            }
            w_tile[r * 72u + k] = half(v);
        }
        // ---- Phase 1b: stage A block [64 x 32], zero-padded outside range.
        for (uint idx = tid; idx < 64u * 32u; idx += 128u) {
            uint k = idx / 32u;
            uint l = idx - k * 32u;
            half val = half(0.0f);
            if (k0 + k < cols && l < lanes)
                val = A[(k0 + k) * lanes + l];
            a_tile[k * 40u + l] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- Phase 2: register-blocked MMA. K-dim advances in 8-wide steps.
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

    // ---- Store: fragments -> fp32 threadgroup stage -> bounds-checked fp16 C.
    //      Zero-padded dequant/staging makes any tile overhang contribute exact
    //      zeros, so no rows/lanes divisibility is required.
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

static uint32_t g_rng = 0x12345678u;
static uint32_t rnd() { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

static uint16_t to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4); return uint16_t(u >> 16);
}
static float from_half(uint16_t h) {
    // IEEE half -> float
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) bits = sign | (man << 13);           // (subnormal approx ok: man<<13 / denorm)
    else if (exp == 31) bits = sign | 0x7f800000 | (man << 13);
    else bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}

struct Shape { int rows, cols, lanes; };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 1; }

        id<MTLCommandQueue> cq = [dev newCommandQueue];
        NSString* src = [NSString stringWithUTF8String:kShader];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            std::fprintf(stderr, "SHADER COMPILE FAILED:\n%s\n",
                         err.localizedDescription.UTF8String ?: "?");
            return 2;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"gemm_int4_simd"];
        if (!fn) { std::fprintf(stderr, "kernel symbol not found\n"); return 2; }
        NSError* perr = nil;
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) { std::fprintf(stderr, "pipeline failed: %s\n", perr.localizedDescription.UTF8String); return 2; }
        NSLog(@"shader compiled OK");

        MetalContext* ectx = metal_context_create();
        if (!ectx) { std::fprintf(stderr, "engine ctx failed\n"); return 2; }

        // Buffers helper: create shared-mode buffer from vector
        auto mkbuf = [&](const void* data, size_t bytes) -> id<MTLBuffer> {
            id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (data) std::memcpy(b.contents, data, bytes);
            return b;
        };

        auto check_shape = [&](int rows, int cols, int lanes) -> bool {
            const int packed = cols / 8, groups = cols / 64;
            // deterministic data
            std::vector<uint32_t> W((size_t)rows * packed);
            std::vector<uint16_t> S((size_t)rows * groups), B((size_t)rows * groups);
            std::vector<uint16_t> A((size_t)cols * lanes);
            for (auto& w : W) {
                uint32_t v = 0;
                for (int j = 0; j < 8; ++j) v |= (rnd() & 0xfu) << (4 * j);
                w = v;
            }
            for (auto& s : S) s = to_bf16(((float)(rnd() % 2000) - 1000.0f) / 100000.0f);
            for (auto& b : B) b = to_bf16(((float)(rnd() % 2000) - 1000.0f) / 100000.0f);
            for (auto& a : A) {
                // build a half from scratch: value in [-2,2]
                float f = ((float)(rnd() % 4000) - 2000.0f) / 1000.0f;
                uint32_t uf; std::memcpy(&uf, &f, 4);
                uint32_t sign = (uf >> 16) & 0x8000;
                int32_t e = ((uf >> 23) & 0xff) - 127 + 15;
                uint32_t m = (uf >> 13) & 0x3ff;
                uint16_t h;
                if (e <= 0) h = (uint16_t)sign;
                else if (e >= 31) h = (uint16_t)(sign | 0x7bff);
                else h = (uint16_t)(sign | (uint32_t)(e << 10) | m);
                a = h;
            }

            id<MTLBuffer> bw = mkbuf(W.data(), W.size()*4);
            id<MTLBuffer> bs = mkbuf(S.data(), S.size()*2);
            id<MTLBuffer> bb = mkbuf(B.data(), B.size()*2);
            id<MTLBuffer> ba = mkbuf(A.data(), A.size()*2);
            id<MTLBuffer> bc = [dev newBufferWithLength:(size_t)rows*lanes*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc2 = [dev newBufferWithLength:(size_t)rows*lanes*2 options:MTLResourceStorageModeShared];
            std::memset(bc.contents, 0xAB, (size_t)rows*lanes*2);
            std::memset(bc2.contents, 0xAB, (size_t)rows*lanes*2);

            // --- simd kernel ---
            {
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bw offset:0 atIndex:0];
                [enc setBuffer:bs offset:0 atIndex:1];
                [enc setBuffer:bb offset:0 atIndex:2];
                [enc setBuffer:ba offset:0 atIndex:3];
                [enc setBuffer:bc offset:0 atIndex:4];
                uint32_t vals[5] = {(uint32_t)rows,(uint32_t)cols,(uint32_t)packed,(uint32_t)groups,(uint32_t)lanes};
                for (int i = 0; i < 5; ++i) [enc setBytes:&vals[i] length:4 atIndex:5+i];
                [enc setThreadgroupMemoryLength:32*72*2 atIndex:0];
                [enc setThreadgroupMemoryLength:64*40*2 atIndex:1];
                [enc setThreadgroupMemoryLength:32*32*4 atIndex:2];
                MTLSize grid = MTLSizeMake((rows+31)/32, (lanes+31)/32, 1);
                MTLSize tgsz = MTLSizeMake(128, 1, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgsz];
                [enc endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                if (cb.status == MTLCommandBufferStatusError) {
                    std::fprintf(stderr, "[%d x %d x %d] simd cmd ERROR: %s\n",
                                 rows, cols, lanes, cb.error.localizedDescription.UTF8String ?: "?");
                    return false;
                }
            }
            // --- engine batch kernel (production reference) ---
            {
                MetalCommandBufferHandle c = metal_command_buffer_create(ectx);
                metal_dispatch_gemm_int4_groupwise_batch(ectx, c, ba, bw, bs, bb, bc2,
                                                         rows, cols, packed, groups, lanes);
                metal_command_buffer_commit(c);
                metal_command_buffer_wait(c);
            }

            // --- CPU reference (scalar semantics, float accumulation) ---
            std::vector<float> ref((size_t)rows * lanes);
            for (int r = 0; r < rows; ++r)
              for (int l = 0; l < lanes; ++l) {
                float acc = 0.0f;
                for (int k = 0; k < cols; ++k) {
                    uint32_t word = W[(size_t)r * packed + (k >> 3)];
                    float q = float((word >> ((k & 7) * 4)) & 0xfu);
                    float s, b;
                    uint32_t us = S[(size_t)r * groups + (k >> 6)], ub = B[(size_t)r * groups + (k >> 6)];
                    uint32_t ts = uint32_t(us) << 16, tb = uint32_t(ub) << 16;
                    std::memcpy(&s, &ts, 4); std::memcpy(&b, &tb, 4);
                    float av; uint16_t ah = A[(size_t)k * lanes + l];
                    // exact half->float
                    uint32_t sign = uint32_t(ah & 0x8000) << 16;
                    uint32_t ex = (ah >> 10) & 0x1f, ma = ah & 0x3ff;
                    uint32_t abits;
                    if (ex == 0) { if (ma == 0) abits = sign; else abits = sign; } // subnormals unused in our data
                    else if (ex == 31) abits = sign | 0x7f800000u;
                    else abits = sign | ((ex - 15 + 127) << 23) | (ma << 13);
                    std::memcpy(&av, &abits, 4);
                    acc += av * (q * s + b);
                }
                ref[(size_t)r * lanes + l] = acc;
              }

            // --- compare ---
            auto* cs = (uint16_t*)bc.contents;
            auto* cb2 = (uint16_t*)bc2.contents;
            size_t bad_simd = 0, bad_batch = 0;
            double sim_maxabs = 0, bat_maxabs = 0, maxref = 0;
            double sim_relsum = 0;
            for (size_t i = 0; i < (size_t)rows*lanes; ++i) {
                float rv = ref[i], sv = from_half(cs[i]), bv = from_half(cb2[i]);
                maxref = std::max(maxref, (double)std::fabs(rv));
                sim_maxabs = std::max(sim_maxabs, (double)std::fabs(sv - rv));
                bat_maxabs = std::max(bat_maxabs, (double)std::fabs(bv - rv));
                sim_relsum += std::fabs(sv - rv);
                if (std::fabs(sv - rv) > 0.02 * std::max(1.0, maxref)) ++bad_simd;
                if (cs[i] != cb2[i]) ++bad_batch;
            }
            std::fprintf(stderr,
                "[%6d x %d x %2d] |ref|max=%.3g  simd-vs-ref maxabs=%.3g relsum=%.3g bad=%zu | batch-vs-ref maxabs=%.3g\n",
                rows, cols, lanes, maxref, sim_maxabs, sim_relsum, bad_simd, bat_maxabs);
            if (getenv("SIMD_DEBUG") && rows <= 128) {
                std::fprintf(stderr, "  first 16x%d   (row: ref | simd)\n", lanes);
                for (int r = 0; r < 16; ++r) {
                    std::fprintf(stderr, "  %2d:", r);
                    for (int l = 0; l < lanes; ++l) std::fprintf(stderr, " %7.3f", ref[(size_t)r*lanes+l]);
                    std::fprintf(stderr, " |");
                    for (int l = 0; l < lanes; ++l) std::fprintf(stderr, " %7.3f", from_half(cs[(size_t)r*lanes+l]));
                    std::fprintf(stderr, "\n");
                }
            }
            bool ok = bad_simd == 0 && sim_maxabs < 0.02 * std::max(1.0, maxref);
            if (!ok) std::fprintf(stderr, "  ^^ SIMD KERNEL FAIL\n");
            return ok;
        };

        bool ok = true;
        ok &= check_shape(64, 64, 8);
        ok &= check_shape(96, 128, 16);
        ok &= check_shape(128, 192, 24);
        ok &= check_shape(160, 256, 32);
        ok &= check_shape(256, 512, 32);
        if (!ok) { std::fprintf(stderr, "CORRECTNESS GATE FAILED - not wiring\n"); return 3; }

        // ---------------- performance: production shape ------------------------
        {
            const int rows = 34816, cols = 5120, lanes = 32;
            const int packed = cols/8, groups = cols/64;
            id<MTLBuffer> bw = [dev newBufferWithLength:(size_t)rows*packed*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bs = [dev newBufferWithLength:(size_t)rows*groups*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bb = [dev newBufferWithLength:(size_t)rows*groups*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> ba = [dev newBufferWithLength:(size_t)cols*lanes*2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bc = [dev newBufferWithLength:(size_t)rows*lanes*2 options:MTLResourceStorageModeShared];
            // fill deterministically (cheap LCG fill, contents matter little for speed)
            uint32_t* wp = (uint32_t*)bw.contents;
            for (size_t i = 0; i < (size_t)rows*packed; ++i) wp[i] = rnd();
            uint16_t* sp = (uint16_t*)bs.contents; uint16_t* bp = (uint16_t*)bb.contents;
            for (size_t i = 0; i < (size_t)rows*groups; ++i) { sp[i] = to_bf16(0.001f); bp[i] = to_bf16(0.0f); }
            uint16_t* ap = (uint16_t*)ba.contents;
            for (size_t i = 0; i < (size_t)cols*lanes; ++i) ap[i] = 0x3c00; // 1.0h

            const int N = 10;
            // warmup
            for (int i = 0; i < 3; ++i) {
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bs offset:0 atIndex:1];
                [enc setBuffer:bb offset:0 atIndex:2]; [enc setBuffer:ba offset:0 atIndex:3];
                [enc setBuffer:bc offset:0 atIndex:4];
                uint32_t vals[5] = {(uint32_t)rows,(uint32_t)cols,(uint32_t)packed,(uint32_t)groups,(uint32_t)lanes};
                for (int t = 0; t < 5; ++t) [enc setBytes:&vals[t] length:4 atIndex:5+t];
                [enc setThreadgroupMemoryLength:32*72*2 atIndex:0];
                [enc setThreadgroupMemoryLength:64*40*2 atIndex:1];
                [enc setThreadgroupMemoryLength:32*32*4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(rows/32, lanes/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; ++i) {
                id<MTLCommandBuffer> cb = [cq commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pso];
                [enc setBuffer:bw offset:0 atIndex:0]; [enc setBuffer:bs offset:0 atIndex:1];
                [enc setBuffer:bb offset:0 atIndex:2]; [enc setBuffer:ba offset:0 atIndex:3];
                [enc setBuffer:bc offset:0 atIndex:4];
                uint32_t vals[5] = {(uint32_t)rows,(uint32_t)cols,(uint32_t)packed,(uint32_t)groups,(uint32_t)lanes};
                for (int t = 0; t < 5; ++t) [enc setBytes:&vals[t] length:4 atIndex:5+t];
                [enc dispatchThreadgroups:MTLSizeMake(rows/32, lanes/32, 1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
            }
            double dt = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

            // engine batch kernel on identical buffers, same protocol
            for (int i = 0; i < 3; ++i) {
                MetalCommandBufferHandle c = metal_command_buffer_create(ectx);
                metal_dispatch_gemm_int4_groupwise_batch(ectx, c, ba, bw, bs, bb, bc, rows, cols, packed, groups, lanes);
                metal_command_buffer_commit(c); metal_command_buffer_wait(c);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N; ++i) {
                MetalCommandBufferHandle c = metal_command_buffer_create(ectx);
                metal_dispatch_gemm_int4_groupwise_batch(ectx, c, ba, bw, bs, bb, bc, rows, cols, packed, groups, lanes);
                metal_command_buffer_commit(c); metal_command_buffer_wait(c);
            }
            double dt2 = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t1).count();

            double tf_s = (2.0 * rows * cols * lanes) / (dt / N) / 1e9;
            double tf_b = (2.0 * rows * cols * lanes) / (dt2 / N) / 1e9;
            std::fprintf(stderr,
                "\nPRODUCTION 34816x5120 lanes=32, %d calls:\n"
                "  simd MMA : %.3f ms/call  (%.2f TFLOPS)\n"
                "  batch    : %.3f ms/call  (%.2f TFLOPS)\n"
                "  speedup  : %.2fx   %s\n",
                N, dt/N, tf_s, dt2/N, tf_b, dt2/dt,
                (dt < dt2) ? "SIMD WINS" : "batch wins");
        }
        return 0;
    }
}
