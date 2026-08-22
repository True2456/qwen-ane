// SPDX-License-Identifier: Apache-2.0
// Isolated Metal GEMM microbench. Times ONE kernel dispatch+wait repeatedly,
// separating kernel speed from Metal submit round-trip. No ANE needed.
#include "../runtime/metal_engine.h"
#include <cstdio>
#include <chrono>
#include <vector>

static double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
}

int main() {
    MetalContext* ctx = metal_context_create();
    if (!ctx) { std::fprintf(stderr, "no metal ctx\n"); return 2; }
    // Real model shapes: 5120 -> 34816 gate, groupwise int4 groupsize 64
    const int rows = 34816, cols = 5120, lanes = 32;
    const int groups = cols / 64;
    std::vector<uint8_t> w(rows * cols / 2);
    std::vector<uint16_t> sc(rows * groups), bias(rows * groups, 0);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t)((i * 7u) & 0x7f);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = 0x3c00u;
    std::vector<uint16_t> a(cols * lanes); for (auto& x : a) x = 0x3c00u;
    std::vector<uint16_t> c(rows * lanes);

    auto W = metal_buffer_create(ctx, w.size());
    auto S = metal_buffer_create(ctx, sc.size() * 2);
    auto Bm = metal_buffer_create(ctx, bias.size() * 2);
    auto A = metal_buffer_create(ctx, cols * lanes * 2);
    auto C = metal_buffer_create(ctx, rows * lanes * 2);
    std::memcpy(metal_buffer_get_contents(W), w.data(), w.size());
    std::memcpy(metal_buffer_get_contents(S), sc.data(), sc.size() * 2);
    std::memcpy(metal_buffer_get_contents(Bm), bias.data(), bias.size() * 2);
    std::memcpy(metal_buffer_get_contents(A), a.data(), cols * lanes * 2);

    // warmup
    for (int it = 0; it < 3; ++it) {
        auto c = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_groupwise(ctx, c, A, W, S, Bm, C,
                                           rows, cols, cols / 8, groups, lanes);
        metal_command_buffer_commit(c);
        metal_command_buffer_wait(c);
    }
    const int N = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < N; ++it) {
        auto c = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_groupwise(ctx, c, A, W, S, Bm, C,
                                           rows, cols, cols / 8, groups, lanes);
        metal_command_buffer_commit(c);
        metal_command_buffer_wait(c);
    }
    double dt = ms_since(t0);
    std::fprintf(stderr, "groupwise gemm 34816x5120 lanes=%d: %d calls %.3f ms = %.3f ms/call\n",
                lanes, N, dt, dt / N);
    std::fflush(stderr);

    // Correctness: run both kernels on identical input, compare bitwise
    auto b1 = metal_buffer_create(ctx, rows * lanes * 2);
    auto b2 = metal_buffer_create(ctx, rows * lanes * 2);
    {
        auto ch = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_groupwise(ctx, ch, A, W, S, Bm, b1,
                                           rows, cols, cols/8, groups, lanes);
        metal_command_buffer_commit(ch); metal_command_buffer_wait(ch);
        auto ch2 = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_groupwise_batch(ctx, ch2, A, W, S, Bm, b2,
                                                 rows, cols, cols/8, groups, lanes);
        metal_command_buffer_commit(ch2); metal_command_buffer_wait(ch2);
    }
    auto c1v = metal_buffer_get_contents(b1);
    auto c2v = metal_buffer_get_contents(b2);
    size_t mism = 0;
    double maxdiff = 0.0;
    for (size_t i = 0; i < rows * lanes; ++i) {
        uint16_t v1 = reinterpret_cast<uint16_t*>(c1v)[i];
        uint16_t v2 = reinterpret_cast<uint16_t*>(c2v)[i];
        if (v1 != v2) { ++mism;
            double va = (v1 >> 15) ? -1.0 * double(v1 & 0x7fffu) / 1024.0 : double(v1) / 1024.0;
            double vb = (v2 >> 15) ? -1.0 * double(v2 & 0x7fffu) / 1024.0 : double(v2) / 1024.0;
            maxdiff = std::max(maxdiff, std::fabs(va - vb));
        }
    }
    std::fprintf(stderr, "BATCH-vs-NAIVE mismatches=%zu/%zu maxdiff=%.6g\n",
                mism, rows * lanes, maxdiff);
    std::fflush(stderr);
    metal_buffer_release(b1); metal_buffer_release(b2);

    // Same shape, new K-parallel batched kernel
    for (int it = 0; it < 3; ++it) {
        auto c = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_groupwise_batch(ctx, c, A, W, S, Bm, C,
                                                 rows, cols, cols / 8, groups, lanes);
        metal_command_buffer_commit(c);
        metal_command_buffer_wait(c);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < N; ++it) {
        auto c = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_groupwise_batch(ctx, c, A, W, S, Bm, C,
                                                 rows, cols, cols / 8, groups, lanes);
        metal_command_buffer_commit(c);
        metal_command_buffer_wait(c);
    }
    double dt2 = ms_since(t1);
    std::fprintf(stderr, "groupwise BATCH gemm 34816x5120 lanes=%d: %d calls %.3f ms = %.3f ms/call\n",
                lanes, N, dt2, dt2 / N);
    std::fflush(stderr);

    // Many kernels in ONE command buffer (one commit+wait) - isolates pure
    // kernel compute from the per-call submit round trip. Mirrors the fused
    // tail, which encodes all projections into one buffer then waits once.
    const int M = 8;
    auto big = metal_command_buffer_create(ctx);
    auto tb0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < M; ++it)
        metal_dispatch_gemm_int4_groupwise_batch(ctx, big, A, W, S, Bm, C,
                                                 rows, cols, cols/8, groups, lanes);
    metal_command_buffer_commit(big);
    metal_command_buffer_wait(big);
    double one = ms_since(tb0);
    std::fprintf(stderr, "BATCH %d kernels in ONE buffer: %.3f ms total, %.3f ms/kernel (no per-call submit)\n",
                M, one, one / M);
    std::fflush(stderr);
    // RowWise correctness + speed: per-row scale, u8 packed nibbles
    {
        const size_t r2 = 17408, c2 = 5120, l2 = 32;
        std::vector<uint8_t> w2(r2 * c2 / 2);
        std::vector<uint16_t> s2(r2, 0x3c00u);
        for (size_t i = 0; i < w2.size(); ++i) w2[i] = (uint8_t)((i * 3u) & 0xff);
        std::vector<uint16_t> a2(c2 * l2); for (auto& x : a2) x = 0x3c00u;
        auto W2 = metal_buffer_create(ctx, w2.size());
        auto S2 = metal_buffer_create(ctx, s2.size() * 2);
        auto A2 = metal_buffer_create(ctx, c2 * l2 * 2);
        auto D1 = metal_buffer_create(ctx, r2 * l2 * 2);
        auto D2 = metal_buffer_create(ctx, r2 * l2 * 2);
        std::memcpy(metal_buffer_get_contents(W2), w2.data(), w2.size());
        std::memcpy(metal_buffer_get_contents(S2), s2.data(), s2.size()*2);
        std::memcpy(metal_buffer_get_contents(A2), a2.data(), c2*l2*2);
        auto cw = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_rowwise_offset(ctx, cw, A2, 0, W2, S2, D1, 0,
                                                r2, c2, c2/2, l2);
        metal_command_buffer_commit(cw); metal_command_buffer_wait(cw);
        auto cw2 = metal_command_buffer_create(ctx);
        metal_dispatch_gemm_int4_rowwise_batched(ctx, cw2, A2, W2, S2, D2,
                                                 r2, c2, c2/2, l2);
        metal_command_buffer_commit(cw2); metal_command_buffer_wait(cw2);
        auto e1 = metal_buffer_get_contents(D1);
        auto e2 = metal_buffer_get_contents(D2);
        size_t mm = 0; double mx = 0.0;
        for (size_t i = 0; i < r2 * l2; ++i) {
            uint16_t v1 = reinterpret_cast<uint16_t*>(e1)[i];
            uint16_t v2 = reinterpret_cast<uint16_t*>(e2)[i];
            if (v1 != v2) {
                ++mm;
                double va = (v1 >> 15) ? -1.0 * double(v1 & 0x7fffu) / 1024.0 : double(v1) / 1024.0;
                double vb = (v2 >> 15) ? -1.0 * double(v2 & 0x7fffu) / 1024.0 : double(v2) / 1024.0;
                mx = std::max(mx, std::fabs(va - vb));
            }
        }
        std::fprintf(stderr, "ROWWISE-batched-naive mism=%zu/%zu maxdiff=%.6g\n", mm, r2*l2, mx);
        metal_buffer_release(W2); metal_buffer_release(S2); metal_buffer_release(A2);
        metal_buffer_release(D1); metal_buffer_release(D2);
    }

    metal_buffer_release(A); metal_buffer_release(W); metal_buffer_release(S);
    metal_buffer_release(Bm); metal_buffer_release(C);
    metal_context_destroy(ctx);
    return 0;
}