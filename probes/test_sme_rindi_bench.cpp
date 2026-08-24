// rindi_bench.cpp - SME2 dense throughput at rindi prefill shapes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <vector>
#include <ane/ane.hpp>

static void* alloc(size_t n) { return std::aligned_alloc(64, ((n + 63) / 64) * 64); }

static void bench_rindi_shapes() {
    struct Cfg { int M, K, N; const char* name; };
    Cfg cfgs[] = {
        {32,  32,   32,   "sanity"},
        {128, 5120, 5120, "proj S=128"},
        {256, 5120, 17408, "dn S=256"},
        {512, 5120, 34816, "gu S=512"},
    };
    for (auto& c : cfgs) {
        float* A    = static_cast<float*>(alloc(c.M * c.K * 4));
        float* B    = static_cast<float*>(alloc(c.K * c.N * 4));
        float* bias = static_cast<float*>(alloc(c.N * 4));
        float* Cc   = static_cast<float*>(alloc(c.M * c.N * 4));
        for (int i = 0; i < c.M * c.K; i++) A[i] = 0.001f * (i % 37 - 18);
        for (int i = 0; i < c.K * c.N; i++) B[i] = 0.001f * (i % 29 - 14);
        std::memset(Cc, 0, c.M * c.N * 4);

        char script[512];
        std::snprintf(script, sizeof(script),
            " dense_fp32(%d, %d, %d, 1.0, 0, params[0], params[1], params[2], params[3]); ",
            c.M, c.N, c.K);
        ane::script s1(script);
        s1.exec({A, B, bias, Cc});

        const int NIT = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NIT; ++i) s1.exec({A, B, bias, Cc});
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count() / NIT;
        double tf = 2.0 * (double)c.M * c.N * c.K / ms / 1e9;
        printf("  %-12s: %.3f ms | %.3f TFLOPS\n", c.name, ms, tf);
        std::fflush(stdout);
        std::free(A); std::free(B); std::free(bias); std::free(Cc);
    }
}

int main() {
    printf("=== RINDI SME PREFILL SHAPES ===\n");
    // warm the interpreter exactly like the coverage suite does
    float* wa    = static_cast<float*>(alloc(32 * 32 * 4));
    float* wb    = static_cast<float*>(alloc(32 * 16 * 4));
    float* wbias = static_cast<float*>(alloc(32 * 4));
    float* wc    = static_cast<float*>(alloc(32 * 32 * 4));
    ane::script warm(R"( dense_fp32(32, 32, 16, 1.0, 0, params[0], params[1], params[2], params[3]); )");
    warm.exec({wa, wb, wbias, wc});
    std::free(wa); std::free(wb); std::free(wbias); std::free(wc);

    bench_rindi_shapes();
    printf("=== done ===\n");
    return 0;
}
