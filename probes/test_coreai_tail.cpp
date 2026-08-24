// test_coreai_tail.cpp — drive the official CoreAI runtime from C++ via the
// Swift shim. Loads our full-size GDN tail bundle, runs one chunk on ANE,
// verifies vs PyTorch golden vectors, benchmarks throughput.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <dlfcn.h>

struct ANEContext;
extern "C" {
ANEContext* ane_context_create(void);
void ane_context_destroy(ANEContext*);
void* rindi_ane_load(const char* path, int preferANE);
long  rindi_ane_run(void* h, const uint16_t* xin, long rows, long cols,
                    uint16_t* outs, long outCap);
void  rindi_ane_free(void* h);
const char* rindi_ane_last_error(void);
}

static std::vector<uint16_t> load_f16(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint16_t> v(sz / 2);
    fread(v.data(), 2, v.size(), f);
    fclose(f);
    return v;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t man  = (h & 0x03FF);
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; exp--; }
               man &= 0x3FF; bits = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out; memcpy(&out, &bits, 4); return out;
}

int main(int argc, char** argv) {
    const char* home = getenv("HOME");
    std::string bundle = argc > 1 ? argv[1] :
        std::string(home) + "/.rindi/aimodels/gdn_tail_with_ip_100_s32.aimodel";
    int preferANE = argc > 2 ? atoi(argv[2]) : 1;

    // shape args: rows cols [default old channel-major golden]
    long ROWS = argc > 3 ? atol(argv[3]) : 11264;
    long COLS = argc > 4 ? atol(argv[4]) : 32;
    std::string base = argc > 5 ? argv[5] : "/tmp/golden";
    auto x  = load_f16((base + "_x.f16").c_str());
    auto yr = load_f16((base + "_y.f16").c_str());
    auto y2r= load_f16((base + "_y2.f16").c_str());
    const size_t OUT_TOTAL = yr.size() + y2r.size();
    printf("[probe] input [%ld,%ld] out=%zu elems\n", ROWS, COLS, OUT_TOTAL);

    printf("[probe] bundle: %s  preferANE=%d\n", bundle.c_str(), preferANE);
    if (getenv("RINDI_TEST_LEGACY_CTX")) {
        ANEContext* ctx = ane_context_create();
        printf("[probe] legacy ANEContext created: %p\n", (void*)ctx);
    }
    void* h = rindi_ane_load(bundle.c_str(), preferANE);
    if (!h) {
        fprintf(stderr, "[probe] LOAD FAILED: %s\n", rindi_ane_last_error());
        return 1;
    }
    printf("[probe] loaded OK\n");

    // dry run to learn output count
    long written = rindi_ane_run(h, x.data(), ROWS, COLS, nullptr, 0);
    if (written != (long)OUT_TOTAL) {
        fprintf(stderr, "[probe] output count mismatch: got %ld want %zu\n",
                written, OUT_TOTAL);
    }
    std::vector<uint16_t> out(written > 0 ? (size_t)written : OUT_TOTAL);

    // correctness: 5 runs, all must match golden
    double max_rel = 0.0;
    for (int rep = 0; rep < 5; ++rep) {
        long n = rindi_ane_run(h, x.data(), ROWS, COLS, out.data(), (long)out.size());
        if (n < 0) { fprintf(stderr, "[probe] RUN FAILED: %s\n", rindi_ane_last_error()); return 2; }
        double mabs_y = 0, mabs_y2 = 0, scale_y = 0, scale_y2 = 0;
        for (size_t i = 0; i < yr.size(); ++i) {
            float g = f16_to_f32(out[i]), r = f16_to_f32(yr[i]);
            mabs_y = fmax(mabs_y, fabs(g - r));
            scale_y = fmax(scale_y, fabs(r));
        }
        for (size_t i = 0; i < y2r.size(); ++i) {
            float g = f16_to_f32(out[yr.size() + i]), r = f16_to_f32(y2r[i]);
            mabs_y2 = fmax(mabs_y2, fabs(g - r));
            scale_y2 = fmax(scale_y2, fabs(r));
        }
        double mrel_y = mabs_y / (scale_y + 1e-9);
        {FILE* df=fopen("/tmp/out_tct.f16","wb");fwrite(out.data(),2,out.size(),df);fclose(df);}
        double mrel_y2 = mabs_y2 / (scale_y2 + 1e-9);
        max_rel = fmax(max_rel, fmax(mrel_y, mrel_y2));
        if (rep == 0) {
            printf("[probe] cxx out[0..4] raw:");
            for (int i = 0; i < 4; ++i) printf(" %04x", out[i]);
            printf("  gold:");
            for (int i = 0; i < 4; ++i) printf(" %04x", yr[i]);
            double s = 0; for (size_t i = 0; i < yr.size(); ++i) s += f16_to_f32(out[i]);
            printf("  cxx y.sum=%.4f\n", s);
            printf("[probe] rel_err(y)=%.5f rel_err(y2)=%.5f\n", mrel_y, mrel_y2);
        }
    }
    bool ok = max_rel < 0.05;
    printf("[probe] %s (max_rel=%.5f over 5 reps)\n", ok ? "NUMERICALLY CORRECT" : "MISMATCH",
           max_rel);

    // benchmark
    using clk = std::chrono::steady_clock;
    for (int w = 0; w < 20; ++w) rindi_ane_run(h, x.data(), ROWS, COLS, out.data(), (long)out.size());
    const int N = 300;
    auto t0 = clk::now();
    for (int i = 0; i < N; ++i)
        rindi_ane_run(h, x.data(), ROWS, COLS, out.data(), (long)out.size());
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / N;
    // tail GEMM flops per chunk (S=32): out_proj + gate/up + down + ip_proj
    double flops = 2.0 * (6144 * 5120 + 5120 * 2 * 17408 + 17408 * 5120 + 5120 * 2048) * 32;
    printf("[probe] %.3f ms/chunk  -> %.1f TFLOPS (tail GEMMs)\n",
           ms, flops / ms / 1e9);
    printf("[probe] tokens/sec at S=32: %.0f\n", 32.0 / ms * 1000.0);

    rindi_ane_free(h);
    return ok ? 0 : 3;
}
