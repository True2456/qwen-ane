// test_coreai_multitail.cpp — P21: load many layer bundles through the Swift
// CoreAI shim, measure cold/warm load times, then evaluate all tails
// sequentially to measure sustained multi-bundle ANE throughput.
//
//   usage: multitail <bundleDir> <nBundles> [preferANE] [S]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <dirent.h>
#include <algorithm>
#include <dlfcn.h>

extern "C" {
void* rindi_ane_load(const char* path, int preferANE);
long  rindi_ane_run(void* h, const uint16_t* xin, long rows, long cols,
                    uint16_t* outs, long outCap);
void  rindi_ane_free(void* h);
const char* rindi_ane_last_error(void);
}

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const char* dir = argc > 1 ? argv[1] : nullptr;
    int maxN = argc > 2 ? atoi(argv[2]) : 64;
    int preferANE = argc > 3 ? atoi(argv[3]) : 1;
    const long S = argc > 4 ? atol(argv[4]) : 32;
    if (!dir) { fprintf(stderr, "need bundle dir\n"); return 1; }

    // discover bundles sorted by layer id
    std::vector<std::string> paths;
    if (DIR* d = opendir(dir)) {
        while (dirent* e = readdir(d)) {
            std::string n = e->d_name;
            if (n.find("qwen38_27b_tail_") == 0 && n.find(".aimodel") != std::string::npos)
                paths.push_back(std::string(dir) + "/" + n);
        }
        closedir(d);
    }
    std::sort(paths.begin(), paths.end());
    if ((int)paths.size() > maxN) paths.resize(maxN);
    printf("[multi] %zu bundles from %s\n", paths.size(), dir);
    if (paths.empty()) return 1;

    const long ROWS = S, COLS = 5120 + 6144;      // token-major tail input
    std::vector<uint16_t> xin(ROWS * COLS, 0x3800); // arbitrary pattern (0.5)
    const size_t OUT = ROWS * 5120;
    std::vector<uint16_t> out(OUT);

    // ---- load phase
    double total_load = 0;
    std::vector<void*> handles;
    handles.reserve(paths.size());
    for (size_t i = 0; i < paths.size(); ++i) {
        auto t0 = clk::now();
        void* h = rindi_ane_load(paths[i].c_str(), preferANE);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (!h) {
            fprintf(stderr, "[multi] LOAD FAIL %s: %s\n",
                    paths[i].c_str(), rindi_ane_last_error());
            return 2;
        }
        handles.push_back(h);
        total_load += ms;
        if (i < 3 || i == paths.size() - 1)
            printf("[multi] loaded %zu in %.0f ms\n", i + 1, ms);
    }
    printf("[multi] total load: %.1f s (avg %.0f ms/bundle)\n",
           total_load / 1000.0, total_load / paths.size());

    // resident memory probe
    FILE* sf = fopen("/proc/self/statm", "r");
    if (sf) { long rss = 0; fscanf(sf, "%ld", &rss); fclose(sf);
              printf("[multi] RSS ~%.1f GB\n", rss * 4096 / 1e9); }

    // ---- eval phase: one pass over all layers = one "token step" of tails
    for (int w = 0; w < 5; ++w)
        for (void* h : handles) rindi_ane_run(h, xin.data(), ROWS, COLS, out.data(), OUT);

    const int REPS = 30;
    auto t0 = clk::now();
    for (int r = 0; r < REPS; ++r)
        for (void* h : handles) {
            long n = rindi_ane_run(h, xin.data(), ROWS, COLS, out.data(), OUT);
            if (n < 0) { fprintf(stderr, "[multi] RUN FAIL: %s\n", rindi_ane_last_error()); return 3; }
        }
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count()
                / (REPS * (double)handles.size());
    printf("[multi] per-layer tail: %.3f ms | full 64-layer tail pass: %.1f ms "
           "| tails tok/s @S=%ld: %.0f\n",
           ms, ms * paths.size(), S, S / (ms / 1000.0));

    for (void* h : handles) rindi_ane_free(h);
    return 0;
}
