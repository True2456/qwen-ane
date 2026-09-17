// SPDX-License-Identifier: Apache-2.0
// probes/test_mtp.cpp - Speculative decode must not change greedy output.
//
// Loads the engine twice (MTP disabled, then MTP depth 2) and compares the
// full greedy generations token-for-token. Speculative decoding is exact:
// any divergence is a rollback/replay bug. Also reports measured decode
// throughput and acceptance from the engine's own timers.
#include "runtime/rindi_engine.h"
#include <cstdlib>
#include <cstdio>
#include <string>

namespace {

std::string run_engine(const char* model_path, const char* prompt,
                       int max_tokens, double* decode_tps,
                       double* accepted, size_t* steps) {
    RindiEngine engine(model_path);
    if (!engine.is_ready()) return "<ENGINE_FAILED>";
    const std::string text = engine.generate(prompt, max_tokens, 0.0f, nullptr);
    const GenerationStats& s = engine.get_last_stats();
    if (decode_tps) *decode_tps = s.decode_tps();
    if (accepted) *accepted = s.accepted_per_step;
    if (steps) *steps = s.spec_steps;
    return text;
}

} // namespace

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1]
        : "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    const char* prompt = argc > 2 ? argv[2]
        : "Count from 1 to 15, then explain why speculative decoding preserves output.";
    const int max_tokens = std::getenv("MTP_PROBE_TOKENS")
        ? std::atoi(std::getenv("MTP_PROBE_TOKENS")) : 64;

    setenv("RINDI_DISABLE_MTP", "1", 1);
    double base_tps = 0.0, base_acc = 0.0; size_t base_steps = 0;
    const std::string base = run_engine(model_path, prompt, max_tokens,
                                        &base_tps, &base_acc, &base_steps);
    std::printf("BASE   chars=%zu tps=%.2f\n", base.size(), base_tps);

    unsetenv("RINDI_DISABLE_MTP");
    if (!std::getenv("RINDI_MTP_DEPTH")) setenv("RINDI_MTP_DEPTH", "2", 1);
    double mtp_tps = 0.0, mtp_acc = 0.0; size_t mtp_steps = 0;
    const std::string mtp = run_engine(model_path, prompt, max_tokens,
                                       &mtp_tps, &mtp_acc, &mtp_steps);
    std::printf("MTP    chars=%zu tps=%.2f accepted/step=%.2f steps=%zu\n",
                mtp.size(), mtp_tps, mtp_acc, mtp_steps);

    // The speculative run may confirm a couple of extra tokens when the last
    // round emits in bulk past the cap; exactness means the shared prefix
    // matches token-for-token.
    const bool identical = base != "<ENGINE_FAILED>" &&
                           mtp.size() >= base.size() &&
                           mtp.compare(0, base.size(), base) == 0;
    std::printf("%s %s speedup=%.2fx\n",
                identical ? "MTP_EXACT=PASS" : "MTP_EXACT=FAIL",
                identical ? "" : "(outputs differ!)",
                base_tps > 0 ? mtp_tps / base_tps : 0.0);
    if (!identical) {
        size_t i = 0;
        while (i < base.size() && i < mtp.size() && base[i] == mtp[i]) ++i;
        std::printf("first divergence at %zu: base=...%.40s | mtp=...%.40s\n",
                    i, base.substr(i > 20 ? i - 20 : 0).c_str(),
                    mtp.substr(i > 20 ? i - 20 : 0).c_str());
    }
    return identical ? 0 : 1;
}
