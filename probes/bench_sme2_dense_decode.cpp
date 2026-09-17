#include "runtime/rindi_engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

uint64_t fnv1a64(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model = argc > 1
        ? argv[1]
        : "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    const int max_tokens = argc > 2 ? std::atoi(argv[2]) : 16;
    if (max_tokens < 2) {
        std::fprintf(stderr, "max_tokens must be at least 2\n");
        return 2;
    }

    // This benchmark isolates ordinary single-token decode. MTP changes the
    // number and width of tail evaluations and would obscure the SME2 split.
    setenv("RINDI_DISABLE_MTP", "1", 1);
    setenv("RINDI_TAIL_COREAI", "1", 0);

    RindiEngine engine(model);
    if (!engine.is_ready()) {
        std::fprintf(stderr, "failed to load model: %s\n", model);
        return 2;
    }

    const std::string prompt =
        "<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n"
        "<|im_start|>assistant\n";
    const std::string output = engine.generate(prompt, max_tokens, 0.0f);
    const GenerationStats& stats = engine.get_last_stats();
    const bool sme2_down = std::getenv("RINDI_SME2_DOWN") != nullptr;
    const bool sme2_split = std::getenv("RINDI_SME2_DOWN_SPLIT") != nullptr;

    std::printf(
        "SME2_DENSE_DECODE mode=%s prompt_tokens=%zu generated_tokens=%zu "
        "prefill_ms=%.3f ttft_ms=%.3f decode_ms=%.3f decode_tok_s=%.3f "
        "total_ms=%.3f output_bytes=%zu output_fnv1a64=%016llx\n",
        sme2_split ? "metal_sme2_row_split" :
            (sme2_down ? "metal_gateup_sme2_down" : "metal"),
        stats.prompt_tokens, stats.generated_tokens, stats.prefill_ms,
        stats.ttft_ms, stats.decode_ms, stats.decode_tps(), stats.total_ms,
        output.size(), static_cast<unsigned long long>(fnv1a64(output)));
    std::printf("--- OUTPUT ---\n%s\n--------------\n", output.c_str());
    return output.empty() || stats.generated_tokens < 2 ? 1 : 0;
}
