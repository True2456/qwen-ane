// SPDX-License-Identifier: Apache-2.0
// Teacher-forced native quality gate for scalar versus fast Qwen prefill.

#include "runtime/rindi_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    const std::string model = argc > 1
        ? argv[1]
        : "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL TEXT_FILE [MAX_TOKENS]\n", argv[0]);
        return 2;
    }
    const size_t max_tokens = argc > 3
        ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10)) : 256;
    std::ifstream stream(argv[2], std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "cannot open evaluation text: %s\n", argv[2]);
        return 2;
    }
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    setenv("RINDI_DISABLE_MTP", "1", 0);
    setenv("RINDI_TAIL_COREAI", "1", 0);
    setenv("RINDI_ENABLE_METAL_TAIL", "1", 0);
    setenv("RINDI_PREFILL_BATCH_ATTENTION", "1", 0);
    setenv("RINDI_ANE_WIDTH", "128", 0);

    RindiEngine engine(model);
    if (!engine.is_ready()) return 3;
    double nll = 0.0;
    size_t scored = 0;
    if (!engine.score_text(text, max_tokens, nll, scored)) {
        std::fprintf(stderr, "native teacher-forced scoring failed\n");
        return 4;
    }
    const bool fast = std::getenv("RINDI_QWEN_PREFILL_FAST") != nullptr;
    const bool int4_head = std::getenv("RINDI_INT4_LM_HEAD") != nullptr;
    std::printf("NATIVE_PPL mode=%s head=%s tokens=%zu nll=%.9f ppl=%.9f\n",
                fast ? "fast" : "scalar", int4_head ? "int4" : "bf16",
                scored, nll, std::exp(nll));
    return 0;
}
