// SPDX-License-Identifier: Apache-2.0
// End-to-end native decode benchmark with externally controlled MTP.

#include "runtime/bpe_tokenizer.h"
#include "runtime/rindi_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>

namespace {

uint64_t fnv1a64(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

double epoch_seconds() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void power_marker(const char* edge, const std::string& phase) {
    std::printf("MEASURE_%s %s %.6f\n", edge, phase.c_str(), epoch_seconds());
    std::fflush(stdout);
}

std::string make_prompt(BPETokenizer& tokenizer, size_t target_tokens) {
    const std::string prefix =
        "<|im_start|>user\n"
        "Read the supplied context, then write a detailed technical explanation "
        "of how speculative decoding improves large-language-model inference. "
        "Discuss memory bandwidth, draft acceptance, verification batching, and "
        "latency. Write at least 400 words and do not finish early.\n\nContext:\n";
    const std::string paragraph =
        "Autoregressive inference repeatedly reads model weights, updates caches, "
        "and selects a token. Performance depends on useful memory bandwidth, "
        "kernel launch overhead, batch width, and the fraction of proposed tokens "
        "accepted by the target model.\n";
    const std::string suffix =
        "\nNow provide the requested explanation.<|im_end|>\n"
        "<|im_start|>assistant\n";

    std::string body;
    std::string prompt = prefix + suffix;
    // Leave room for one-token padding below. Keeping the prose coherent makes
    // early EOS much less likely than a prompt made entirely from repeated junk.
    while (tokenizer.encode(prompt).size() + 64 < target_tokens) {
        body += paragraph;
        prompt = prefix + body + suffix;
    }

    std::string best = prompt;
    size_t best_distance = best.empty()
        ? target_tokens
        : static_cast<size_t>(std::llabs(
              static_cast<long long>(tokenizer.encode(best).size()) -
              static_cast<long long>(target_tokens)));
    while (tokenizer.encode(prompt).size() < target_tokens) {
        prompt.insert(prompt.size() - suffix.size(), " context");
        const size_t count = tokenizer.encode(prompt).size();
        const size_t distance = count > target_tokens
            ? count - target_tokens : target_tokens - count;
        if (distance < best_distance) {
            best = prompt;
            best_distance = distance;
        }
        if (count >= target_tokens) break;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model = argc > 1
        ? argv[1]
        : "~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    const int target_prompt_tokens = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int max_tokens = argc > 3 ? std::atoi(argv[3]) : 128;
    if (target_prompt_tokens < 64 || max_tokens < 2) {
        std::fprintf(stderr, "prompt_tokens must be >= 64 and max_tokens >= 2\n");
        return 2;
    }

    setenv("RINDI_TAIL_COREAI", "1", 0);
    setenv("RINDI_ENABLE_METAL_TAIL", "1", 0);

    BPETokenizer tokenizer;
    if (!tokenizer.load(model + "/tokenizer.json")) {
        std::fprintf(stderr, "failed to load tokenizer from %s\n", model.c_str());
        return 2;
    }
    const std::string prompt = make_prompt(
        tokenizer, static_cast<size_t>(target_prompt_tokens));
    const size_t measured_prompt_tokens = tokenizer.encode(prompt).size();

    RindiEngine engine(model);
    if (!engine.is_ready()) {
        std::fprintf(stderr, "failed to load model: %s\n", model.c_str());
        return 2;
    }

    const bool power_markers = std::getenv("RINDI_POWER_MARKERS") != nullptr;
    const char* width_env = std::getenv("RINDI_ANE_WIDTH");
    const std::string width = width_env && *width_env ? width_env : "32";
    const std::string prefill_phase = "prefill:w" + width;
    const std::string decode_phase = "decode:w" + width;
    bool decode_started = false;
    std::function<void(const std::string&)> stream_cb = nullptr;
    if (power_markers) {
        power_marker("START", prefill_phase);
        stream_cb = [&](const std::string&) {
            if (!decode_started) {
                power_marker("END", prefill_phase);
                power_marker("START", decode_phase);
                decode_started = true;
            }
        };
    }

    const std::string output = engine.generate(
        prompt, max_tokens, 0.0f, std::move(stream_cb));
    if (power_markers) {
        if (!decode_started) {
            power_marker("END", prefill_phase);
        } else {
            power_marker("END", decode_phase);
        }
    }
    const GenerationStats& stats = engine.get_last_stats();
    const bool mtp_requested = std::getenv("RINDI_DISABLE_MTP") == nullptr;
    const double tpot_ms = stats.generated_tokens > 1
        ? stats.decode_ms / static_cast<double>(stats.generated_tokens - 1)
        : 0.0;
    const double total_throughput = stats.total_ms > 0.0
        ? static_cast<double>(stats.prompt_tokens + stats.generated_tokens) *
              1000.0 / stats.total_ms
        : 0.0;

    std::printf(
        "NATIVE_MTP_BENCH requested=%s spec_used=%s head=%s target_prompt_tokens=%d "
        "pretokenized_prompt_tokens=%zu prompt_tokens=%zu generated_tokens=%zu "
        "ttft_ms=%.3f tpot_ms=%.3f ppTPS=%.3f tgTPS=%.3f e2e_s=%.3f "
        "throughput=%.3f spec_steps=%zu confirmed_per_step=%.3f "
        "output_bytes=%zu output_fnv1a64=%016llx\n",
        mtp_requested ? "mtp" : "dense",
        stats.spec_used ? "yes" : "no",
        std::getenv("RINDI_INT4_LM_HEAD") ? "int4" : "bf16",
        target_prompt_tokens, measured_prompt_tokens, stats.prompt_tokens,
        stats.generated_tokens, stats.ttft_ms, tpot_ms, stats.prefill_tps(),
        stats.decode_tps(), stats.total_ms / 1000.0, total_throughput,
        stats.spec_steps, stats.accepted_per_step, output.size(),
        static_cast<unsigned long long>(fnv1a64(output)));
    std::printf("--- OUTPUT ---\n%s\n--------------\n", output.c_str());

    return output.empty() || stats.prompt_tokens != measured_prompt_tokens ||
            stats.generated_tokens < 2
        ? 1 : 0;
}
