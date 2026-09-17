#include "runtime/rindi_engine.h"
#include <cstdio>

int main() {
    RindiEngine engine("~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi");
    if (!engine.is_ready()) return 2;
    const std::string prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";
    const std::string text = engine.generate(prompt, 32, 0.7f);
    std::printf("NATIVE_GENERATE=%s bytes=%zu\n--- OUTPUT ---\n%s\n--------------\n",
                text.empty() ? "FAIL" : "PASS", text.size(), text.c_str());
    return text.empty() ? 1 : 0;
}
