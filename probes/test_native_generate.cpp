#include "runtime/rindi_engine.h"
#include <cstdio>

int main() {
    RindiEngine engine("/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi");
    if (!engine.is_ready()) return 2;
    const std::string text = engine.generate("Explain what ANE and GPU are in 2 sentences.", 8, 0.0f);
    std::printf("NATIVE_GENERATE=%s bytes=%zu text=%s\n",
                text.empty() ? "FAIL" : "PASS", text.size(), text.c_str());
    return text.empty() ? 1 : 0;
}
