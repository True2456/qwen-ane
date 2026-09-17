#include "runtime/rindi_engine.h"
#include <cstdio>

int main() {
    RindiEngine engine("~/.lmstudio/models/Qwen/Qwen3.8-27B.rindi");
    if (!engine.is_ready()) return 2;
    std::string text = engine.generate("2+2", 1, 0.0f);
    std::printf("ENGINE_SAMPLING=%s text_bytes=%zu\n", text.empty() ? "FAIL" : "PASS", text.size());
    return text.empty() ? 1 : 0;
}
