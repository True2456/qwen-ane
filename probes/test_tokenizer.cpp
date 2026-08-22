#include "runtime/bpe_tokenizer.h"
#include <cstdio>

int main() {
    BPETokenizer tok;
    if (!tok.load("/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B/tokenizer.json")) return 2;
    const auto ids = tok.encode("Reply with exactly NATIVE_OK");
    std::printf("TOKENIZER_TOKENS=%zu\n", ids.size());
    for (int id : ids) std::printf("%d:%s\n", id, tok.decode(id).c_str());
    return ids.empty() ? 1 : 0;
}
