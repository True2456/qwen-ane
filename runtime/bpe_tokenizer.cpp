/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/bpe_tokenizer.cpp - Implementation of Fast Native C++ BPE Tokenizer.
 */

#include "bpe_tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cstring>

namespace {
struct Utf8Char {
    uint32_t cp;
    size_t bytes;
};

Utf8Char decode_utf8_at(const std::string& text, size_t pos) {
    const unsigned char c = static_cast<unsigned char>(text[pos]);
    if (c < 0x80) return {c, 1};
    if ((c & 0xe0) == 0xc0 && pos + 1 < text.size()) {
        return {static_cast<uint32_t>(c & 0x1f) << 6 |
                    (static_cast<unsigned char>(text[pos + 1]) & 0x3f), 2};
    }
    if ((c & 0xf0) == 0xe0 && pos + 2 < text.size()) {
        return {static_cast<uint32_t>(c & 0x0f) << 12 |
                    (static_cast<unsigned char>(text[pos + 1]) & 0x3f) << 6 |
                    (static_cast<unsigned char>(text[pos + 2]) & 0x3f), 3};
    }
    if ((c & 0xf8) == 0xf0 && pos + 3 < text.size()) {
        return {static_cast<uint32_t>(c & 0x07) << 18 |
                    (static_cast<unsigned char>(text[pos + 1]) & 0x3f) << 12 |
                    (static_cast<unsigned char>(text[pos + 2]) & 0x3f) << 6 |
                    (static_cast<unsigned char>(text[pos + 3]) & 0x3f), 4};
    }
    return {c, 1};
}

bool is_unicode_letter(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
           (cp >= 0x00c0 && cp <= 0x02af) || (cp >= 0x0370 && cp <= 0x052f) ||
           (cp >= 0x1e00 && cp <= 0x1eff) || (cp >= 0x3040 && cp <= 0x30ff) ||
           (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) ||
           (cp >= 0xac00 && cp <= 0xd7af);
}

bool is_unicode_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036f) ||
           (cp >= 0x1ab0 && cp <= 0x1aff) ||
           (cp >= 0x1dc0 && cp <= 0x1dff) ||
           (cp >= 0x20d0 && cp <= 0x20ff) ||
           (cp >= 0xfe20 && cp <= 0xfe2f);
}

bool is_unicode_number(uint32_t cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06f0 && cp <= 0x06f9) ||
           (cp >= 0x0966 && cp <= 0x096f) || (cp >= 0xff10 && cp <= 0xff19);
}

bool is_unicode_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x00a0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 ||
           cp == 0x202f || cp == 0x205f || cp == 0x3000;
}
}

static std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

static std::string json_unescape_str(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '\\' && i + 1 < input.size()) {
            char next = input[i + 1];
            if (next == '"') { out += '"'; i++; }
            else if (next == '\\') { out += '\\'; i++; }
            else if (next == '/') { out += '/'; i++; }
            else if (next == 'b') { out += '\b'; i++; }
            else if (next == 'f') { out += '\f'; i++; }
            else if (next == 'n') { out += '\n'; i++; }
            else if (next == 'r') { out += '\r'; i++; }
            else if (next == 't') { out += '\t'; i++; }
            else if (next == 'u' && i + 5 < input.size()) {
                std::string hex_str = input.substr(i + 2, 4);
                try {
                    unsigned int cp = std::stoul(hex_str, nullptr, 16);
                    out += utf8_encode(cp);
                } catch (...) {
                    out += "?";
                }
                i += 5;
            } else {
                out += next;
                i++;
            }
        } else {
            out += input[i];
        }
    }
    return out;
}

BPETokenizer::BPETokenizer() {
    init_byte_mappings();
}

BPETokenizer::~BPETokenizer() {}

void BPETokenizer::init_byte_mappings() {
    std::vector<int> bs;
    for (int i = L'!'; i <= L'~'; ++i) bs.push_back(i);
    for (int i = L'¡'; i <= L'¬'; ++i) bs.push_back(i);
    for (int i = L'®'; i <= L'ÿ'; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        std::string u_str = utf8_encode(static_cast<uint32_t>(cs[i]));
        byte_to_unicode_[byte_val] = u_str;
        unicode_to_byte_[u_str] = byte_val;
    }
}

bool BPETokenizer::load(const std::string& tokenizer_json_path) {
    std::ifstream file(tokenizer_json_path);
    if (!file.is_open()) {
        std::cerr << "[BPETokenizer] Failed to open " << tokenizer_json_path << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // 1. Find "vocab" object
    size_t vocab_pos = content.find("\"vocab\":");
    if (vocab_pos != std::string::npos) {
        size_t open_brace = content.find('{', vocab_pos);
        if (open_brace != std::string::npos) {
            size_t p = open_brace + 1;
            while (p < content.size()) {
                size_t q_start = content.find('"', p);
                if (q_start == std::string::npos) break;

                size_t q_end = q_start + 1;
                while (q_end < content.size()) {
                    if (content[q_end] == '"') {
                        size_t bs = 0;
                        size_t k = q_end;
                        while (k > q_start && content[k - 1] == '\\') {
                            bs++;
                            k--;
                        }
                        if (bs % 2 == 0) break;
                    }
                    q_end++;
                }
                if (q_end >= content.size()) break;

                std::string raw_key = content.substr(q_start + 1, q_end - q_start - 1);
                std::string key = json_unescape_str(raw_key);
                size_t col = content.find(':', q_end + 1);
                if (col == std::string::npos) break;

                size_t num_start = content.find_first_of("0123456789", col);
                if (num_start == std::string::npos) break;
                size_t num_end = content.find_first_not_of("0123456789", num_start);
                int val = std::stoi(content.substr(num_start, num_end - num_start));

                vocab_[key] = val;
                if ((size_t)val >= id_to_token_.size()) {
                    id_to_token_.resize(val + 1);
                }
                id_to_token_[val] = key;

                size_t next_comma = content.find_first_of(",}", num_end);
                if (next_comma != std::string::npos && content[next_comma] == '}') {
                    break;
                }
                p = (next_comma != std::string::npos) ? next_comma + 1 : content.size();
            }
        }
    }

    // 2. Find "merges" array
    size_t merges_pos = content.find("\"merges\":");
    if (merges_pos != std::string::npos) {
        size_t open_bracket = content.find('[', merges_pos);
        if (open_bracket != std::string::npos) {
            size_t p = open_bracket + 1;
            int rank = 0;
            while (p < content.size()) {
                size_t q_start = content.find('"', p);
                if (q_start == std::string::npos) break;

                size_t q_end = q_start + 1;
                while (q_end < content.size()) {
                    if (content[q_end] == '"') {
                        size_t bs = 0;
                        size_t k = q_end;
                        while (k > q_start && content[k - 1] == '\\') {
                            bs++;
                            k--;
                        }
                        if (bs % 2 == 0) break;
                    }
                    q_end++;
                }
                if (q_end >= content.size()) break;

                std::string raw_merge = content.substr(q_start + 1, q_end - q_start - 1);
                std::string merge_line = json_unescape_str(raw_merge);
                merges_[merge_line] = rank++;

                size_t next_comma = content.find_first_of(",]", q_end + 1);
                if (next_comma != std::string::npos && content[next_comma] == ']') {
                    break;
                }
                p = (next_comma != std::string::npos) ? next_comma + 1 : content.size();
            }
        }
    }

    // Hugging Face's tokenizer JSON stores Qwen's control tokens in the
    // separate added_tokens array rather than in model.vocab.  They must be
    // registered as atomic tokens; otherwise the chat markers are BPE-split
    // into '<', '|', 'im', ... and the model sees a completely different
    // sequence.
    size_t added_pos = content.find("\"added_tokens\":");
    if (added_pos != std::string::npos) {
        size_t p = content.find('[', added_pos);
        while (p != std::string::npos && p < content.size() && content[p] != ']') {
            size_t object_end = content.find('}', p);
            if (object_end == std::string::npos) break;
            size_t content_key = content.find("\"content\"", p);
            if (content_key == std::string::npos || content_key > object_end) {
                p = object_end + 1;
                continue;
            }
            size_t content_colon = content.find(':', content_key + 9);
            if (content_colon == std::string::npos || content_colon > object_end) {
                p = object_end + 1;
                continue;
            }
            size_t raw_start = content.find('"', content_colon + 1);
            if (raw_start == std::string::npos || raw_start > object_end) {
                p = object_end + 1;
                continue;
            }
            ++raw_start;
            size_t raw_end = raw_start;
            while (raw_end < object_end) {
                if (content[raw_end] == '"') {
                    size_t backslashes = 0;
                    for (size_t q = raw_end; q > raw_start && content[q - 1] == '\\'; --q) {
                        ++backslashes;
                    }
                    if ((backslashes & 1u) == 0) break;
                }
                ++raw_end;
            }
            if (raw_end >= object_end) break;
            // In tokenizer.json the id field precedes content for these
            // objects, so search the complete object rather than only the
            // suffix after the content string.
            size_t id_key = content.find("\"id\":", p);
            if (id_key != std::string::npos && id_key < object_end) {
                size_t id_start = content.find_first_of("0123456789", id_key);
                if (id_start != std::string::npos && id_start < object_end) {
                    size_t id_end = content.find_first_not_of("0123456789", id_start);
                    const std::string token = json_unescape_str(
                        content.substr(raw_start, raw_end - raw_start));
                    const int id = std::stoi(content.substr(id_start, id_end - id_start));
                    vocab_[token] = id;
                    if (static_cast<size_t>(id) >= id_to_token_.size()) id_to_token_.resize(id + 1);
                    id_to_token_[id] = token;
                }
            }
            p = content.find('{', object_end + 1);
        }
    }

    // Special token lookups
    if (vocab_.count("<|im_start|>")) im_start_id_ = vocab_["<|im_start|>"];
    if (vocab_.count("<|im_end|>")) im_end_id_ = vocab_["<|im_end|>"];
    if (vocab_.count("<|endoftext|>")) eos_id_ = vocab_["<|endoftext|>"];
    if (vocab_.count("<think>")) think_start_id_ = vocab_["<think>"];
    if (vocab_.count("</think>")) think_end_id_ = vocab_["</think>"];

    std::cout << "[BPETokenizer] Loaded " << vocab_.size() << " tokens and " << merges_.size() << " merges." << std::endl;
    return true;
}

std::vector<std::string> BPETokenizer::bpe(const std::string& token) const {
    if (token.empty()) return {};
    std::vector<std::string> word;
    for (char c : token) {
        word.push_back(byte_to_unicode_.at(static_cast<uint8_t>(c)));
    }
    if (word.size() <= 1) return word;

    while (true) {
        int min_rank = 1e9;
        int min_idx = -1;
        for (size_t i = 0; i < word.size() - 1; ++i) {
            std::string pair_key = word[i] + " " + word[i + 1];
            auto it = merges_.find(pair_key);
            if (it != merges_.end() && it->second < min_rank) {
                min_rank = it->second;
                min_idx = i;
            }
        }
        if (min_idx == -1) break;

        std::string merged = word[min_idx] + word[min_idx + 1];
        word[min_idx] = merged;
        word.erase(word.begin() + min_idx + 1);
        if (word.size() <= 1) break;
    }
    return word;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    if (text.empty()) return tokens;

    auto emit_piece = [&](const std::string& piece) {
        for (const auto& sw : bpe(piece)) {
            auto it = vocab_.find(sw);
            if (it != vocab_.end()) tokens.push_back(it->second);
        }
    };

    // Match the tokenizer.json Sequence(Split(regex, Isolated), ByteLevel)
    // pre-tokenizer. In particular, the optional non-letter prefix in the
    // word rule keeps a leading space attached to the word, which is encoded
    // as the GPT-2/ByteLevel marker 'Ġ'.
    auto emit_chunk = [&](const std::string& chunk) {
        size_t p = 0;
        while (p < chunk.size()) {
            const size_t start = p;
            const Utf8Char first = decode_utf8_at(chunk, p);

            // Contractions are an explicit first alternative in the JSON
            // regex and must remain one pre-tokenized piece.
            if (first.cp == '\'' && p + 1 < chunk.size()) {
                const std::string rest = chunk.substr(p + 1);
                const char* suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
                for (const char* suffix : suffixes) {
                    const size_t n = std::strlen(suffix);
                    if (rest.size() >= n &&
                        std::tolower(static_cast<unsigned char>(rest[0])) == suffix[0] &&
                        (n == 1 || rest.compare(0, n, suffix) == 0)) {
                        p += 1 + n;
                        emit_piece(chunk.substr(start, p - start));
                        goto next_piece;
                    }
                }
            }

            // Optional one-character prefix followed by letters/marks. This
            // is what makes " with" one piece rather than " " + "with".
            {
                size_t q = p;
                if (!is_unicode_letter(first.cp) && !is_unicode_mark(first.cp) &&
                    !is_unicode_number(first.cp) && first.cp != '\r' && first.cp != '\n') {
                    q += first.bytes;
                }
                const size_t letters_start = q;
                while (q < chunk.size()) {
                    const Utf8Char c = decode_utf8_at(chunk, q);
                    if (!is_unicode_letter(c.cp) && !is_unicode_mark(c.cp)) break;
                    q += c.bytes;
                }
                if (q > letters_start) {
                    p = q;
                    emit_piece(chunk.substr(start, p - start));
                    goto next_piece;
                }
            }

            // Numbers are isolated one code point at a time by the source
            // regex.
            if (is_unicode_number(first.cp)) {
                p += first.bytes;
                emit_piece(chunk.substr(start, p - start));
                goto next_piece;
            }

            // Punctuation runs may have one leading space and trailing CR/LF.
            {
                size_t q = p;
                if (first.cp == ' ') q += first.bytes;
                const size_t punct_start = q;
                while (q < chunk.size()) {
                    const Utf8Char c = decode_utf8_at(chunk, q);
                    if (is_unicode_space(c.cp) || is_unicode_letter(c.cp) ||
                        is_unicode_mark(c.cp) || is_unicode_number(c.cp)) break;
                    q += c.bytes;
                }
                if (q > punct_start) {
                    while (q < chunk.size()) {
                        const Utf8Char c = decode_utf8_at(chunk, q);
                        if (c.cp != '\r' && c.cp != '\n') break;
                        q += c.bytes;
                    }
                    p = q;
                    emit_piece(chunk.substr(start, p - start));
                    goto next_piece;
                }
            }

            // Whitespace/newline alternatives. Keep each run intact so BPE
            // can merge repeated spaces/newlines exactly as the Rust tokenizer.
            if (is_unicode_space(first.cp)) {
                p += first.bytes;
                while (p < chunk.size()) {
                    const Utf8Char c = decode_utf8_at(chunk, p);
                    if (!is_unicode_space(c.cp)) break;
                    p += c.bytes;
                }
                emit_piece(chunk.substr(start, p - start));
                goto next_piece;
            }

            p += first.bytes;
            emit_piece(chunk.substr(start, p - start));
        next_piece:
            ;
        }
    };

    size_t i = 0;
    while (i < text.size()) {
        bool found_special = false;
        if (text[i] == '<') {
            for (const auto& sp : {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>"}) {
                const size_t len = std::strlen(sp);
                if (i + len <= text.size() && text.compare(i, len, sp) == 0 && vocab_.count(sp)) {
                    tokens.push_back(vocab_.at(sp));
                    i += len;
                    found_special = true;
                    break;
                }
            }
        }
        if (found_special) continue;

        size_t next_special = text.find('<', i + 1);
        if (next_special == std::string::npos) next_special = text.size();
        emit_chunk(text.substr(i, next_special - i));
        i = next_special;
    }
    return tokens;
}

std::string BPETokenizer::decode(int token_id) const {
    if (token_id < 0 || (size_t)token_id >= id_to_token_.size()) return "";
    const std::string& token_str = id_to_token_[token_id];
    if (token_str.empty()) return "";

    std::string out;
    size_t i = 0;
    while (i < token_str.size()) {
        bool matched = false;
        for (size_t len = 4; len >= 1; --len) {
            if (i + len <= token_str.size()) {
                std::string sub = token_str.substr(i, len);
                auto it = unicode_to_byte_.find(sub);
                if (it != unicode_to_byte_.end()) {
                    out += static_cast<char>(it->second);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            out += token_str[i];
            i++;
        }
    }
    return out;
}

std::string BPETokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int tid : token_ids) {
        result += decode(tid);
    }
    return result;
}

std::string BPETokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& tools_json,
    bool enable_thinking,
    const std::string& reasoning_effort
) const {
    std::string out;
    bool has_system = false;

    // Keep this in sync with the checkpoint's chat_template.jinja.  In
    // particular, Qwen3.5 puts the reasoning instruction in a synthetic
    // system message when the caller does not provide one; using the generic
    // "You are a helpful assistant" fallback changes the prefill sequence
    // and therefore changes the first generated token.
    std::string reasoning_instruction;
    if (enable_thinking) {
        if (reasoning_effort == "xhigh") {
            reasoning_instruction =
                "Reasoning effort is set to xhigh. Please think carefully through "
                "the task, validate key assumptions, consider plausible alternatives, "
                "and prioritize correctness, consistency, and clarity in the final answer.";
        } else if (reasoning_effort == "low") {
            reasoning_instruction =
                "Reasoning effort is set to low. Keep your thinking brief and focused, "
                "moving directly to the conclusion without unnecessary elaboration.";
        }
    }

    for (const auto& msg : messages) {
        if (msg.first == "system") {
            has_system = true;
            break;
        }
    }

    std::string tool_instructions;
    if (!tools_json.empty() && tools_json != "[]") {
        tool_instructions = "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
            tools_json + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "</IMPORTANT>";
    }

    if (has_system) {
        for (const auto& msg : messages) {
            if (msg.first != "system") break;
            std::string system_content = msg.second;
            while (!system_content.empty() &&
                   (system_content.back() == ' ' || system_content.back() == '\n' ||
                    system_content.back() == '\r' || system_content.back() == '\t')) {
                system_content.pop_back();
            }
            std::string header = reasoning_instruction;
            if (!tool_instructions.empty()) {
                header = header.empty() ? tool_instructions : header + "\n\n" + tool_instructions;
            }
            if (!header.empty() && !system_content.empty()) {
                out += "<|im_start|>system\n" + header + "\n\n" + system_content + "<|im_end|>\n";
            } else if (!header.empty()) {
                out += "<|im_start|>system\n" + header + "<|im_end|>\n";
            } else if (!system_content.empty()) {
                out += "<|im_start|>system\n" + system_content + "<|im_end|>\n";
            }
            break;
        }
    } else {
        std::string header = reasoning_instruction;
        if (!tool_instructions.empty()) {
            header = header.empty() ? tool_instructions : header + "\n\n" + tool_instructions;
        }
        if (!header.empty()) {
            out += "<|im_start|>system\n" + header + "<|im_end|>\n";
        }
    }

    for (const auto& msg : messages) {
        if (msg.first == "system") continue;
        if (msg.first == "tool") {
            out += "<|im_start|>user\n<tool_response>\n" + msg.second + "\n</tool_response><|im_end|>\n";
        } else {
            out += "<|im_start|>" + msg.first + "\n" + msg.second + "<|im_end|>\n";
        }
    }

    out += "<|im_start|>assistant\n";
    if (enable_thinking) {
        out += "<think>\n";
    } else {
        out += "<think>\n\n</think>\n\n";
    }
    return out;
}
