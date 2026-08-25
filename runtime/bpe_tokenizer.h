/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/bpe_tokenizer.h - Fast Native C++ BPE Tokenizer for Qwen3.8 / Qwen Models.
 */

#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <utility>

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    bool load(const std::string& tokenizer_json_path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        const std::string& tools_json = "",
        bool enable_thinking = true,
        const std::string& reasoning_effort = "xhigh"
    ) const;

    int bos_token_id() const { return bos_id_; }
    int eos_token_id() const { return eos_id_; }
    int im_start_id() const { return im_start_id_; }
    int im_end_id() const { return im_end_id_; }
    int think_start_id() const { return think_start_id_; }
    int think_end_id() const { return think_end_id_; }

    bool is_stop_token(int id) const {
        return id == eos_id_ || id == im_end_id_ || id == 151645 || id == 151643 ||
               id == 248046 || id == 248044;
    }

    size_t vocab_size() const { return vocab_.size(); }

private:
    std::unordered_map<std::string, int> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> merges_;
    std::unordered_map<uint8_t, std::string> byte_to_unicode_;
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    int bos_id_{248044};
    int eos_id_{248044};
    int im_start_id_{248045};
    int im_end_id_{248046};
    int think_start_id_{248068};
    int think_end_id_{248069};

    void init_byte_mappings();
    std::vector<std::string> bpe(const std::string& token) const;
};

#endif // BPE_TOKENIZER_H
