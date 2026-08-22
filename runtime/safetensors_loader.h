/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/safetensors_loader.h - Zero-Copy Memory-Mapped Safetensors Reader in Pure C++.
 */

#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdint>

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_offset{0};
    size_t nbytes{0};
};

class SafeTensorsLoader {
public:
    SafeTensorsLoader() : fd_(-1), mmap_data_(nullptr), file_size_(0), data_start_offset_(0) {}

    ~SafeTensorsLoader() {
        close();
    }

    bool open_file(const std::string& path) {
        close();
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "[SafeTensorsLoader] Failed to open " << path << std::endl;
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) < 0) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        file_size_ = st.st_size;

        mmap_data_ = (const uint8_t*)mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mmap_data_ == MAP_FAILED) {
            std::cerr << "[SafeTensorsLoader] mmap failed for " << path << std::endl;
            ::close(fd_);
            fd_ = -1;
            mmap_data_ = nullptr;
            return false;
        }

        uint64_t header_len = *(const uint64_t*)mmap_data_;
        data_start_offset_ = 8 + header_len;

        std::string header_json((const char*)(mmap_data_ + 8), header_len);
        parse_header(header_json);

        std::cout << "[SafeTensorsLoader] Loaded " << path << " (" << tensors_.size() << " tensors, " << (file_size_ / 1024 / 1024) << " MB)" << std::endl;
        return true;
    }

    void close() {
        if (mmap_data_ && mmap_data_ != MAP_FAILED) {
            munmap((void*)mmap_data_, file_size_);
            mmap_data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        tensors_.clear();
    }

    bool has_tensor(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    std::vector<std::string> tensor_names(const std::string& prefix = "") const {
        std::vector<std::string> names;
        for (const auto& item : tensors_) {
            if (item.first.compare(0, prefix.size(), prefix) == 0) {
                names.push_back(item.first);
            }
        }
        std::sort(names.begin(), names.end());
        return names;
    }

    const TensorInfo* get_tensor_info(const std::string& name) const {
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? &it->second : nullptr;
    }

    const void* get_tensor_data(const std::string& name) const {
        const TensorInfo* info = get_tensor_info(name);
        if (!info || !mmap_data_) return nullptr;
        return mmap_data_ + data_start_offset_ + info->data_offset;
    }

    // Read one logical row from either a native floating-point tensor or the
    // exported Rindi U32 int4 format. U32 rows pack eight 4-bit values per
    // uint32; the matching BF16 scale/bias tensors contain one value per
    // group of 64 logical columns.
    bool get_row_fp16(const std::string& name, size_t row,
                      std::vector<uint16_t>& out_fp16) const {
        const TensorInfo* info = get_tensor_info(name);
        if (!info || info->shape.size() < 2 || row >= static_cast<size_t>(info->shape[0])) {
            return false;
        }
        const size_t cols = static_cast<size_t>(info->shape[1]);
        if (info->dtype == "U32") {
            out_fp16.resize(cols * 8);
            const uint32_t* packed = static_cast<const uint32_t*>(get_tensor_data(name));
            if (!packed) return false;
            const std::string scale_name = name.substr(0, name.size() - 6) + "scales";
            const std::string bias_name = name.substr(0, name.size() - 6) + "biases";
            const TensorInfo* scales_info = get_tensor_info(scale_name);
            if (!scales_info) return false;
            const uint16_t* scales = static_cast<const uint16_t*>(get_tensor_data(scale_name));
            const uint16_t* biases = has_tensor(bias_name)
                ? static_cast<const uint16_t*>(get_tensor_data(bias_name)) : nullptr;
            const size_t logical_cols = cols * 8;
            const size_t groups = (logical_cols + 63) / 64;
            if (scales_info->shape.size() < 2 ||
                static_cast<size_t>(scales_info->shape[0]) <= row ||
                static_cast<size_t>(scales_info->shape[1]) < groups) {
                return false;
            }
            const size_t packed_row = row * cols;
            const size_t scale_row = row * static_cast<size_t>(scales_info->shape[1]);
            for (size_t c = 0; c < logical_cols; ++c) {
                const uint32_t word = packed[packed_row + c / 8];
                const uint32_t q = (word >> ((c % 8) * 4)) & 0x0F;
                const float scale = bf16_to_float(scales[scale_row + c / 64]);
                const float bias = biases ? bf16_to_float(biases[scale_row + c / 64]) : 0.0f;
                out_fp16[c] = float_to_fp16(static_cast<float>(q) * scale + bias);
            }
            return true;
        }

        out_fp16.resize(cols);
        const uint8_t* raw = static_cast<const uint8_t*>(get_tensor_data(name));
        if (!raw) return false;
        if (info->dtype == "F16") {
            std::memcpy(out_fp16.data(), raw + row * cols * sizeof(uint16_t), cols * sizeof(uint16_t));
            return true;
        }
        for (size_t c = 0; c < cols; ++c) {
            float value = 0.0f;
            if (info->dtype == "BF16") {
                uint16_t bits;
                std::memcpy(&bits, raw + (row * cols + c) * sizeof(uint16_t), sizeof(bits));
                value = bf16_to_float(bits);
            } else if (info->dtype == "F32") {
                std::memcpy(&value, raw + (row * cols + c) * sizeof(float), sizeof(value));
            } else {
                return false;
            }
            out_fp16[c] = float_to_fp16(value);
        }
        return true;
    }

    // Consume the packaged groupwise-int4 row directly. Re-dequantizing a
    // row and requantizing it with one whole-row scale loses the original
    // 64-column scales and materially degrades native projections.
    bool dot_row_fp16(const std::string& name, size_t row,
                      const uint16_t* input, size_t input_size,
                      float& result) const {
        const TensorInfo* info = get_tensor_info(name);
        if (!info || !input || info->shape.size() < 2 ||
            row >= static_cast<size_t>(info->shape[0])) return false;
        const size_t packed_cols = static_cast<size_t>(info->shape[1]);
        const size_t logical_cols = info->dtype == "U32" ? packed_cols * 8 : packed_cols;
        if (input_size < logical_cols) return false;

        if (info->dtype == "U32") {
            const uint32_t* packed = static_cast<const uint32_t*>(get_tensor_data(name));
            const std::string scale_name = name.substr(0, name.size() - 6) + "scales";
            const std::string bias_name = name.substr(0, name.size() - 6) + "biases";
            const TensorInfo* scales_info = get_tensor_info(scale_name);
            if (!packed || !scales_info) return false;
            const uint16_t* scales = static_cast<const uint16_t*>(get_tensor_data(scale_name));
            const uint16_t* biases = has_tensor(bias_name)
                ? static_cast<const uint16_t*>(get_tensor_data(bias_name)) : nullptr;
            const size_t groups = (logical_cols + 63) / 64;
            if (!scales || scales_info->shape.size() < 2 ||
                static_cast<size_t>(scales_info->shape[0]) <= row ||
                static_cast<size_t>(scales_info->shape[1]) < groups) return false;
            const size_t packed_row = row * packed_cols;
            const size_t scale_row = row * static_cast<size_t>(scales_info->shape[1]);
            float sum = 0.0f;
            for (size_t c = 0; c < logical_cols; ++c) {
                const uint32_t word = packed[packed_row + c / 8];
                const float q = static_cast<float>((word >> ((c % 8) * 4)) & 0x0f);
                const float scale = bf16_to_float(scales[scale_row + c / 64]);
                const float bias = biases ? bf16_to_float(biases[scale_row + c / 64]) : 0.0f;
                sum += fp16_to_float(input[c]) * (q * scale + bias);
            }
            result = sum;
            return true;
        }

        std::vector<uint16_t> row_fp16;
        if (!get_row_fp16(name, row, row_fp16) || row_fp16.size() != logical_cols) return false;
        float sum = 0.0f;
        for (size_t c = 0; c < logical_cols; ++c)
            sum += fp16_to_float(input[c]) * fp16_to_float(row_fp16[c]);
        result = sum;
        return true;
    }

    bool get_tensor_fp16(const std::string& name, std::vector<uint16_t>& out) const {
        const TensorInfo* info = get_tensor_info(name);
        if (!info || info->shape.empty()) return false;
        size_t count = 1;
        for (int64_t dim : info->shape) {
            if (dim <= 0) return false;
            count *= static_cast<size_t>(dim);
        }
        const uint8_t* raw = static_cast<const uint8_t*>(get_tensor_data(name));
        if (!raw) return false;
        out.resize(count);
        if (info->dtype == "F16") {
            std::memcpy(out.data(), raw, count * sizeof(uint16_t));
            return true;
        }
        for (size_t i = 0; i < count; ++i) {
            if (info->dtype == "BF16") {
                uint16_t bits;
                std::memcpy(&bits, raw + i * sizeof(uint16_t), sizeof(bits));
                out[i] = float_to_fp16(bf16_to_float(bits));
            } else if (info->dtype == "F32") {
                float value;
                std::memcpy(&value, raw + i * sizeof(float), sizeof(value));
                out[i] = float_to_fp16(value);
            } else {
                return false;
            }
        }
        return true;
    }

    bool get_embedding_row_fp16(int token_id, size_t hidden_dim, std::vector<uint16_t>& out_fp16) const {
        out_fp16.resize(hidden_dim);

        const TensorInfo* info = get_tensor_info("embed_tokens.weight");
        if (!info) info = get_tensor_info("model.embed_tokens.weight");
        if (!info) info = get_tensor_info("model.language_model.embed_tokens.weight");
        if (!info) return false;

        const uint8_t* raw = (const uint8_t*)get_tensor_data(info->name);
        if (!raw) return false;

        if (info->dtype == "BF16") {
            const uint16_t* bf16_data = (const uint16_t*)raw + token_id * hidden_dim;
            for (size_t i = 0; i < hidden_dim; ++i) {
                uint32_t u32 = static_cast<uint32_t>(bf16_data[i]) << 16;
                float f;
                std::memcpy(&f, &u32, sizeof(f));
                uint32_t x = u32;
                uint32_t sign = (x >> 16) & 0x8000;
                int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (x >> 13) & 0x3FF;
                uint16_t h = (exp <= 0) ? 0 : ((exp >= 31) ? (sign | 0x7C00) : (sign | (exp << 10) | mant));
                out_fp16[i] = h;
            }
            return true;
        } else if (info->dtype == "F16") {
            const uint16_t* f16_data = (const uint16_t*)raw + token_id * hidden_dim;
            std::memcpy(out_fp16.data(), f16_data, hidden_dim * sizeof(uint16_t));
            return true;
        } else if (info->dtype == "U32") {
            const TensorInfo* s_info = get_tensor_info("embed_tokens.scales");
            const TensorInfo* b_info = get_tensor_info("embed_tokens.biases");
            if (!s_info) return false;

            const uint32_t* u32_row = (const uint32_t*)raw + token_id * (hidden_dim / 8);
            const uint16_t* scales = (const uint16_t*)get_tensor_data(s_info->name) + token_id * (hidden_dim / 64);
            const uint16_t* biases = b_info ? (const uint16_t*)get_tensor_data(b_info->name) + token_id * (hidden_dim / 64) : nullptr;

            for (size_t g = 0; g < hidden_dim / 64; ++g) {
                uint32_t s_u32 = static_cast<uint32_t>(scales[g]) << 16;
                float scale_f;
                std::memcpy(&scale_f, &s_u32, sizeof(float));

                float bias_f = 0.0f;
                if (biases) {
                    uint32_t b_u32 = static_cast<uint32_t>(biases[g]) << 16;
                    std::memcpy(&bias_f, &b_u32, sizeof(float));
                }

                for (size_t elem = 0; elem < 64; ++elem) {
                    size_t global_idx = g * 64 + elem;
                    uint32_t packed = u32_row[global_idx / 8];
                    uint32_t q = (packed >> ((elem % 8) * 4)) & 0x0F;
                    float val = static_cast<float>(q) * scale_f + bias_f;

                    uint32_t val_u32;
                    std::memcpy(&val_u32, &val, sizeof(float));
                    uint32_t sign = (val_u32 >> 16) & 0x8000;
                    int32_t exp = ((val_u32 >> 23) & 0xFF) - 127 + 15;
                    uint32_t mant = (val_u32 >> 13) & 0x3FF;
                    uint16_t h = (exp <= 0) ? 0 : ((exp >= 31) ? (sign | 0x7C00) : (sign | (exp << 10) | mant));
                    out_fp16[global_idx] = h;
                }
            }
            return true;
        }
        return false;
    }

private:
    int fd_;
    const uint8_t* mmap_data_;
    size_t file_size_;
    size_t data_start_offset_;
    std::unordered_map<std::string, TensorInfo> tensors_;

    static float bf16_to_float(uint16_t bits) {
        uint32_t value = static_cast<uint32_t>(bits) << 16;
        float result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    static float fp16_to_float(uint16_t bits) {
        const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
        const uint32_t exponent = (bits >> 10) & 0x1fu;
        const uint32_t mantissa = bits & 0x3ffu;
        uint32_t value;
        if (exponent == 0) value = sign | (mantissa << 13);
        else if (exponent == 31) value = sign | 0x7f800000u | (mantissa << 13);
        else value = sign | ((exponent + 112u) << 23) | (mantissa << 13);
        float result;
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }

    static uint16_t float_to_fp16(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t sign = (bits >> 16) & 0x8000;
        const int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
        const uint32_t mantissa = (bits >> 13) & 0x3ff;
        if (exponent <= 0) return static_cast<uint16_t>(sign);
        if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00);
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
    }

    void parse_header(const std::string& json) {
        size_t i = 0;
        size_t n = json.size();
        while (i < n) {
            size_t k_open = json.find('"', i);
            if (k_open == std::string::npos) break;
            size_t k_close = json.find('"', k_open + 1);
            if (k_close == std::string::npos) break;
            std::string key = json.substr(k_open + 1, k_close - k_open - 1);

            size_t colon = json.find(':', k_close + 1);
            if (colon == std::string::npos) break;

            size_t val_start = json.find_first_not_of(" \t\r\n", colon + 1);
            if (val_start == std::string::npos) break;

            if (json[val_start] == '{') {
                int depth = 0;
                bool in_str = false;
                size_t val_end = std::string::npos;
                for (size_t cur = val_start; cur < n; ++cur) {
                    if (json[cur] == '"' && (cur == 0 || json[cur - 1] != '\\')) {
                        in_str = !in_str;
                    }
                    if (!in_str) {
                        if (json[cur] == '{') depth++;
                        else if (json[cur] == '}') {
                            depth--;
                            if (depth == 0) {
                                val_end = cur;
                                break;
                            }
                        }
                    }
                }
                if (val_end == std::string::npos) break;

                if (key != "__metadata__") {
                    std::string obj_str = json.substr(val_start, val_end - val_start + 1);
                    TensorInfo info;
                    info.name = key;

                    size_t dt_pos = obj_str.find("\"dtype\":");
                    if (dt_pos != std::string::npos) {
                        size_t s1 = obj_str.find('"', dt_pos + 8);
                        size_t s2 = obj_str.find('"', s1 + 1);
                        info.dtype = obj_str.substr(s1 + 1, s2 - s1 - 1);
                    }

                    size_t off_pos = obj_str.find("\"data_offsets\":");
                    if (off_pos != std::string::npos) {
                        size_t b1 = obj_str.find('[', off_pos);
                        size_t comma = obj_str.find(',', b1);
                        size_t b2 = obj_str.find(']', comma);
                        size_t start = std::stoull(obj_str.substr(b1 + 1, comma - b1 - 1));
                        size_t end = std::stoull(obj_str.substr(comma + 1, b2 - comma - 1));
                        info.data_offset = start;
                        info.nbytes = end - start;
                    }
                    size_t shape_pos = obj_str.find("\"shape\":");
                    if (shape_pos != std::string::npos) {
                        size_t open = obj_str.find('[', shape_pos);
                        size_t close = obj_str.find(']', open);
                        if (open != std::string::npos && close != std::string::npos) {
                            size_t p = open + 1;
                            while (p < close) {
                                p = obj_str.find_first_of("0123456789", p);
                                if (p == std::string::npos || p >= close) break;
                                size_t e = obj_str.find_first_not_of("0123456789", p);
                                info.shape.push_back(std::stoll(obj_str.substr(p, e - p)));
                                p = e;
                            }
                        }
                    }
                    tensors_[key] = info;
                }
                i = val_end + 1;
            } else {
                size_t next_comma = json.find(',', val_start);
                i = (next_comma != std::string::npos) ? next_comma + 1 : n;
            }
        }
    }
};

#endif // SAFETENSORS_LOADER_H
