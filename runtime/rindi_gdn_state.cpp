// SPDX-License-Identifier: Apache-2.0
#include "rindi_gdn_state.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
float half_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t value;
    if (exponent == 0) value = sign;
    else if (exponent == 31) value = sign | 0x7f800000u | (mantissa << 13);
    else value = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    const uint32_t mantissa = (bits >> 13) & 0x3ffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | mantissa);
}
}

RindiGdnState::RindiGdnState(size_t heads, size_t key_dim, size_t value_dim)
    : heads_(heads), key_dim_(key_dim), value_dim_(value_dim),
      state_(heads * value_dim * key_dim, 0.0f) {}

void RindiGdnState::reset() {
    std::fill(state_.begin(), state_.end(), 0.0f);
}

std::vector<uint16_t> RindiGdnState::snapshot() const {
    std::vector<uint16_t> result(state_.size());
    for (size_t i = 0; i < state_.size(); ++i) result[i] = float_to_half(state_[i]);
    return result;
}

bool RindiGdnState::restore(const std::vector<uint16_t>& state) {
    if (state.size() != state_.size()) return false;
    for (size_t i = 0; i < state_.size(); ++i) state_[i] = half_to_float(state[i]);
    return true;
}

void RindiGdnState::step(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                         const uint16_t* decay, const uint16_t* beta,
                         std::vector<uint16_t>& output) {
    output.resize(heads_ * value_dim_);
    for (size_t h = 0; h < heads_; ++h) {
        const float d = half_to_float(decay[h]);
        const float b = half_to_float(beta[h]);
        for (size_t dv = 0; dv < value_dim_; ++dv) {
            for (size_t dk = 0; dk < key_dim_; ++dk) {
                state_[(h * value_dim_ + dv) * key_dim_ + dk] *= d;
            }
        }
        for (size_t dv = 0; dv < value_dim_; ++dv) {
            float memory = 0.0f;
            for (size_t dk = 0; dk < key_dim_; ++dk) {
                memory += state_[(h * value_dim_ + dv) * key_dim_ + dk] * half_to_float(k[h * key_dim_ + dk]);
            }
            const float delta = (half_to_float(v[h * value_dim_ + dv]) - memory) * b;
            for (size_t dk = 0; dk < key_dim_; ++dk) {
                state_[(h * value_dim_ + dv) * key_dim_ + dk] += delta * half_to_float(k[h * key_dim_ + dk]);
            }
            float y = 0.0f;
            for (size_t dk = 0; dk < key_dim_; ++dk) {
                y += state_[(h * value_dim_ + dv) * key_dim_ + dk] * half_to_float(q[h * key_dim_ + dk]);
            }
            output[h * value_dim_ + dv] = float_to_half(64.0f * y);
        }
    }
}
