// SPDX-License-Identifier: Apache-2.0
#ifndef RINDI_GDN_STATE_H
#define RINDI_GDN_STATE_H

#include <cstddef>
#include <cstdint>
#include <vector>

// Mutable Qwen gated-delta recurrence state. Layout is [head, value_dim,
// key_dim], matching the pure-ANE implementation's state surface.
class RindiGdnState {
public:
    RindiGdnState(size_t heads = 48, size_t key_dim = 128, size_t value_dim = 128);

    void reset();
    std::vector<uint16_t> snapshot() const;
    bool restore(const std::vector<uint16_t>& state);

    // q/k are [heads,key_dim], v is [heads,value_dim]. The caller supplies
    // normalized q/k, per-head decay and beta values, exactly as emitted by
    // the ANE recurrence-preparation stage.
    void step(const uint16_t* q, const uint16_t* k, const uint16_t* v,
              const uint16_t* decay, const uint16_t* beta,
              std::vector<uint16_t>& output);

    size_t heads() const { return heads_; }
    size_t key_dim() const { return key_dim_; }
    size_t value_dim() const { return value_dim_; }

private:
    size_t heads_;
    size_t key_dim_;
    size_t value_dim_;
    std::vector<float> state_;
};

#endif
