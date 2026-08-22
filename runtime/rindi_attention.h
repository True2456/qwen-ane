#ifndef RINDI_ATTENTION_H
#define RINDI_ATTENTION_H

#include "rindi_ane_projection.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace rindi_attn {
constexpr size_t kHQ = 24;   // query heads
constexpr size_t kHK = 4;    // key/value heads
constexpr size_t kD = 256;   // head dim
constexpr size_t kKV = kHK * kD;
constexpr size_t kQ = kHQ * kD;
}

class RindiAttention {
public:
    enum class ProjectionMode { Int4Auto, Bf16Metal, Int4FromBf16 };
    bool compile(ANEContext* ctx, const SafeTensorsLoader& loader, int layer,
                 size_t context = 256, size_t width = 32);
    bool compile_core(ANEContext* ctx, const SafeTensorsLoader& loader, int layer,
                      size_t context = 256, size_t width = 32);
    // Same layer shape under an arbitrary tensor prefix (e.g. the MTP draft
    // block's "mtp.layers.0.self_attn."), optionally driving unquantized
    // BF16 weights through the direct Metal GEMM path.
    bool compile_prefixed(ANEContext* ctx, const SafeTensorsLoader& loader,
                          const std::string& self_attn_prefix,
                          size_t context, size_t width,
                          ProjectionMode mode = ProjectionMode::Int4Auto);
    bool project(const uint16_t* hidden, size_t lanes,
                 std::vector<uint16_t>& q, std::vector<uint16_t>& k,
                 std::vector<uint16_t>& v);
    bool step(const uint16_t* hidden, size_t lanes,
              std::vector<uint16_t>& output);
    bool core_step(const std::vector<uint16_t>& qraw,
                   const std::vector<uint16_t>& kraw,
                   const std::vector<uint16_t>& vraw,
                   std::vector<uint16_t>& attended);
    bool core_step(const uint16_t* qraw, size_t qraw_len,
                   const uint16_t* kraw, size_t kraw_len,
                   const uint16_t* vraw, size_t vraw_len,
                   std::vector<uint16_t>& attended);
    bool core_step_batch(const std::vector<uint16_t>& qraw,
                         const std::vector<uint16_t>& kraw,
                         const std::vector<uint16_t>& vraw,
                         size_t lanes,
                         std::vector<uint16_t>& attended);
    // Channel-major inputs without requiring contiguous per-lane copies.
    bool core_step_batch(const uint16_t* qraw, const uint16_t* kraw,
                         const uint16_t* vraw, size_t lanes,
                         std::vector<uint16_t>& attended);
    void reset();
    void set_position(size_t pos) { position_ = pos; }
    size_t position() const { return position_; }
    bool ready() const { return ready_; }

    // Speculative-decode rollback: capture/restore KV rows written since
    // `from_pos` (host arrays and their Metal mirrors) and the position.
    void snapshot_kv(size_t from_pos, std::vector<uint16_t>& keys,
                     std::vector<uint16_t>& values) const {
                const size_t row = std::min(from_pos, position_);
        const size_t n = (position_ - row) * rindi_attn::kKV;
        keys.assign(keys_.begin() + row * rindi_attn::kKV, keys_.begin() + row * rindi_attn::kKV + n);
        values.assign(values_.begin() + row * rindi_attn::kKV, values_.begin() + row * rindi_attn::kKV + n);
    }
    void restore_kv(size_t from_pos, const std::vector<uint16_t>& keys,
                    const std::vector<uint16_t>& values) {
        const size_t row = std::min(from_pos, position_);
        const size_t n = keys.size();
        if (row * rindi_attn::kKV + n > keys_.size()) return;
        std::memcpy(keys_.data() + row * rindi_attn::kKV, keys.data(), n * sizeof(uint16_t));
        std::memcpy(values_.data() + row * rindi_attn::kKV, values.data(), n * sizeof(uint16_t));
        if (metal_k_cache_ && n) {
            std::memcpy(static_cast<uint16_t*>(metal_buffer_get_contents(metal_k_cache_)) + row * rindi_attn::kKV,
                        keys.data(), n * sizeof(uint16_t));
            std::memcpy(static_cast<uint16_t*>(metal_buffer_get_contents(metal_v_cache_)) + row * rindi_attn::kKV,
                        values.data(), n * sizeof(uint16_t));
        }
        position_ = row;
    }
    static constexpr size_t kv_row_elems() { return rindi_attn::kKV; }

private:
    void reset_metal_state();
    bool ensure_metal_resources();
    void prepare_qkv(const uint16_t* qraw, const uint16_t* kraw,
                     const uint16_t* vraw, size_t rope_pos,
                     float* q, float* k, float* v);
    void attend_reference(const float* q, size_t lane_index,
                          size_t valid, size_t lanes,
                          std::vector<uint16_t>& attended) const;
    bool core_batch_impl(const uint16_t* qraw, const uint16_t* kraw,
                         const uint16_t* vraw, size_t lanes,
                         std::vector<uint16_t>& attended);

public:
    ~RindiAttention() {
        if (metal_k_cache_) metal_buffer_release(metal_k_cache_);
        if (metal_v_cache_) metal_buffer_release(metal_v_cache_);
    }

private:
    ProjectionMode proj_mode_{ProjectionMode::Int4Auto};
    RindiAneProjection q_proj_, k_proj_, v_proj_, o_proj_;
    // Optional fused q/k/v projection (single dispatch); rows [q; k; v].
    RindiAneProjection qkv_fused_;
    bool qkv_fused_ready_{false};
    std::vector<uint16_t> q_norm_, k_norm_;
    std::vector<uint16_t> keys_, values_;
    // Reused scratch so the decode loop does not reallocate per layer.
    std::vector<float> q_f_, k_f_, v_f_, score_f_;
    std::vector<uint16_t> q_lane_, k_lane_, v_lane_, out_lane_;
    MetalContext* metal_ctx_{nullptr};
    MetalBufferHandle metal_k_cache_{nullptr};
    MetalBufferHandle metal_v_cache_{nullptr};
    size_t context_{0}, position_{0}, width_{32};
    bool ready_{false};
};

#endif
