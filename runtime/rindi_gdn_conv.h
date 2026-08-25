// SPDX-License-Identifier: Apache-2.0
#ifndef RINDI_GDN_CONV_H
#define RINDI_GDN_CONV_H

#include "ane_c_bridge.h"
#include "safetensors_loader.h"
#include <IOSurface/IOSurfaceRef.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class RindiGdnConv {
public:
    RindiGdnConv() = default;
    ~RindiGdnConv();
    RindiGdnConv(const RindiGdnConv&) = delete;

    bool compile(ANEContext* ctx, const SafeTensorsLoader& loader,
                 const std::string& weight_name, size_t width = 32);
    bool evaluate(const uint16_t* current, size_t lanes,
                  std::vector<uint16_t>& output);
    void reset();
    // Causal-window capture for speculative-decode rollback.
    void snapshot_history(std::vector<uint16_t>& out) const { out = history_; }
    // P6 diagnostic/rollback: the ANE conv IOSurface is state NOT covered by
    // history_ (stale columns + written_lanes_ affect nothing for gathered
    // outputs in theory, but hash-diffs say otherwise - expose it).
    // CPU fallback (macOS 27): no IOSurface state exists; history_ only.
    void snapshot_surface(std::vector<uint16_t>& out) const {
        if (!input_surface_) { out.clear(); return; }
        const uint16_t* src =
            static_cast<const uint16_t*>(IOSurfaceGetBaseAddress(input_surface_));
        out.assign(src, src + channels_ * width_);
    }
    void restore_surface(const std::vector<uint16_t>& in) {
        if (!input_surface_ || in.size() != channels_ * width_) return;
        uint16_t* dst =
            static_cast<uint16_t*>(IOSurfaceGetBaseAddress(input_surface_));
        std::memcpy(dst, in.data(), in.size() * sizeof(uint16_t));
    }
    size_t written_lanes() const { return written_lanes_; }
    void restore_history(const std::vector<uint16_t>& in) {
        if (in.size() == history_.size()) history_ = in;
    }
    size_t channels() const { return channels_; }

private:
    ANEContext* ctx_{nullptr};
    ANEModel* model_{nullptr};
    ANERequest* request_{nullptr};
    IOSurfaceRef input_surface_{nullptr};
    IOSurfaceRef output_surface_{nullptr};
    size_t channels_{0};
    size_t width_{0};
    size_t written_lanes_{0};
    std::vector<uint16_t> history_;
    // macOS 27 CPU fallback: legacy MIL no longer compiles, so the causal
    // 4-tap FIR + SiLU runs on the CPU with identical history semantics.
    bool cpu_fallback_{false};
    std::vector<uint16_t> weights_;   // [channels x 4] taps
};

#endif
