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
};

#endif
