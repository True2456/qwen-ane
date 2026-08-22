// SPDX-License-Identifier: Apache-2.0
#ifndef RINDI_GDN_LAYER_H
#define RINDI_GDN_LAYER_H

#include "rindi_ane_projection.h"
#include "rindi_gdn_conv.h"
#include "rindi_gdn_recurrence.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct RindiGdnProjectionOutput {
    std::vector<uint16_t> qkv;
    std::vector<uint16_t> z;
    std::vector<uint16_t> beta;
    std::vector<uint16_t> a;
};

// Native representation of one Qwen linear-attention layer, including its
// input projections, causal convolution, recurrent core, output projection,
// and gated MLP tail.
class RindiGdnLayer {
public:
    bool compile(ANEContext* ctx, const SafeTensorsLoader& loader, int layer,
                 size_t width = 32);
    // Compile only the attention/GDN core. The residual, output projection,
    // normalization, and MLP are supplied by the fused native tail.
    bool compile_core(ANEContext* ctx, const SafeTensorsLoader& loader, int layer,
                      size_t width = 32);
    bool project(const uint16_t* hidden, size_t lanes,
                 RindiGdnProjectionOutput& output);
    // Runs projection, causal SiLU convolution, host preparation of the
    // recurrent gates, and the ANE gated-delta state update. Returns the
    // compact 6144-channel GDN core before the output/tail MLP.
    bool step(const uint16_t* hidden, size_t lanes,
              std::vector<uint16_t>& core);
    bool core_step(const uint16_t* hidden, size_t lanes,
                   std::vector<uint16_t>& core,
                   std::vector<uint16_t>& z);
    bool core_from_projected(const RindiGdnProjectionOutput& projected,
                             size_t lanes, std::vector<uint16_t>& core,
                             std::vector<uint16_t>& z);
    bool gate_core(const std::vector<uint16_t>& core,
                   const std::vector<uint16_t>& z,
                   std::vector<uint16_t>& gated) const;
    bool gate_core_batch(const std::vector<uint16_t>& core,
                         const std::vector<uint16_t>& z,
                         size_t lanes,
                         std::vector<uint16_t>& gated) const;
    void reset();
    bool ready() const { return ready_; }

private:
    std::array<RindiAneProjection, 4> projections_;
    RindiGdnConv conv_;
    RindiGdnRecurrence recurrence_;
    std::vector<uint16_t> a_log_;
    std::vector<uint16_t> dt_bias_;
    std::vector<uint16_t> gdn_norm_;
    std::vector<uint16_t> post_norm_;
    RindiAneProjection out_proj_;
    RindiAneProjection mlp_gate_;
    RindiAneProjection mlp_up_;
    RindiAneProjection mlp_down_;
    size_t width_{32};
    bool ready_{false};
};

#endif
