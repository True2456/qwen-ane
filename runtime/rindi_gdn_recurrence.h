// SPDX-License-Identifier: Apache-2.0
#ifndef RINDI_GDN_RECURRENCE_H
#define RINDI_GDN_RECURRENCE_H

#include "ane_c_bridge.h"
#include "metal_engine.h"
#include <IOSurface/IOSurfaceRef.h>
#include <cstddef>
#include <cstdint>
#include <vector>

class RindiGdnRecurrence {
public:
    RindiGdnRecurrence() = default;
    ~RindiGdnRecurrence();
    RindiGdnRecurrence(const RindiGdnRecurrence&) = delete;

    bool compile(ANEContext* ctx, size_t heads = 48, size_t key_dim = 128,
                 size_t value_dim = 128, size_t width = 160);
    bool step(const uint16_t* packed_inputs, std::vector<uint16_t>& output);
    bool step_batch(const uint16_t* decay, const uint16_t* key,
                   const uint16_t* query, const uint16_t* value,
                   const uint16_t* beta, size_t lanes,
                   std::vector<uint16_t>& output);
    bool step_batch_raw(const uint16_t* activated, const uint16_t* a,
                        const uint16_t* b, const uint16_t* a_log,
                        const uint16_t* dt_bias, size_t lanes,
                        std::vector<uint16_t>& output);
    void reset();
    // Byte-exact capture/rollback of the recurrent state (one shared store
    // for the Metal and ANE execution paths).
    void snapshot_state(std::vector<uint16_t>& out) const;
    void restore_state(const std::vector<uint16_t>& in);
    size_t state_elems() const { return hidden_keys_ * value_dim_; }

private:
    bool compile_prepared(ANEContext* ctx, size_t heads, size_t key_dim,
                          size_t value_dim, size_t width);
    ANEContext* ctx_{nullptr};
    ANEModel* model_{nullptr};
    ANERequest* request_{nullptr};
    IOSurfaceRef input_surface_{nullptr};
    IOSurfaceRef output_surface_{nullptr};
    IOSurfaceRef state_surface_{nullptr};
    size_t heads_{0}, key_dim_{0}, value_dim_{0}, hidden_keys_{0};
    size_t input_channels_{0}, width_{0};
    MetalContext* metal_ctx_{nullptr};
    MetalBufferHandle metal_state_{nullptr};
    MetalBufferHandle metal_decay_{nullptr};
    MetalBufferHandle metal_key_{nullptr};
    MetalBufferHandle metal_query_{nullptr};
    MetalBufferHandle metal_value_{nullptr};
    MetalBufferHandle metal_beta_{nullptr};
    MetalBufferHandle metal_output_{nullptr};
    MetalBufferHandle metal_activated_{nullptr};
    MetalBufferHandle metal_a_{nullptr};
    MetalBufferHandle metal_b_{nullptr};
    MetalBufferHandle metal_a_log_{nullptr};
    MetalBufferHandle metal_dt_bias_{nullptr};
};

#endif
