#ifndef RINDI_ATTENTION_H
#define RINDI_ATTENTION_H

#include "rindi_ane_projection.h"
#include <cstddef>
#include <cstdint>
#include <vector>

class RindiAttention {
public:
    bool compile(ANEContext* ctx, const SafeTensorsLoader& loader, int layer,
                 size_t context = 256, size_t width = 32);
    bool compile_core(ANEContext* ctx, const SafeTensorsLoader& loader, int layer,
                      size_t context = 256, size_t width = 32);
    bool project(const uint16_t* hidden, size_t lanes,
                 std::vector<uint16_t>& q, std::vector<uint16_t>& k,
                 std::vector<uint16_t>& v);
    bool step(const uint16_t* hidden, size_t lanes,
              std::vector<uint16_t>& output);
    bool core_step(const std::vector<uint16_t>& qraw,
                   const std::vector<uint16_t>& kraw,
                   const std::vector<uint16_t>& vraw,
                   std::vector<uint16_t>& attended);
    bool core_step_batch(const std::vector<uint16_t>& qraw,
                         const std::vector<uint16_t>& kraw,
                         const std::vector<uint16_t>& vraw,
                         size_t lanes,
                         std::vector<uint16_t>& attended);
    void reset();
    size_t position() const { return position_; }
    bool ready() const { return ready_; }

private:
    RindiAneProjection q_proj_, k_proj_, v_proj_, o_proj_;
    std::vector<uint16_t> q_norm_, k_norm_;
    std::vector<uint16_t> keys_, values_;
    size_t context_{0}, position_{0}, width_{32};
    bool ready_{false};
};

#endif
