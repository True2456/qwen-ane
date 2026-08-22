#include "rindi_attention.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
float h2f(uint16_t b) {
    uint32_t s=(uint32_t(b)&0x8000u)<<16, e=(b>>10)&31u, m=b&1023u, v=s;
    if (e == 0) v=s|(m<<13); else if (e == 31) v=s|0x7f800000u|(m<<13);
    else v=s|((e+112u)<<23)|(m<<13);
    float f; std::memcpy(&f,&v,4); return f;
}
uint16_t f2h(float f) {
    uint32_t b; std::memcpy(&b,&f,4); int e=int((b>>23)&255)-127+15;
    uint32_t s=(b>>16)&0x8000u, m=(b>>13)&1023u;
    if (e<=0) return uint16_t(s); if (e>=31) return uint16_t(s|0x7c00u);
    return uint16_t(s|(uint32_t(e)<<10)|m);
}
}

bool RindiAttention::compile(ANEContext* ctx, const SafeTensorsLoader& loader,
                             int layer, size_t context, size_t width) {
    if (!compile_core(ctx, loader, layer, context, width)) return false;
    const std::string p = "layers." + std::to_string(layer) + ".self_attn.";
    return o_proj_.compile_int4(ctx, loader, p+"o_proj.weight", width_, "attn_o");
}

bool RindiAttention::compile_core(ANEContext* ctx, const SafeTensorsLoader& loader,
                                  int layer, size_t context, size_t width) {
    if (!ctx || layer < 0 || context == 0) return false;
    width_ = std::max<size_t>(32, width); context_ = context;
    const std::string p = "layers." + std::to_string(layer) + ".self_attn.";
    if (!q_proj_.compile_int4_host(loader, p+"q_proj.weight") ||
        !k_proj_.compile_int4_host(loader, p+"k_proj.weight") ||
        !v_proj_.compile_int4_host(loader, p+"v_proj.weight") ||
        !loader.get_tensor_fp16(p+"q_norm.weight", q_norm_) ||
        !loader.get_tensor_fp16(p+"k_norm.weight", k_norm_) ||
        q_norm_.size() != 256 || k_norm_.size() != 256) return false;
    keys_.assign(4 * context_ * 256, 0);
    values_.assign(4 * context_ * 256, 0);
    position_ = 0; ready_ = true; return true;
}

bool RindiAttention::project(const uint16_t* hidden, size_t lanes,
                             std::vector<uint16_t>& q,
                             std::vector<uint16_t>& k,
                             std::vector<uint16_t>& v) {
    if (!ready_ || !hidden || lanes == 0 || lanes > width_) return false;
    return q_proj_.evaluate(hidden, lanes, q) &&
           k_proj_.evaluate(hidden, lanes, k) &&
           v_proj_.evaluate(hidden, lanes, v);
}

void RindiAttention::reset() {
    std::fill(keys_.begin(), keys_.end(), 0);
    std::fill(values_.begin(), values_.end(), 0); position_ = 0;
}

bool RindiAttention::step(const uint16_t* hidden, size_t lanes,
                          std::vector<uint16_t>& output) {
    if (!ready_ || !hidden || lanes != 1 || position_ >= context_) return false;
    std::vector<uint16_t> qraw, kraw, vraw;
    if (!project(hidden, 1, qraw, kraw, vraw)) return false;
    std::vector<uint16_t> attended;
    if (!core_step(qraw, kraw, vraw, attended)) return false;
    for (size_t h = 0; h < 24; ++h) {
        for (size_t d = 0; d < 256; ++d) {
            const size_t c = h * 256 + d;
            const float gate = h2f(qraw[h * 512 + 256 + d]);
            attended[c] = f2h(h2f(attended[c]) /
                              (1.0f + std::exp(-gate)));
        }
    }
    if (!o_proj_.evaluate(attended.data(), 1, output)) return false;
    for(size_t c=0;c<5120;++c){float y=h2f(output[c])+h2f(hidden[c]);output[c]=f2h(y);}
    return true;
}

bool RindiAttention::core_step(const std::vector<uint16_t>& qraw,
                               const std::vector<uint16_t>& kraw,
                               const std::vector<uint16_t>& vraw,
                               std::vector<uint16_t>& attended) {
    if (!ready_ || position_ >= context_ || qraw.size() < 24 * 512 ||
        kraw.size() < 4 * 256 || vraw.size() < 4 * 256) return false;
    constexpr size_t HQ=24, HK=4, D=256, Q=HQ*D, K=HK*D;
    std::vector<float> q(Q), k(K), v(K);
    for (size_t h=0; h<HQ; ++h) {
        float ss=0;
        for (size_t d = 0; d < D; ++d) {
            const float x = h2f(qraw[h * (2 * D) + d]);
            ss += x * x;
        }
        float den=std::sqrt(ss/D+1e-6f);
        for (size_t d = 0; d < D; ++d)
            q[h * D + d] = h2f(qraw[h * (2 * D) + d]) / den * h2f(q_norm_[d]);
    }
    for (size_t h=0; h<HK; ++h) {
        float ss=0; for(size_t d=0;d<D;++d){float x=h2f(kraw[h*D+d]);ss+=x*x;}
        float den=std::sqrt(ss/D+1e-6f);
        for(size_t d=0;d<D;++d) {
            k[h*D+d]=h2f(kraw[h*D+d])/den*h2f(k_norm_[d]);
            v[h*D+d]=h2f(vraw[h*D+d]);
        }
    }
    const float inv=1.0f/std::sqrt(256.0f);
    // Qwen3.5's non-traditional MLX RoPE pairs the first 64 channels as two
    // 32-wide halves: (0,32), (1,33), ... .  The q_proj rows themselves are
    // laid out per head as [q(256), gate(256)], so qraw was unpacked above.
    for (size_t h = 0; h < HQ; ++h) {
        for (size_t d = 0; d < 32; ++d) {
            const float angle = static_cast<float>(position_) *
                std::pow(10000000.0f, -static_cast<float>(d) / 32.0f);
            const float co = std::cos(angle), si = std::sin(angle);
            const float a = q[h * D + d], b = q[h * D + 32 + d];
            q[h * D + d] = a * co - b * si;
            q[h * D + 32 + d] = a * si + b * co;
        }
    }
    for (size_t h = 0; h < HK; ++h) {
        for (size_t d = 0; d < 32; ++d) {
            const float angle = static_cast<float>(position_) *
                std::pow(10000000.0f, -static_cast<float>(d) / 32.0f);
            const float co = std::cos(angle), si = std::sin(angle);
            const float a = k[h * D + d], b = k[h * D + 32 + d];
            k[h * D + d] = a * co - b * si;
            k[h * D + 32 + d] = a * si + b * co;
        }
    }
    for (size_t i = 0; i < K; ++i) {
        keys_[position_ * K + i] = f2h(k[i]);
        values_[position_ * K + i] = f2h(v[i]);
    }
    attended.assign(Q, 0);
    const size_t valid=position_+1;
    for(size_t h=0;h<HQ;++h){
        std::vector<float> score(valid);
        float mx=-1e30f;
        for(size_t t=0;t<valid;++t){float s=0;size_t kh=(h/(HQ/HK))*D;for(size_t d=0;d<D;++d)s+=q[h*D+d]*h2f(keys_[t*K+kh+d]);score[t]=s*inv;mx=std::max(mx,score[t]);}
        float den=0;for(float& s:score){s=std::exp(s-mx);den+=s;}
        for(size_t d=0;d<D;++d){float y=0;for(size_t t=0;t<valid;++t){size_t kh=(h/(HQ/HK))*D;y+=score[t]/den*h2f(values_[t*K+kh+d]);}attended[h*D+d]=f2h(y);}
    }
    ++position_; return true;
}

bool RindiAttention::core_step_batch(const std::vector<uint16_t>& qraw,
                                     const std::vector<uint16_t>& kraw,
                                     const std::vector<uint16_t>& vraw,
                                     size_t lanes,
                                     std::vector<uint16_t>& attended) {
    constexpr size_t QRAW = 24 * 512;
    constexpr size_t KRAW = 4 * 256;
    constexpr size_t VRAW = 4 * 256;
    constexpr size_t OUT = 24 * 256;
    if (!ready_ || lanes == 0 || lanes > width_ ||
        qraw.size() != QRAW * lanes || kraw.size() != KRAW * lanes ||
        vraw.size() != VRAW * lanes) return false;
    attended.assign(OUT * lanes, 0);
    std::vector<uint16_t> q_lane(QRAW), k_lane(KRAW), v_lane(VRAW), out_lane;
    for (size_t lane = 0; lane < lanes; ++lane) {
        for (size_t c = 0; c < QRAW; ++c) q_lane[c] = qraw[c * lanes + lane];
        for (size_t c = 0; c < KRAW; ++c) k_lane[c] = kraw[c * lanes + lane];
        for (size_t c = 0; c < VRAW; ++c) v_lane[c] = vraw[c * lanes + lane];
        if (!core_step(q_lane, k_lane, v_lane, out_lane) || out_lane.size() != OUT)
            return false;
        for (size_t c = 0; c < OUT; ++c) attended[c * lanes + lane] = out_lane[c];
    }
    return true;
}
