// Corrected lane-width invariance: SAME lane-0 data, different total widths.
#include "runtime/rindi_native_chain.h"
#include "runtime/safetensors_loader.h"
#include <cstdio>
#include <vector>
static uint32_t rs=7;
static uint16_t rnd16(){ rs=rs*1103515245u+12345u; return (uint16_t)((rs>>16)|0x3800); }
int main(int argc, char** argv){
    const char* model = argc>1?argv[1]:"/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    SafeTensorsLoader loader;
    if(!loader.open_file(std::string(model)+"/gpu_backbone.safetensors")) return 2;
    RindiNativeChain chain(5120, 32);
    ANEContext* ane = ane_context_create();
    if(!ane) return 2;
    if(!chain.compile_layer(3, std::string(model)+"/ane_layers", loader)){fprintf(stderr,"compile\n");return 2;}
    const size_t core_dim=6144,H=5120;
    // fixed lane-0 data
    std::vector<uint16_t> core0(core_dim), res0(H);
    for(size_t c=0;c<core_dim;++c) core0[c]=rnd16();
    for(size_t c=0;c<H;++c) res0[c]=rnd16();
    for(size_t L : {1u,2u,3u,4u,8u,16u}){
        std::vector<uint16_t> core(core_dim*L), res(H*L), out;
        for(size_t c=0;c<core_dim;++c){ core[c*L]=core0[c]; for(size_t l=1;l<L;++l) core[c*L+l]=rnd16(); }
        for(size_t c=0;c<H;++c){ res[c*L]=res0[c]; for(size_t l=1;l<L;++l) res[c*L+l]=rnd16(); }
        chain.evaluate_tail_batch(3, core.data(), core_dim, res.data(), L, out, nullptr);
        std::vector<uint16_t> l0(out.size()/L);
        for(size_t c=0;c<l0.size();++c) l0[c]=out[c*L];
        unsigned long long h=1469598103934665603ull;
        for(auto x:l0){h^=x;h*=1099511628211ull;}
        printf("L=%2zu lane0_hash=%llx\n", L, h);
    }
    return 0;
}
