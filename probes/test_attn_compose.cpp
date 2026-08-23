// Width-invariance of attention core_step_batch: same lane-0 data within
// widths 1 vs 3 vs 4 -> identical attended output AND identical KV rows.
#include "runtime/rindi_gdn_layer.h"
#include "runtime/rindi_attention.h"
#include "runtime/safetensors_loader.h"
#include "runtime/ane_c_bridge.h"
#include "runtime/metal_engine.h"
#include <cstdio>
#include <vector>
static uint32_t rs=11;
static uint16_t rnd16(){ rs=rs*1103515245u+12345u; return (uint16_t)((rs>>16)|0x3800); }
int main(int argc, char** argv){
    const char* model = argc>1?argv[1]:"/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    SafeTensorsLoader loader;
    if(!loader.open_file(std::string(model)+"/gpu_backbone.safetensors")) return 2;
    ANEContext* ane = ane_context_create();
    if(!ane) return 2;
    RindiAttention att;
    if(!att.compile_core(ane, loader, 3, 4096, 32)){fprintf(stderr,"compile\n");return 2;}
    constexpr size_t HID=5120, QG=12288, K=1024;
    // normalized input, lanes up to 4
    std::vector<uint16_t> norm(HID*4);
    for(auto&x:norm) x=rnd16();

    auto run=[&](size_t lanes, std::vector<uint16_t>& att_out,
                 unsigned long long& kv_hash){
        std::vector<uint16_t> q,k,v;
        fprintf(stderr,"[A] run lanes=%zu\n",lanes);
        if(!att.project(norm.data(), lanes, q,k,v)) {fprintf(stderr,"project %zu\n",lanes); return false;}
        if(!att.core_step_batch(q,k,v,lanes,att_out)) {fprintf(stderr,"core %zu\n",lanes); return false;}
        std::vector<uint16_t> sk,sv;
        att.snapshot_kv(0,sk,sv);
        unsigned long long h=1469598103934665603ull;
        for(auto x:sk){h^=x;h*=1099511628211ull;}
        for(auto x:sv){h^=x;h*=1099511628211ull;}
        kv_hash=h;
        return true;
    };

    // A: width 3
    std::vector<uint16_t> att3; unsigned long long kv3h=0;
    if(!run(3, att3, kv3h)) return 2;
    // rewind for B/C
    att.reset();
    // B: width 4
    std::vector<uint16_t> att4; unsigned long long kv4h=0;
    if(!run(4, att4, kv4h)) return 2;
    // C: width 1 three times (sequential single-lane)
    att.reset();
    std::vector<uint16_t> attC(QG*3);
    for(size_t l=0;l<3;++l){
        std::vector<uint16_t> n1(HID), q,k,v, a1;
        for(size_t c=0;c<HID;++c) n1[c]=norm[c*4+l];
        if(!att.project(n1.data(),1,q,k,v)) return 2;
        if(!att.core_step_batch(q,k,v,1,a1)) return 2;
        for(size_t c=0;c<QG;++c) attC[c*3+l]=a1[c];
    }
    // compare attended lane0..2 across A/B/C (channel-major [QG x 3])
    size_t badAB=0,badAC=0;
    for(size_t c=0;c<QG;++c) for(size_t l=0;l<3;++l){
        uint16_t x=att3[c*3+l], y=att4[c*4+l], z=attC[c*3+l];
        if(x!=y) ++badAB;
        if(x!=z) ++badAC;
    }
    printf("attended: B-vs-A differ=%zu/%zu  C-vs-A differ=%zu/%zu\n",badAB,QG*3,badAC,QG*3);
    // KV rows compare: A vs B rows 0..2 (k then v)
    printf("KV hash: A=%llx B=%llx %s\n",(unsigned long long)kv3h,(unsigned long long)kv4h,(kv3h==kv4h)?"MATCH":"DIVERGE");
    printf("%s\n",(badAB||badAC||kv3h!=kv4h)?"WIDTH-DEPENDENT":"INVARIANT");
    return 0;
}
