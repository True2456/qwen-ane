// P11 follow-up: is the folded next_projection (np) width-invariant?
// The P7 tail-invariance proof covered only the hidden outputs of
// evaluate_tail_batch. The MTP rebuild feeds np CAPTURED at k+1-wide verify
// while legacy replay REGENERATES np at keep-wide; if np channels differ
// across widths, attention KV diverges exactly as observed in dual-debug.
//
// Method: fix lane l's input data for l in [0,4) across ALL widths; hash each
// lane's np slice per width; compare hashes for the same lane across widths.
// PASS: every lane's hash identical at every width that includes it.
#include "runtime/rindi_native_chain.h"
#include "runtime/safetensors_loader.h"
#include <cstdio>
#include <vector>
static uint32_t rs=7;
static uint16_t rnd16(){ rs=rs*1103515245u+12345u; return (uint16_t)((rs>>16)|0x3800); }
static unsigned long long fnv(const std::vector<uint16_t>& v){
    unsigned long long h=1469598103934665603ull;
    for(auto x:v){h^=x;h*=1099511628211ull;}
    return h;
}
int main(int argc, char** argv){
    const char* model = argc>1?argv[1]:"/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    const std::string mode = argc>2?argv[2]:"one"; // one | all | metal
    SafeTensorsLoader loader;
    if(!loader.open_file(std::string(model)+"/gpu_backbone.safetensors")) return 2;
    RindiNativeChain chain(5120, 32);
    ANEContext* ane = ane_context_create();
    if(!ane) return 2;
    if(mode=="all"){
        for(int l=0;l<64;++l)
            if(!chain.compile_layer(l, std::string(model)+"/ane_layers", loader)){fprintf(stderr,"compile %d\n",l);return 2;}
        fprintf(stderr,"compiled all 64 layers\n");
    } else if(mode=="metal"){
        if(!chain.compile_layer(3, std::string(model)+"/ane_layers", loader)){fprintf(stderr,"compile\n");return 2;}
        if(!chain.compile_metal_tails(loader, std::string(model)+"/ane_layers")) fprintf(stderr,"metal tails failed (continuing)\n");
        else fprintf(stderr,"metal tails compiled\n");
    } else {
        // Layer 3 = GDN layer (ch=16480), same index the engine validated.
        if(!chain.compile_layer(3, std::string(model)+"/ane_layers", loader)){fprintf(stderr,"compile\n");return 2;}
    }
    const size_t core_dim=6144,H=5120;
    constexpr size_t FIXED_LANES=4;
    // Fixed per-lane inputs so lane-l slices are comparable across widths.
    std::vector<std::vector<uint16_t>> core0(FIXED_LANES), res0(FIXED_LANES);
    for(size_t l=0;l<FIXED_LANES;++l){
        core0[l].resize(core_dim); res0[l].resize(H);
        for(size_t c=0;c<core_dim;++c) core0[l][c]=rnd16();
        for(size_t c=0;c<H;++c) res0[l][c]=rnd16();
    }
    bool all_pass=true;
    unsigned long long ref[FIXED_LANES]={0,0,0,0};
    for(size_t L : {1u,2u,3u,4u,8u,16u}){
        std::vector<uint16_t> core(core_dim*L), res(H*L), out, np;
        for(size_t l=0;l<L;++l){
            const auto& cl = core0[l<FIXED_LANES?l:FIXED_LANES-1];
            const auto& rl = res0[l<FIXED_LANES?l:FIXED_LANES-1];
            if(l>=FIXED_LANES){ // lanes beyond 4: fresh random per width (don't care)
                for(size_t c=0;c<core_dim;++c) core[c*L+l]=rnd16();
                for(size_t c=0;c<H;++c) res[c*L+l]=rnd16();
            } else {
                for(size_t c=0;c<core_dim;++c) core[c*L+l]=cl[c];
                for(size_t c=0;c<H;++c) res[c*L+l]=rl[c];
            }
        }
        if(!chain.evaluate_tail_batch(3, core.data(), core_dim, res.data(), L, out, &np)){
            fprintf(stderr,"L=%zu evaluate failed\n", L); return 3;
        }
        const size_t ch = np.size()/L;
        printf("[%s] L=%2zu ch=%zu", mode.c_str(), L, ch);
        // Sanity: per-lane HIDDEN hashes must be distinct (validates harness).
        for(size_t l=0;l<L && l<FIXED_LANES;++l){
            std::vector<uint16_t> hs(out.size()/L);
            for(size_t c=0;c<hs.size();++c) hs[c]=out[c*L+l];
            printf(" | hid%zu=%llx", l, fnv(hs));
        }
        // np per-lane hashes.
        for(size_t l=0;l<L && l<FIXED_LANES;++l){
            std::vector<uint16_t> slice(ch);
            for(size_t c=0;c<ch;++c) slice[c]=np[c*L+l];
            unsigned long long h=fnv(slice);
            if(ref[l]==0) ref[l]=h;
            const bool ok = (h==ref[l]);
            if(!ok) all_pass=false;
            printf(" | np%zu=%llx%s", l, h, ok?"":" DIVERGE");
        }
        // Raw first-8 np bytes per lane for L=4 (visual diff).
        if(L==4){
            for(size_t l=0;l<4;++l){
                printf("\n  raw lane%zu:", l);
                for(size_t c=0;c<8;++c) printf(" %04x", np[c*L+l]);
            }
        }
        printf("\n");
    }
    printf(all_pass?"NP_WIDTH_INVARIANT: PASS\n":"NP_WIDTH_INVARIANT: FAIL\n");
    return all_pass?0:1;
}
