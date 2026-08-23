/* Bit-exactness harness for the register-blocked MMA int4 GEMM.
 * Verifies gemm_int4_simd == gemm_int4_groupwise_batch bitwise at the real
 * gate shape (34816x5120, lanes=32). Run: make test-gemm-simd-exact */
#include "runtime/metal_engine.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
static double dec(uint16_t h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,u;
 if(e==0)u=s|(m?0:0);else if(e==31)u=s|0x7f800000|(m<<13);else u=s|((e-15+127)<<23)|(m<<13);
 float f;std::memcpy(&f,&u,4);return f;}
int main() {
    MetalContext* ctx = metal_context_create();
    const int rows=34816, cols=5120, lanes=32, groups=cols/64, packed=cols/8;
    std::vector<uint8_t> w((size_t)rows*packed*4);
    std::vector<uint16_t> sc((size_t)rows*groups,0x3c00u), bi((size_t)rows*groups,0);
    for(size_t i=0;i<w.size();++i) w[i]=(uint8_t)((i*7u)&0x7f);
    std::vector<uint16_t> a((size_t)cols*lanes,0x3c00u);
    auto W=metal_buffer_create(ctx,w.size()); auto S=metal_buffer_create(ctx,sc.size()*2);
    auto B=metal_buffer_create(ctx,bi.size()*2); auto A=metal_buffer_create(ctx,a.size()*2);
    auto C1=metal_buffer_create(ctx,(size_t)rows*lanes*2); auto C2=metal_buffer_create(ctx,(size_t)rows*lanes*2);
    std::memcpy(metal_buffer_get_contents(W),w.data(),w.size());
    std::memcpy(metal_buffer_get_contents(S),sc.data(),sc.size()*2);
    std::memcpy(metal_buffer_get_contents(B),bi.data(),bi.size()*2);
    std::memcpy(metal_buffer_get_contents(A),a.data(),a.size()*2);
    {auto c=metal_command_buffer_create(ctx);
     metal_dispatch_gemm_int4_groupwise_batch(ctx,c,A,W,S,B,C1,rows,cols,packed,groups,lanes);
     metal_command_buffer_commit(c);metal_command_buffer_wait(c);}
    {auto c=metal_command_buffer_create(ctx);
     metal_dispatch_gemm_int4_simd(ctx,c,A,W,S,B,C2,rows,cols,packed,groups,lanes);
     metal_command_buffer_commit(c);metal_command_buffer_wait(c);}
    uint16_t* p1=(uint16_t*)metal_buffer_get_contents(C1);
    uint16_t* p2=(uint16_t*)metal_buffer_get_contents(C2);
    // CPU expected for row 0, lanes 0: sum of dequantized nibbles
    uint32_t* w32=(uint32_t*)w.data();
    double exp=0;
    for(int c8=0;c8<packed;++c8){uint32_t word=w32[c8];
        for(int j=0;j<8;++j){uint32_t q=(word>>(j*4))&0xf; exp+=q;}}
    printf("row0 lane0: batch=%.2f simd=%.2f cpu=%.2f\n",dec(p1[0]),dec(p2[0]),exp);
    size_t mism=0; double maxd=0; int firstm=-1;
    for(size_t i=0;i<(size_t)rows*lanes;++i){ if(p1[i]!=p2[i]){ if(firstm<0)firstm=(int)i; ++mism;
        maxd=std::max(maxd,std::fabs(dec(p1[i])-dec(p2[i])));}}
    printf("mism=%zu/%zu maxdiff=%.3f first_mismatch_idx=%d (row=%d lane=%d): batch=%.2f simd=%.2f\n",
        mism,(size_t)rows*lanes,maxd,firstm,firstm/lanes,firstm%lanes,
        dec(p1[firstm]),dec(p2[firstm]));
    // distribution: check a few rows
    for(int r : {0, 31, 32, 33, 1000})
        printf("row%-5d batch=%.2f simd=%.2f\n", r, dec(p1[(size_t)r*lanes]), dec(p2[(size_t)r*lanes]));
    return 0;
}
