#!/usr/bin/env python3
"""Isolate the chunk arithmetic from projections; compare against NumPy WY.

Outputs selected with --tap make the compiler prune downstream work, allowing
cumulative stage timings. These are schedules, not additive per-node costs.
"""
import argparse
import time
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
for path in [ROOT,ROOT/'scripts',ROOT/'probes']: sys.path.insert(0,str(path))
import flashnext_mil_layer as m
from flashnext_mil_chunk import gdn_chunk
from gdn_chunk_reference import chunk
from runtime.q38_ane_engine import _BlobPacker, _iosurface_view


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tap', default='all',choices=['all','cdecay','ci4','cdelta','cy','q_state31'])
    ap.add_argument('--repeats',type=int,default=301)
    args=ap.parse_args()
    h,c,d=48,32,128
    m.B.clear()
    m.emit('tensor<bool, [4]> mm = const()[name=string("mm"), val=tensor<bool, [4]>([false,false,false,false])];')
    m.emit('tensor<int32, [4]> pm = const()[name=string("pm"), val=tensor<int32, [4]>([0,1,3,2])];')
    m.emit('fp16 nho = const()[name=string("nho"), val=fp16(-1)];')
    def prepare(t):
        for name,src in [('qn48','a_q'),('kn48','b_k'),('vv','c_v'),('dec','d_g'),('bet','f_b')]:
            width = 1 if name == 'bet' else d
            m.sl4(f'g{t}{name}',src,(0,0,t,0),(1,h,t+1,width),(1,h,1,width))
    m.gdn_prepare=prepare
    m.gdn_finish=lambda t: None
    pack=_BlobPacker();offs={}
    for name,mask in [('lower',np.tril(np.ones((c,c)))),('strict',np.tril(np.ones((c,c)),-1)),('eye',np.eye(c))]:
        offs['chunk_'+name]=pack.append(mask.astype(np.float16).tobytes())+64
    row,col=np.arange(c)[:,None],np.arange(c)[None,:]
    for stage in range(5):
        half=1<<stage
        mask=(row//(2*half)==col//(2*half)) & ((row//half)%2==1) & ((col//half)%2==0)
        offs[f"chunk_block{stage}"]=pack.append(mask.astype(np.float16).tobytes())+64
    gdn_chunk(m,c,offs)
    tap=args.tap
    names=['cy','q_state31'] if tap=='all' else [tap]
    shapes={'cy':(1,h,c,d),'q_state31':(1,h,d,d),'cdecay':(1,h,c,c),'ci4':(1,h,c,c),'cdelta':(1,h,c,d)}
    sig=[]
    for name in ['a_q','b_k','c_v','d_g','e_state','f_b']:
        shape=(1,h,d,d) if name=='e_state' else (1,h,c,d)
        sig.append(f'tensor<fp16, [{", ".join(map(str,shape))}]> {name}')
    mil=f'program(1.3)\n{m.E._BUILD_INFO}\n{{\n func main<ios18>({", ".join(sig)}) {{\n'+ '\n'.join(m.B)+f'\n }} -> ({", ".join(names)});\n}}'
    t0=time.perf_counter()
    prog=m.eng.compile_multiproc(mil,{'weight_scale.bin':pack.getvalue()},h*c,h*c,d,raw_weight_files=frozenset(['weight_scale.bin']))
    if prog is None: raise RuntimeError('core compile failed')
    prog.input_elems=[h*c*d]*4+[h*d*d,h*c*d]
    prog.output_elems=[int(np.prod(shapes[name])) for name in sorted(names)]
    m.eng._ensure_io(prog)
    rng=np.random.default_rng(20260913)
    q,k=[rng.normal(size=(1,h,c,d)).astype(np.float32) for _ in range(2)]
    k/=np.linalg.norm(k,axis=-1,keepdims=True)
    q/=np.linalg.norm(q,axis=-1,keepdims=True)*np.sqrt(d)
    v=rng.normal(0,.02,size=q.shape).astype(np.float32)
    state=rng.normal(0,.02,size=(1,h,d,d)).astype(np.float32)
    gates=rng.uniform(.1,1,size=(1,h,c,1)).astype(np.float32)
    beta=rng.uniform(0,1,size=gates.shape).astype(np.float32)
    arrays=[q,k,v,np.broadcast_to(gates,q.shape),state,np.broadcast_to(beta,q.shape)]
    arrays=[np.asarray(x,np.float16) for x in arrays]
    for surf,arr in zip(prog._in_surfs,arrays):
        with _iosurface_view(surf,arr.shape,np.float16) as dst: np.copyto(dst,arr)
    ref_y,ref_state=chunk(*[arr.astype(np.float64) for arr in arrays[:3]],arrays[3][...,0].astype(np.float64),arrays[5][...,0].astype(np.float64),arrays[4].astype(np.float64))
    for _ in range(12): assert m.eng.submit(prog,procedure_index=0)
    ts=[]
    for _ in range(args.repeats):
        t0=time.perf_counter();assert m.eng.submit(prog,procedure_index=0);ts.append((time.perf_counter()-t0)*1e3)
    print(f'tap={tap} median_ms={np.median(ts):.6f} p10={np.percentile(ts,10):.6f} p90={np.percentile(ts,90):.6f}')
    for name,surf in zip(sorted(names),prog._out_surfs):
        with _iosurface_view(surf,shapes[name],np.float16) as src: got=np.array(src,np.float64)
        if name in ['cy','q_state31']:
            ref=ref_y if name=='cy' else ref_state
            print(f'{name}: rel={np.linalg.norm(got-ref)/np.linalg.norm(ref):.8f} max={np.max(np.abs(got-ref)):.8g} finite={np.isfinite(got).all()}')

if __name__=='__main__': main()
