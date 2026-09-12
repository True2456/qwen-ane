#!/usr/bin/env python3
"""Compare the NumPy chunk equations to the local MLX per-token reference.

Run with the user's mlx-lm and mlx/python paths on PYTHONPATH. CPU only, so
this check does not compete with the ANE / GPU performance measurements.
"""
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import _gated_delta_step_ops, compute_g
from gdn_chunk_reference import chunk

mx.set_default_device(mx.cpu)
rng=np.random.default_rng(93)
for c in [1,4,8,16,32]:
    q,k=[rng.normal(size=(1,16,c,128)).astype(np.float32) for _ in range(2)]
    k/=np.linalg.norm(k,axis=-1,keepdims=True)
    q/=np.linalg.norm(q,axis=-1,keepdims=True)*np.sqrt(128)
    q,k=[np.repeat(a,3,axis=1) for a in [q,k]]
    v=rng.normal(0,.02,(1,48,c,128)).astype(np.float32)
    state=rng.normal(0,.02,(1,48,128,128)).astype(np.float32)
    # Exercise compute_g's actual [B,T,Hv] scalar-gate shape.
    a=mx.array(rng.normal(size=(1,c,48)).astype(np.float32))
    gate=compute_g(mx.zeros((48,)),a,mx.zeros((48,)))
    assert gate.shape==(1,c,48)
    gates=np.asarray(gate).transpose(0,2,1)
    beta=rng.uniform(0,1,(1,48,c)).astype(np.float32)
    expected_y,expected_s=chunk(q,k,v,gates,beta,state)
    st=mx.array(state);ys=[]
    for t in range(c):
        y,st=_gated_delta_step_ops(*[mx.array(a[:,:,t]) for a in [q,k,v,gates,beta]],st)
        ys.append(y)
    actual_y=np.asarray(mx.stack(ys,axis=2));actual_s=np.asarray(st)
    ey=float(np.linalg.norm(actual_y-expected_y)/np.linalg.norm(expected_y))
    es=float(np.linalg.norm(actual_s-expected_s)/np.linalg.norm(expected_s))
    print(f'K={c}: vs MLX CPU _gated_delta_step_ops y={ey:.9g} state={es:.9g}',flush=True)
    assert ey<2e-6 and es<2e-6
