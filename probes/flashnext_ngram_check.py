"""Validate checkpoint-derived SSD rows, hashing and PLE against MLX."""
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.flashnext_ngram import NgramRows, CpuPLE
from mlx_lm.models.qwen4_exp import TextConfig, PLELayer, NGramTable

root=Path("/Users/true/models/Qwen3.8-Flash-Next")
rows=NgramRows(root)
index=Path(__file__).resolve().parents[1]/"artifacts/coreai/ngram_index.json"
rows.write_index(index)
print(f"validated {len(rows.entries)} shards; {rows.total} rows x {rows.dim}; index={index}",flush=True)
cpu=CpuPLE(rows,1)
cfg=TextConfig.from_dict(rows.config)
model=PLELayer(cfg)
weights=[]
for name,weight in cpu.weights.items():
    if name.startswith("norm_"):weight=weight+1
    if name=="conv1d":weight=weight.transpose(0,2,1)
    weights.append((name+".weight",mx.array(weight)))
for name,weight in cpu.hash.items():
    weights.append(("ple_embedding."+name,mx.array(weight)))
model.load_weights(weights)
table=NGramTable(root,index_name=str(index))
captured=[]
def lookup(ids):
    captured.append(np.array(ids).reshape(-1))
    return table(ids).astype(mx.float32)
model.ple_embedding.set_lookup(lookup)
cache=[None,None,None,None]
rng=np.random.default_rng(919)
times=[]
for token in (760,220,17,15,16,16,4006,16,17,248044,760,220):
    hidden=rng.normal(0,.1,(1,1,10240)).astype(np.float32)
    expected_ids=cpu.hash_ids(token)
    t=time.perf_counter()
    result=cpu.step(hidden,token)
    times.append(cpu.last_lookup_ms)
    xx=mx.array(hidden)
    want=xx+model(xx,mx.array([[token]]),cache)
    mx.eval(want)
    np.testing.assert_array_equal(captured[-1],expected_ids)
    rel=np.linalg.norm(np.array(want)-result)/np.linalg.norm(np.array(want))
    assert rel<2e-5,(token,rel)
print(f"PASS: 12 PLE steps/hash histories vs MLX; row lookup median={np.median(times):.3f}ms max={max(times):.3f}ms",flush=True)
# Boundary checks use the independent mmap implementation.
ids=np.array([0,rows.total-1]+rows.starts[1:],np.int64)
got=rows.lookup(ids)
want=np.array(table(mx.array(ids.astype(np.int32))).astype(mx.float32))
np.testing.assert_array_equal(got,want)
print("PASS: all shard boundaries and final row match independent mmap lookup")
rows.close()
