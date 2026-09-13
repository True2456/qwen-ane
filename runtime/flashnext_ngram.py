"""Read-only checkpoint-backed PLE lookup and single-token CPU execution.

Shard order and hash parameters come from checkpoint metadata/tensors. The
large table is never materialized: each lookup uses pread for selected rows.
"""
import bisect
import json
import os
import re
import struct
import time
from pathlib import Path
import numpy as np


class NgramRows:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}
        self.entries, self.starts = [], []
        total = 0
        pattern = r"ngram_embedding\.shard_(\d+)\.weight$"
        keys = [k for k in self.index if re.search(pattern, k)]
        number = lambda k: int(re.search(pattern,k).group(1))
        keys.sort(key=number)
        config = json.loads((self.root/"config.json").read_text())
        self.config = config.get("text_config", config)
        count = self.config["split_ngram_parts"]
        if [number(k) for k in keys] != list(range(count)):
            raise ValueError("expected exactly one complete ordered n-gram shard set")
        for key in keys:
            fd, base, header = self._file(self.index[key])
            meta = header[key]
            rows, dim = meta["shape"]
            if meta["dtype"] != "BF16" or dim != self.config["ple_embed_dim"] // ((self.config["ngram_size"]-1)*self.config["heads_per_ngram"]):
                raise ValueError(f"invalid n-gram tensor {key}")
            start, end = meta["data_offsets"]
            if end-start != rows*dim*2 or base+end > os.fstat(fd).st_size:
                raise ValueError(f"invalid tensor extent {key}")
            self.starts.append(total)
            self.entries.append((fd, base+start, rows, dim, key))
            total += rows
        self.total, self.dim = total, self.entries[0][3]

    def _file(self, name):
        if name not in self.files:
            fd = os.open(self.root/name, os.O_RDONLY)
            size = struct.unpack("<Q",os.pread(fd,8,0))[0]
            header = json.loads(os.pread(fd,size,8))
            self.files[name] = (fd,size+8,header)
        return self.files[name]

    def tensor(self, key):
        fd, base, header = self._file(self.index[key])
        m=header[key]
        lo,hi=m["data_offsets"]
        data=os.pread(fd,hi-lo,base+lo)
        if len(data)!=hi-lo: raise IOError(f"short tensor read: {key}")
        if m["dtype"]=="BF16":
            return (np.frombuffer(data,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(m["shape"])
        dtype={"I64":np.int64,"F32":np.float32,"F16":np.float16}[m["dtype"]]
        return np.frombuffer(data,dtype).reshape(m["shape"]).copy()

    def lookup(self, ids):
        ids=np.asarray(ids,np.int64)
        if np.any(ids<0) or np.any(ids>=self.total): raise IndexError("n-gram row outside table")
        out=np.empty((ids.size,self.dim),np.uint16)
        for j,row in enumerate(ids.flat):
            s=bisect.bisect_right(self.starts,int(row))-1
            fd,start,_,dim,_=self.entries[s]
            data=os.pread(fd,dim*2,start+(int(row)-self.starts[s])*dim*2)
            if len(data)!=dim*2: raise IOError("short n-gram row read")
            out[j]=np.frombuffer(data,np.uint16)
        return (out.astype(np.uint32)<<16).view(np.float32).reshape(*ids.shape,self.dim)

    def write_index(self, destination):
        destination=Path(destination)
        if self.root.resolve() in (destination.resolve(), *destination.resolve().parents):
            raise ValueError("index artifact must be outside the read-only model tree")
        destination.parent.mkdir(parents=True,exist_ok=True)
        data={"shards":[{"file":str(self.root/self.index[e[4]]),"key":e[4]} for e in self.entries],
              "total_rows":self.total,"dim":self.dim}
        destination.write_text(json.dumps(data,indent=2)+"\n")

    def close(self):
        for fd,_,_ in self.files.values(): os.close(fd)
        self.files.clear()


class CpuPLE:
    def __init__(self, rows, layer):
        self.rows=rows
        cfg=rows.config
        self.h,self.hc=cfg["hidden_size"],cfg["hc_count"]
        self.eps=cfg.get("rms_norm_eps",1e-6)
        self.ng,self.heads=cfg["ngram_size"],cfg["heads_per_ngram"]
        eos=cfg["eos_token_id"]
        self.history=[eos[0] if isinstance(eos,list) else eos]*(self.ng-1)
        p=f"model.language_model.layers.{layer}.ple."
        self.weights={n:rows.tensor(p+n+".weight") for n in ("key_proj","value_proj","norm_key","norm_query","norm_conv","conv1d")}
        self.hash={n:rows.tensor(p+"ple_embedding."+n) for n in ("layer_multipliers","ngram_heads_offsets","ngram_heads_vocab_sizes")}
        self.conv=np.zeros(((cfg["ple_conv_kernel_size"]-1)*self.ng,self.h*self.hc),np.float32)
        self.last_lookup_ms=0.

    def hash_ids(self, token):
        history=[int(token)]+list(reversed(self.history))
        mult=self.hash["layer_multipliers"]
        terms=np.multiply(np.asarray(history,np.int64),mult,dtype=np.int64)
        ids=[]
        mixed=terms[0]
        for n in range(2,self.ng+1):
            mixed=np.bitwise_xor(mixed,terms[n-1])
            sl=slice((n-2)*self.heads,(n-1)*self.heads)
            ids.extend((mixed % self.hash["ngram_heads_vocab_sizes"][sl]+self.hash["ngram_heads_offsets"][sl]).tolist())
        return np.asarray(ids,np.int64)

    def norm(self,x,name):
        a=x.reshape(self.hc,self.h)
        return (a/np.sqrt(np.mean(a*a,axis=-1,keepdims=True)+self.eps)).reshape(-1)*(self.weights[name]+1)

    def step(self,hidden,token):
        t=time.perf_counter()
        emb=self.rows.lookup(self.hash_ids(token)).reshape(-1)
        self.last_lookup_ms=(time.perf_counter()-t)*1e3
        self.history=(self.history+[int(token)])[-(self.ng-1):]
        key=self.norm(self.weights["key_proj"]@emb,"norm_key").reshape(self.hc,self.h)
        query=self.norm(hidden.reshape(-1),"norm_query").reshape(self.hc,self.h)
        gate=(key*query).sum(-1,keepdims=True)/np.sqrt(self.h)
        gate=np.sqrt(np.maximum(np.abs(gate),1e-6))*np.sign(gate)
        value=self.weights["value_proj"]@emb
        gated=(value[None,:]/(1+np.exp(-gate))).reshape(-1)
        xp=np.concatenate((self.conv,self.norm(gated,"norm_conv")[None]),axis=0)
        taps=self.weights["conv1d"].reshape(self.h*self.hc,-1)
        out=(xp[::self.ng].T*taps).sum(-1)
        self.conv=xp[1:].copy()
        out=out/(1+np.exp(-np.clip(out,-80,80)))
        return hidden+(gated+out).reshape(hidden.shape)

# ---------------------------------------------------------------
# Draft-side suffix matching. Unrelated to the PLE tables above: this
# one indexes the tokens of the current turn so speculation can guess
# from repetition instead of from the MTP head.


class ContextLookup:
    """Last-occurrence index over n-grams of the tokens seen so far."""

    __slots__ = ("ids", "g", "index", "hits", "calls")

    def __init__(self, g: int | None = None):
        # Three is the shortest match that is worth trusting. Two fires
        # constantly on common bigrams and proposes noise; four rarely fires
        # on anything the head was not already going to get right.
        if g is None:
            g = int(os.environ.get("FLASHNEXT_NGRAM_G", "3"))
        self.g = max(1, int(g))
        self.ids: list[int] = []
        self.index: dict[tuple, int] = {}
        self.hits = 0
        self.calls = 0

    def extend(self, tokens) -> None:
        """Append confirmed tokens and index the n-grams they complete."""
        g = self.g
        ids = self.ids
        start = len(ids)
        ids.extend(int(t) for t in tokens)
        # An n-gram ending at position p - 1 points at p. Re-index from g
        # positions back, because the tail of the previous call only became a
        # complete n-gram once these tokens arrived.
        for p in range(max(g, start), len(ids)):
            self.index[tuple(ids[p - g:p])] = p

    def next_token(self, drafted) -> int | None:
        """The token that last followed this suffix, or None."""
        self.calls += 1
        g = self.g
        d = len(drafted)
        if d >= g:
            key = tuple(int(t) for t in drafted[d - g:])
        else:
            tail = self.ids[len(self.ids) - (g - d):] if g > d else []
            if len(tail) < g - d:
                return None
            key = tuple(tail) + tuple(int(t) for t in drafted)
        p = self.index.get(key)
        if p is None or p >= len(self.ids):
            return None
        self.hits += 1
        return self.ids[p]

    def stats(self) -> str:
        return f"{self.hits}/{self.calls}"
