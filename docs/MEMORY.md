# What the process actually holds

A run was driving a 128 GiB machine to 117 GiB in use with nothing left free,
for a model whose weights are 68 GB. `vm_stat` arithmetic could not settle
where it went; `footprint -p` could. RSS still cannot: MLX's Metal buffers
show up here, not in `ps`.

| | before this pass | after |
| --- | --- | --- |
| process footprint | 79 GB | **78 GB** |
| expert bank (IOAccelerator, wired) | 67 GB | 68 GB |
| host allocation (Malloc Large) | 5.3 GB | **4.26 GB** |
| ANE weight dictionaries (Foundation) | 5.1 GB, 109 regions | **5.10 GB, 144 regions** |
| IOSurfaces | 0.5 GB | **57 MB** |
| `phys_footprint_peak` | **93 GB** | **81 GB** |

The 144 Foundation regions are the decode programs plus the k=32 prefill
procedures sharing those programs (`FLASHNEXT_PREFILL_MIL_K=32` on that
sample). Bytes did not move. A decode-only load is 108 regions at the same
5.1 GB.

Two things that were being held for no reason, from the previous pass, are
still gone: `layer_cache` after setup (`FLASHNEXT_KEEP_LAYER_CACHE=1` puts it
back) and the ANE compiler blobs (`Q38_ANE_KEEP_WEIGHT_BLOBS=1` puts those
back). This pass was the leftover 11 GB and the 93 GB peak.

## Peak was not the 71 GB file in the unified buffer cache

Mid-load `footprint`, 24 of 48 MoE layers resident: 71 GB current and peak.
IOAccelerator 47 GB, Malloc Large 18 GB, Foundation 5.1 GB. The 71 GB
safetensors mapping sat in *clean* `mapped file` (~7 GB in that sample), which
`phys_footprint` does not count. Two `vm_stat` samples minutes apart giving
113 GB then 46 GB were that cache decaying, not the process shrinking.

The 93 GB peak was the 68 GB Metal bank overlapping the still-live
`layer_cache` (~12 GB of host tensors) and then the MTP drafter, because
`layer_cache.clear()` ran after all 48 `HostLayer`s *and* after the drafter.
Dropping each layer's leftover host tensors after that layer's `HostLayer` is
built, and clearing whatever remains *before* the drafter, takes the peak to
80–81. `FLASHNEXT_KEEP_LAYER_CACHE=1` restores the old overlap.

`F_NOCACHE` on the bank fd and `MADV_DONTNEED` after each tensor is copied
into an evaluated MLX array are still on (`FLASHNEXT_FILE_CACHE=1` restores
the old reads). They keep the file from sitting in the UBC next to the bank.
They were not what made `phys_footprint_peak` 93.

## Malloc Large, 5.3 GB → 4.26 GB

`malloc_zone_pressure_relief` still returns zero. The arrays are live.

A `gc.get_objects()` scan of numpy arrays over 4 MB, including views, reports
**0.00 GB**. Numpy's numeric arrays are not gc-tracked, so the previous scan
that filtered `o.base is None` was wrong *and* a view-aware gc scan is still
blind. `runtime/mem_report.py` walks `HostLayer` fields instead.

On a loaded mlxresident process those fields are:

| | |
| --- | --- |
| `moe._resident` (MLX, billed as IOAccelerator) | 63.7 GB |
| mixers `attn/mlp` `down_w`/`up_w` (fp32, live decode weights) | 4 × 0.586 = **2.34 GB** |
| `moe.router_w` | **0.234 GB** |
| inject / hc_norm | 18 MB |

mlxresident no longer keeps the BF16 expert slabs or a second fp32 copy of the
shared expert (~1 GB). Decode never read them; the GPU bank already holds
both. That is the 5.3 → 4.26. The mixers and the router stay. The remaining
~1.7 GB of Malloc Large is not in Python's heap: ANE/framework/drafter host
side.

A per-layer `gc.collect()` during the bank copy pushed Malloc Large to 6.4 GB
(arenas that did not go back). Collecting once at the end, after the cache
drop, is enough. `FLASHNEXT_MEM_REPORT=1` prints the walk.

## Foundation, 5.1 GB — lead did not pay

Dropping our own references to the compiler blobs left 5.1 GB in 108–109
regions, ~18 MB resident and ~51 MB swapped each. They are the `NSData` handed
to `_ANEInMemoryModel`. `probes/ane_compiled_from_disk.py` on a 32 MB blob:

- `hexStringIdentifier` is `MILHASH_WEIGHTSHASH_OPTIONSHASH`. The empty
  SHA-256 (`e3b0c442…`) is the options/plist slot, not the weights.
- Same MIL + the same weight bytes: identical ident, `compiledModelExists=True`
  after compile.
- Empty dictionary: different middle hash (`DF3F6198…`),
  **`compiledModelExists=False`**. Same-size zeros: different hash, miss.
- `_ANEInMemoryModel` has no `initWithURL:`. Only `localModelPath` /
  `modelURL` / `setModelURL:`. The ivar `_descriptor` retains `_weights`
  `NSData`. There is no `setWeights:`.
- `AneDirectEngine.load_compiled_package(localModelPath)` fails looking for
  espresso `model.espresso.net`.
- `_ANEModel.modelAtURL:key:` returns an object. Evaluate-via-InMemory from
  that URL was not proven and is not wired in.

The cache key includes the weight bytes. Passing an empty dictionary on a
cache hit misses. A model cannot be rebuilt from the compiled artifact on
disk without the weights dictionary, so the 5.1 GB stays for the life of the
loaded programs.

## Not a leak

Memory is still flat across requests. The process still returns the machine
when it exits. The tight window is load, and that window is now the loaded
size rather than 12 GB on top of it.

What is left that is not the 68 GB bank: Foundation 5.1 GB (stuck), mixers
2.34 GB + router 0.23 GB (decode), ~1.7 GB unattributed Malloc Large, the MTP
drafter, Malloc Small, IOSurface.

## Gates

Same path as the rest of this notebook:
`FLASHNEXT_SPEC=4 FLASHNEXT_MOE=mlxresident FLASHNEXT_HEAD=mlx FLASHNEXT_MIL_GDN=1 FLASHNEXT_MIL_QSA=1 FLASHNEXT_PLE=0`.
The ppl number 1.886208 is the zero-table, **k=32 MIL prefill** number
(`FLASHNEXT_PREFILL_MIL_K=32`). Walking the same 512 tokens through the K=4
decode graphs instead (`PREFILL_MIL_K` unset) scores 1.884456 — a different
prefill, not a weight change.

- `probes/mil_k_check.py` K=4: `t0=0.0282 t1=0.0341 t2=0.0321 t3=0.0289`
  state `0.00627` conv `0.01217`. K=32: same per-slot list as
  `results/gdn_chunk/chunk_blocked_all_k.log`, state `0.00726` conv `0.01129`.
- Decode, prompt `760`, 64 tokens: ids `[220, 17, 15, 16, 21, 4006, …, 220, 17]`,
  BF16 MATCH, 10.59 tok/s. (21.3 tok/s was a 32-token speculative prompt on
  the same graphs.)
- `--ppl-file eval/prose.txt --ppl-tokens 1024 --ppl-prefill 512` with
  `FLASHNEXT_PREFILL_MIL_K=32`: nll **1.886208** ppl **6.5943**, per-slot
  `s0=2.0543 s1=2.0408 s2=1.6981 s3=1.7516`, 23.5 tok/s. Prefill of 511 tokens
  at 86.4 tok/s.
