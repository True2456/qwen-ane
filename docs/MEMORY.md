# What the process actually holds

A run was driving a 128 GiB machine to 117 GiB in use with nothing left free,
for a model whose weights are 68 GB. `vm_stat` arithmetic could not settle
where it went; `footprint -p` could.

| | before | after |
| --- | --- | --- |
| process footprint | 98 GB | 79 GB |
| expert bank (IOAccelerator, wired) | 68 GB | 67 GB |
| host allocation (Malloc Large) | 24 GB | 5.3 GB |
| ANE weight dictionaries (Foundation) | 5.1 GB | 5.1 GB |
| IOSurfaces | 0.5 GB | 0.5 GB |

Two things were being held for no reason.

**Every layer's raw weights, for the life of the process.** `layer_cache` in
the exporter exists so the MIL builds, the host mixers and the Core AI asset
paths can each ask for a layer without re-reading it. Nothing reads it once the
programs are baked. Releasing it after setup is worth 12 GB.
`FLASHNEXT_KEEP_LAYER_CACHE=1` puts it back.

**The weight blobs handed to the ANE compiler.** `compile_multiproc` kept them
on `prog._keep_alive` after the model was compiled *and loaded*, and they are
also on disk under the model's `localModelPath`, which is how the compile cache
reloads a program without recompiling. Dropping them after a successful load is
worth another 7 GB. `Q38_ANE_KEEP_WEIGHT_BLOBS=1` puts them back.

Neither changes a number. `probes/mil_k_check.py` prints the same per-slot
errors at K=4 and K=32, a decode run emits the same 64 token ids at 21.3 tok/s,
and 1024 tokens scored after a 512-token prefill give the same 1.886208.

## What is left

67 GB of expert bank, wired, which is the model and is not going anywhere while
it runs.

5.3 GB of host allocation that is live: `malloc_zone_pressure_relief` returns
zero, so it is not spans libmalloc is sitting on. Not attributed yet. A scan of
gc-tracked arrays over 4 MB finds nothing, which means it is held through
views, torch tensors, or an allocator Python cannot see.

5.1 GB of Foundation objects, unchanged by dropping our own references to the
weight blobs, so something inside the framework retains them. The loaded model
object is the obvious suspect, and it cannot be released while the program is
in use. Re-creating the model from the compiled artifact on disk rather than
from a weights dictionary would be the way to test that.

## Not a leak

Memory is flat across requests. Twelve consecutive MMLU-sized prompts moved the
footprint by less than 1 GB, and every process returns the machine to about
18 GB used and 98 GB free when it exits. The peak is at load, where a 71 GB
file is read through the page cache into 68 GB of arrays, and that window is
still the tightest moment in a run.
