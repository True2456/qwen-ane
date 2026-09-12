# Chunked Flash-Next GDN prefill experiment

Work in progress, 2026-09-13. Baseline commit: `8fa3605`. Experimental switch:
`MIL_GDN_CHUNK=1`, applied only to `single_state=True` prefill graphs.
`MIL_K_SINGLE_STATE=1` makes the layer probes exercise that contract.
Decode retains the original prefix-state recurrence.

Implementation follows Yang, Kautz and Hatamizadeh,
[Gated Delta Networks, section 3.3](https://arxiv.org/html/2412.06464v3#S3.SS3).
With Q/K/V rows representing tokens, gamma_i=product(g_0..g_i) and
D_ij=product(g_(j+1)..g_i) for i>=j:

```
A = strictLower(diag(beta) (K K^T * D))
R = (I + A)^-1
Delta = R [diag(beta) (V - diag(gamma) K S_initial^T)]
Y = diag(gamma) Q S_initial^T + ((Q K^T) * D) Delta
S_final = gamma_last S_initial + Delta^T (diag(D_last,:) K)
```

The output uses inclusive causal D (including its diagonal), so it observes
the updated state. The initial-state term is present in both the residual
solve and output. State is only updated at the chunk boundary. There are
three fixed initial-state consumers, independent of token count; this is not
a claim that the compiler physically reads the state exactly once.

The inverse uses a finite geometric series: P=-A, R=I+P; repeatedly P=P²,
R=R+P R. For C=32 this is four pairs of matmuls after initialization (eight
matmuls), not a claimed five total matmuls. It covers degree 31 exactly in
real arithmetic because A is strictly lower triangular. Decay and prefix
products use a five-level product scan, avoiding reciprocal underflow and
log(0). Matrix work is padded to C=32 for smaller live chunks; neutral tokens
have gate 1, beta 0, and Q/K/V 0. I/O widths remain 32/64/128 and state output
names are zero padded. Existing normalization reduces the 128-element axis.

## Baseline measurements

`results/gdn_chunk/baseline_checks.log`: 301 repeats for layer medians and
submit-only medians; the existing wide probe used its original 31 repeats.
These initial samples were taken while another session was using the machine;
controlled paired measurements are still required.

| K | worst mixed | final state | conv | layer ms | submit-only ms |
|---|---:|---:|---:|---:|---:|
|1|0.0269|0.00558|0.00981|1.348|1.277|
|4|0.0338|0.00613|0.01217|1.793|1.720|
|8|0.0321|0.00505|0.01171|2.353|2.244|
|16|0.0344|0.00477|0.01128|3.522|3.378|
|32|0.0344|0.00652|0.01129|5.847|5.745|

## Negative results and fixes so far

- Raw, unscaled chunk matmuls at K=32: mixed 0.0583–0.1038 (FAIL), state
  0.00731, conv 0.01129, 2.375 ms/layer (25 repeats, uncontrolled timing).
  Multiplying Q by 64 before either output matmul and dividing Y by 64 before
  normalization changes mixed to 0.0263–0.0340, state 0.00731, conv 0.01129.
  The initial 2.370 ms measurement is not a qualified speedup; full gates pending.
- Separate mask blobs: `verifyBundleAtPath: hash mismatch` / invalid model.
  Appending masks to the established scale blob avoids this failure.
- P@P in the inverse: compiler duplicate tensor name in height-fold conversion;
  explicitly transposing the second P did not fix it. Computing -(P@(-P)) did.
- Internal C=4 matrix work then failed instruction validation. C=32 compiled.
- Three full-model baseline attempts stopped after MIL L5 load failed with
  `no ANE resources` (Code 54, underlying 0x5), then attempted Core AI fallback.
  Explicit Q38_ANE_KEEP_WIRED=0 did not help. These runs yield no valid rate or
  quality result. Another session was repeatedly launching 132-model runs.
- Compiler system log at 07:28:42 recorded `F_PREALLOCATE ... No space left on
  device`, requesting growth from 1,159,725,056 to 1,432,354,816 bytes, then a
  268,435,456-byte allocation failure. It cannot be attributed to a particular
  graph from the redacted system log. `df -h /System/Volumes/Data` reported
  128 GiB initially, 102 GiB during overlapping runs, then 120 GiB after exit.
  No scratch relocation or disk cleanup was performed.

## Validation still pending

All-width final chunk checks, isolated slope, full-model quality against
6.58 ±0.01, MLX reference on the identical scored interval, and end-to-end
511-token prefill with breakdown. No production default has been changed.
