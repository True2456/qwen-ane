# The Apple Neural Engine on M5 Max: access path, API, and constraints

Probed 2026-08-19 on this machine. Everything in §1–§5 is read off the running
system or extracted from shipping binaries. §6 onward is analysis and is marked
as such. Reproduction commands are in §8.

Machine: MacBook Pro `Mac17,6`, Apple M5 Max, 128 GB, 18 CPU cores (6 Super +
12 Performance).

---

## 1. The hardware, as the system reports it

From the `H11ANEIn` driver node's `DeviceProperties`:

| property | value |
|---|---|
| `ANEDevicePropertyNumANECores` | **16** |
| `ANEDevicePropertyTypeANEArchitectureTypeStr` | **`h17`** |
| `ANEDevicePropertyANEVersion` | 256 |
| `ANEDevicePropertyANEMinorVersion` | 17 |
| `ANEDevicePropertyANEHWBoardType` | 544 (`0x220`) |
| `ANEDevicePropertyANEHWBoardSubType` | 0 |
| `ANEDevicePropertyANECPUSubType` | 9 |
| `FirmwareLoaded` | Yes |

Driver identity is one generation behind the architecture string: `IOClass` is
`H11ANEIn`, the kext is `com.apple.driver.AppleH16ANEInterface`, and the
architecture reports `h17`. The `H11` prefix is vestigial naming carried since
the A11, not a version.

**There is exactly one ANE device node: `ane0`.** Not two. Supporting detail:

- DART mappers: `mapper-ane0`, `mapper-ane0-iso1` … `iso7`, `mapper-ane0-mpm`
  — one device with 7 isolation contexts.
- IOReport interrupt subgroups: `ane0 0` and `ane0 1` — two interrupt channels
  on the single device.
- Power management is binary: `MaxPowerState = 1`, idle at `CurrentPowerState = 0`.

This matters because both our stacks default `dual_ane` to true, and oMLX's own
comment attributes that design to "the reference M3 Ultra" — a two-die part.
See §7.

## 2. The ANE is behind an Exclave on this chip

```
"IONameMatch"   = "ane,t8132exclave"
"IOExclaveProxy" = Yes
+-o ANEExclaveProxy  <class ANEExclaveProxy>
exclave-service     = "com.apple.service.ANEExclave"
exclave-edk-service = "com.apple.service.ANEExclave_EDK"
```

`t8132` is the SoC identifier. The `exclave` suffix and the `ANEExclaveProxy`
node mean ANE access is mediated through Apple's Exclave boundary rather than
being a plain kernel-driver relationship.

**Consequence:** the M1-era raw-IOKit reverse engineering (tinygrad's ANE work,
the Asahi driver) was done on hardware without this. Do not assume that
poking `H11ANEInUserClient` directly transfers to `t8132`. The supported-ish
path through `AppleNeuralEngine.framework` demonstrably works (§3–§5); anything
below it is unverified on this chip.

User clients present: `H11ANEInUserClient`, `H11ANEInDirectPathClient`.

## 3. Access path

`/System/Library/PrivateFrameworks/AppleNeuralEngine.framework` — not linked,
`dlopen`'d at runtime, ObjC classes resolved by name. Supporting services:

| component | path |
|---|---|
| daemon | `/usr/libexec/aned` |
| user agent | `/usr/libexec/aneuserd` |
| compiler | `AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc` |
| storage | `AppleNeuralEngine.framework/XPCServices/ANEStorageMaintainer.xpc` |
| compiler fw | `/System/Library/PrivateFrameworks/ANECompiler.framework` |

No entitlement is required for an ordinary non-sandboxed binary — oMLX is a
normal app bundle and it works. App Store distribution is not possible.

### Exported classes

`dyld_info -exports` on the framework gives the full surface:

```
_ANEBuffer                      _ANEModel
_ANEChainingRequest             _ANEModelInstanceParameters
_ANEClient                      _ANEModelToken
_ANECloneHelper                 _ANEOutputSetEnqueue
_ANEDaemonConnection            _ANEPerformanceStats
_ANEDataReporter                _ANEPerformanceStatsIOSurface
_ANEDeviceController            _ANEProcedureData
_ANEDeviceInfo                  _ANEProgramForEvaluation
_ANEErrors                      _ANEProgramIOSurfacesMapper
_ANEHashEncoding                _ANEQoSMapper
_ANEIOSurfaceObject             _ANERequest
_ANEIOSurfaceOutputSets         _ANESandboxingHelper
_ANEInMemoryModel               _ANESharedEvents
_ANEInMemoryModelDescriptor     _ANESharedSignalEvent
_ANEInputBuffersReady           _ANESharedWaitEvent
_ANELog                         _ANEStrings
                                _ANEVirtualClient
                                _ANEWeight
```

Plus C functions, which name the accepted input formats outright:

```
_ANEValidateMILNetworkOnHost      _ANEValidateNetworkCreate
_ANEValidateMLIRNetworkOnHost     _ANEValidateNetworkCreateVMHost
_ANEGetValidateNetworkSupportedVersion
```

## 4. The working API sequence

Recovered from the ObjC selectors referenced by oMLX's shipping
`libomlx_qwen35_prefill_kernel_ops.dylib` (280 KB). This is the subset that is
known-good on this hardware, not the full framework surface.

**Build and load — no file on disk required:**

```
_ANEInMemoryModelDescriptor  modelWithMILText:weights:optionsPlist:
_ANEInMemoryModel            inMemoryModelWithDescriptor:
                             compileWithQoS:options:error:
                             loadWithQoS:options:error:
                             unloadWithQoS:error:
```

`modelWithMILText:weights:optionsPlist:` is the important one: **MIL as text, a
weights buffer, and an options plist.** You are not required to produce a
`.mlpackage` and run `coremltools`. You emit MIL source and hand it over.

**Options plist keys used:**

| key | meaning |
|---|---|
| `kANEFAneInstanceHint` | target instance, valid range 1–4 |
| `kANEFProcedureVariantHint` | procedure variant selection |

**Submit:**

```
_ANERequest  requestWithInputs:inputIndices:outputs:outputIndices:
                 weightsBuffer:perfStats:procedureIndex:
             evaluateWithQoS:options:request:error:
             inputSymbolIndicesForProcedureIndex:
             outputSymbolIndicesForProcedureIndex:
```

`procedureIndex:` is how one compiled program holds many procedures and you pick
one per call. This is what removes the per-dispatch cost that makes the CoreML
path unusable for this workload.

**Buffers — zero-copy with Metal:**

```
_ANEIOSurfaceObject  objectWithIOSurface:
MTLDevice            newBufferWithIOSurface:
```

One IOSurface, wrapped twice. The ANE and the GPU read and write the same
unified-memory allocation with no copy. This is what makes a split matmul —
part of the output columns on ANE, the rest on GPU — actually pay.

**Synchronisation — no CPU round trip:**

```
encodeSignalEvent:value:    encodeWaitForEvent:value:
setSignaledValue:           addCompletedHandler:
```

ANE completion and Metal command buffers rendezvous on shared events. The GPU
half of a split can be enqueued to wait on the ANE half directly.

## 5. What the compiled program actually looks like

oMLX generates MIL as text. Reconstructed from format strings in the dylib:

```
program(1.3)
  func main<ios18>(tensor<fp16, [1, C, 1, S]> x) {
    tensor<int32, [2]> st = const()[val=tensor<int32, [2]>([1,1])];   // strides
    tensor<int32, [2]> dl = const()[val=tensor<int32, [2]>([1,1])];   // dilations
    tensor<int32, [4]> pd = const()[val=tensor<int32, [4]>([0,0,0,0])];
    string pt = const()[val=string("valid")];
    int32  gr = const()[val=int32(1)];

    tensor<fp16, [O, I, 1, 1]> gw = const()[val=...BLOBFILE(
        path="@model_path/weights/gate.bin", offset=64)];

    tensor<fp16, [1, O, 1, S]> gate = conv(weight=gw, x=x, strides=st,
        dilations=dl, pad=pd, pad_type=pt, groups=gr)[name="gate"];
    tensor<fp16, [1, O, 1, S]> sig  = sigmoid(x=gate)[name="sigmoid"];
    tensor<fp16, [1, O, 1, S]> silu = mul(x=gate, y=sig)[name="silu"];
    tensor<fp16, [1, O, 1, S]> up   = conv(weight=uw, x=x, ...)[name="up"];
    tensor<fp16, [1, O, 1, S]> act  = mul(x=silu, y=up)[name="swiglu"];
    tensor<fp16, [1, D, 1, S]> y    = conv(weight=dw, x=act, ...)[name="down"];
  }
```

Multi-procedure programs use `func procedure%03zu<ios18>(...)` instead of
`main`, one per slice, in a single compiled artifact.

Five things worth extracting from that:

1. **There is no matmul.** A linear layer is a `conv` with `[O, I, 1, 1]`
   weights, unit stride, unit dilation, zero pad, `valid`, `groups=1`. Every
   projection is expressed as 1×1 convolution.
2. **Everything is 4D and fp16**, in `[1, C, 1, S]` layout — channels are the
   feature dim, H is pinned to 1, W carries the sequence.
3. **Opset is `ios18`**, program version `1.3`.
4. **Weights are external blobs**, referenced by path from the MIL text —
   `BLOBFILE(path="@model_path/weights/gate.bin", offset=64)`. A 64-byte header
   precedes the payload.
5. **Quantised weights dequantise in-graph** via
   `constexpr_blockwise_shift_scale(data=int8[...], scale=fp16[...])`. The
   compiler folds the dequant; the ANE does not see 4-bit weights. Note the
   observed scale tensor is templated `[N, 1, 1, 1]` — per-output-channel — which
   does not obviously correspond to the gs=64 blockwise scales on the MLX side.
   Unresolved; see §7.

## 6. Constraints, verbatim from the runtime

```
ANE hybrid input must be fp16 or bf16.
ANE instance hint must be between 1 and 4.
ANE SwiGLU weights must be contiguous rank-2 float32 arrays.
ANE bank weights must be contiguous rank-2 float32 MLX arrays.
ANE compile weight must be a contiguous rank-2 float32 MLX array.
Invalid fixed shape for ANE SwiGLU model.
Invalid fixed shape for ANE linear model.
Invalid ANE linear procedure bank shape.
ANE procedure has unexpected I/O symbols.
```

Plus, from the MLX-side patch:

- Fixed sequence length, exact match. `_eligible_input` requires
  `x.size // input_dim == config.sequence_length`. **There is no padding path** —
  a short chunk falls through to GPU entirely.
- `sequence_length` must be a multiple of 64 and ≥ 1024.
- Eligible weights only: affine, 4 or 5 bit, group_size 64 or 128. bf16 and
  8-bit builds are rejected.
- Compilation is eager and per-shape, at model load.

## 7. Measured: instance hints are inert, and `dual_ane` does not pay

Probed 2026-08-19 with a synthetic Qwen3.8-shaped MLP (H=5120, I=17408, 4-bit
affine gs64) driven through `_compile_pair` and `_backend` directly at S=4096.
Two independent runs, 7 iterations each, median reported. Script:
`artifacts/ane_probes/ane_dual_probe.py`.

At `fraction=0.53`, `alignment=128` (dual) gives `ane_outputs=9216`,
`gpu_outputs=8192`.

### Both submissions execute and overlap — but not on separate engines

| per call | run 1 | run 2 |
|---|---|---|
| `ane0_eval` | 40.6 ms | 34.8 ms |
| `ane1_eval` | 37.7 ms | 47.5 ms |
| `ane_region` (wall) | 52.7 ms | 56.1 ms |
| `gpu_qmm` | 52.9 ms | 55.0 ms |
| `ane0_launch` | 26.1 us | 28.8 us |
| `ane1_launch` | 29.1 us | 39.1 us |

`ane1_eval_ns` is non-zero and comparable to `ane0_eval_ns` in both runs.
**`kANEFAneInstanceHint` reaches hardware that genuinely executes in parallel**,
despite the IORegistry exposing a single `ane0` node — consistent with the two
IOReport interrupt channels (`ane0 0`, `ane0 1`) being separate engines inside
the 16-core block.

Overlap factor `(ane0+ane1)/ane_region` is 1.49x (run 1) and 1.47x (run 2).
Serial would be 1.0x; ideal two-way would approach 2.0x. So ~70% of ideal
two-way parallelism, stable across runs.

`ane1_eval_ns` is non-zero and comparable to `ane0_eval_ns`, and the overlap
factor `(ane0+ane1)/ane_region` is ~1.49x — real concurrency, well short of the
2.0x that two independent engines would give.

**The instance hint is inert.** Running the same hint twice is the control, and
it gives the same overlap as two different hints:

| hints | median | overlap |
|---|---|---|
| 1,2 | 60.91 ms | 1.488 |
| 3,4 | 61.69 ms | 1.482 |
| 1,3 | 62.22 ms | 1.496 |
| 2,4 | 61.48 ms | 1.489 |
| **1,1** | 61.59 ms | **1.486** |
| **3,3** | 61.55 ms | **1.490** |

If `kANEFAneInstanceHint` selected distinct hardware, `(1,1)` would serialise
toward 1.0x while `(1,2)` approached 2.0x. All six pairings sit at 1.48–1.50
with medians inside noise. The overlap is two asynchronous submissions
pipelining on a single engine; the hint does not steer them.

Hints 1–4 compile, 5 raises `ANE instance hint must be between 1 and 4`, 0 is
the unpinned default. Accepted is not the same as honoured.

This supersedes the earlier reading of a non-zero `ane1_eval_ns` as proof of
parallel hardware. The counter is real; the inference was not. The null control
is what settled it.

### End to end, dual is slightly slower

| | run 1 median | run 2 median |
|---|---|---|
| `dual_ane=True` | — | **77.61 ms** |
| `dual_ane=False` | **73.15 ms** | **75.57 ms** |

Single wins both. The mechanism is in the table above: `ane_region` and
`gpu_qmm` are within 2 ms of each other in both runs. At `fraction=0.53` the
ANE and GPU halves are already balanced, so the ANE side is not the critical
path — speeding it up internally by splitting into two banks cannot shorten the
call, and the split adds coordination.

**Implication:** `--no-ane-dual` is worth testing as the default on this
machine, and `fraction` is the parameter that actually matters, since it sets
where the balance point sits. Neither conclusion is settled — see caveats.

### Launch overhead is not a concern

26–39 us per launch, measured. The multi-procedure program design removes
dispatch cost as a design constraint; a per-expert or per-slice CoreML
`prediction` call would be one to two orders of magnitude worse.


### Numerical fidelity: the ANE path costs ~4x accuracy

Relative L2 error against an fp32 reference (dequantised weights, fp32 matmul),
same random 4-bit gs64 MLP, S=4096:

| path | rel. error vs fp32 ref |
|---|---|
| pure GPU (stock MLX quantised matmul) | **9.50e-4** |
| ANE+GPU hybrid, `dual_ane=True` | 4.12e-3 |
| ANE+GPU hybrid, `dual_ane=False` | 4.12e-3 |
| hybrid vs pure GPU | 4.13e-3 |

**The hybrid path is ~4.3x less accurate than pure GPU.** Dual and single are
bit-identical to each other, so the split is not the cause — the ANE path
itself is. Expected mechanism: `dense_slice` dequantises to fp32, the MIL
declares fp16, and the ANE accumulates in fp16, while MLX's quantised kernel
keeps more precision.

This is per-layer, on one layer. Whether it compounds across 64 layers of
prefill is not measured here. **Action:** confirm whether the 503/564
HumanEval/GSM8K/MMLU figures in `Q38-ENGINE-AND-SPECULATIVE-FINDINGS.md` were
taken with `ane_prefill` on. If they were, this error level is already
validated. If not, production is running an unvalidated numerical delta.

### Sequence length: the >=1024 rule is policy, not hardware

`qwen35_ane_compile_linear` compiles every length tried, with near-flat
compile time:

| S | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|
| compile (s) | 0.039 | 0.036 | 0.044 | 0.043 | 0.045 | 0.047 | 0.045 | 0.048 | 0.051 |

The `multiple of 64, >= 1024` check lives in `enable_qwen35_ane_prefill` and
`configure_qwen35_ane_prefill_scheduler` — it is an oMLX policy choice, not a
runtime constraint. An independent engine may compile short-sequence programs.
Compile time is flat in S, so it is dominated by weight handling, and may be
partly deferred to first evaluation.

### Effective throughput

18432 ANE rows (gate+up stacked) x K=5120 x S=4096 = **773 GFLOP per call**,
ANE region 58.61 ms:

**~13.2 TFLOP/s fp16** sustained across the ANE region, with the GPU half of
the same call taking 45.30 ms. Note the ANE side is the *slower* half in this
synthetic split, which suggests `fraction=0.53` over-allocates to the ANE.

### Caveats

- **One synthetic layer, not the model.** No thermal behaviour, no cross-layer
  cache effects, no real weights.
- **Not a clean room.** The oMLX server (4.1% CPU) and a q38 engine (0.1%) were
  resident. Per this repo's own rule that is tolerable but not ideal.
- **The ranges overlap**: dual [76.63, 77.92], single run 2 [75.26, 77.71]. The
  MTP norm A/B in `Q38-ENGINE-AND-SPECULATIVE-FINDINGS.md` required
  non-overlapping groups before a 9% effect was believed. A 2.7% gap with
  overlap does not clear that bar.
- **Random weights.** The numerics section uses random normal weights, not
  trained ones; error magnitudes on real weight distributions may differ.
- **Asymmetric instrumentation.** The non-dual path runs
  `qwen35_ane_q4_swiglu_t`, which reports all counters as zero. Only wall clock
  is comparable between the two configurations.

To settle it: run the same A/B on the loaded 27B across a real prefill sweep,
alternating configurations, with nothing else resident.

## 8. Analysis, and what is still unverified

Everything above is observed. This section is not.

**`dual_ane` is measured in §7** and is superseded there: the banks do run
in parallel, but the split does not pay because the ANE half is not the
critical path.

**The quantisation scale shape is unresolved.** §5 item 5. Either the per-channel
scale is a re-derivation for the ANE path, or the template hides a flattened
blockwise layout. Worth settling before trusting the ANE path to be numerically
equivalent to the GPU path — a silent precision difference between the two halves
of a split matmul is exactly the failure mode this repo keeps meeting.

**Decode is out of scope by construction.** The backend returns `None` on
`target_verify`, and a single decode token cannot match a ≥1024 fixed shape. The
ANE contributes to prefill only. Nothing here reduces bytes-per-token, so none
of it addresses decode throughput or the energy of a bandwidth-bound decode.

**Prior art does not transfer cleanly.** §2. Assume the Exclave boundary
invalidates M1-era IOKit findings until shown otherwise.

## 9. Reproducing the probes

```bash
# hardware identity
sysctl -n machdep.cpu.brand_string
system_profiler SPHardwareDataType | grep -E "Chip|Model Identifier|Memory"

# ANE device properties, instance count, exclave status
ioreg -rc H11ANEIn -d1 | grep -v IOReportLegend
ioreg -l | grep -o -E "ane[0-9]" | sort -u
ioreg -l | grep -io -E "AppleH[0-9]+ANE[A-Za-z]*|H11ANE[A-Za-z]*" | sort -u

# framework surface
dyld_info -exports \
  /System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/Current/AppleNeuralEngine \
  | grep -i "_ANE"

# the working call sequence and MIL template
D=/Applications/oMLX.app/Contents/Resources/omlx/custom_kernels/qwen35_prefill
strings -a "$D"/libomlx_qwen35_prefill_kernel_ops.dylib | grep -E "^[a-zA-Z][a-zA-Z0-9_]*(:[a-zA-Z0-9_]*)+:?$" | sort -u
strings -a "$D"/libomlx_qwen35_prefill_kernel_ops.dylib | grep -E "tensor<|func |program\(|_ANE|kANEF"

# runtime availability
O=/Applications/oMLX.app/Contents/Resources
PYTHONPATH="$O/Python/framework-mlx-base/lib/python3.11/site-packages:$O" \
  "$O/Python/cpython-3.11/bin/python3.11" -c \
  "from omlx.custom_kernels.qwen35_prefill import fast; print(fast.qwen35_ane_available())"
```

Power draw was not measured; `sudo powermetrics --samplers ane_power` is the
next step and needs a password this session did not have.

## 10. Building an independent engine

The path is viable and the recipe is §4. Restated as steps:

1. Emit MIL text for a fixed shape, linears as 1×1 `conv`, all fp16,
   `[1, C, 1, S]`, opset `ios18`.
2. Write weights to blob files with the 64-byte header; reference them from the
   MIL by `@model_path` path.
3. `dlopen` the framework; build an `_ANEInMemoryModelDescriptor` with
   `modelWithMILText:weights:optionsPlist:`, then compile and load.
4. Allocate IOSurfaces; wrap for ANE with `_ANEIOSurfaceObject objectWithIOSurface:`
   and for Metal with `newBufferWithIOSurface:`.
5. Submit with `_ANERequest`, selecting `procedureIndex:` per call.
6. Rendezvous with Metal on shared events.

You are not writing ANE machine code and not bypassing Apple's compiler. What
you gain over CoreML is the dispatch path: multi-procedure programs, direct
request submission, and shared IOSurfaces instead of feature-dictionary
marshalling. That difference is the entire reason this is worth doing.

What you inherit: fixed shapes compiled per shape, fp16 only, conv-shaped
graphs, a private API with no stability guarantee across macOS updates, and no
App Store distribution.

## 11. The in-memory model API, captured from a live caller

The guesswork in the previous revision is replaced by ground truth. lldb cannot
attach (the bundled Python is hardened without `get-task-allow`), so the
arguments were captured by **swizzling the method from inside the process** with
ctypes — no debugger, no entitlements. Harness: `artifacts/ane_probes/swizzle_capture.py`,
`artifacts/ane_probes/swizzle_compile.py`, `artifacts/ane_probes/dump_blobs.py`.

### The full MIL template

`artifacts/ane_probes/captured/mil.txt`, emitted by `qwen35_ane_compile_linear` for a
128x64 fp32 weight at S=1024:

```
program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
  func main<ios18>(tensor<fp16, [1, 64, 1, 1024]> x) {
    tensor<int8, [128, 64, 1, 1]> wd = const()[name=string("wd"), val=tensor<int8, [128, 64, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_data.bin"), offset=uint64(64)))];
    tensor<fp16, [128, 1, 1, 1]> ws = const()[name=string("ws"), val=tensor<fp16, [128, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_scale.bin"), offset=uint64(64)))];
    tensor<fp16, [128, 64, 1, 1]> w = constexpr_blockwise_shift_scale(data=wd, scale=ws)[name=string("dequant")];
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
    tensor<fp16, [1, 128, 1, 1024]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=x)[name=string("conv")];
  } -> (y);
}
```

The `[buildInfo = ...]` block is mandatory — its absence is why hand-written
programs returned a nil descriptor.

### Argument schema

| argument | type | content |
|---|---|---|
| `MILText` | NSData | UTF-8 MIL source above |
| `weights` | NSDictionary | key = the full `BLOBFILE` path string; value = `{ data: NSData, offset: NSNumber }` |
| `optionsPlist` | NSData | **zero bytes** — no keys required |

Blob layout (`artifacts/ane_probes/captured/*.bin`): 64-byte header
`01000000 02000000` followed by 56 zero bytes, payload at offset 64, trailing
padding to a 64-byte boundary. `weight_data.bin` is 8320 bytes for 128x64 int8;
`weight_scale.bin` is 384 bytes for 128 fp16.

### Compile arguments

`-[_ANEInMemoryModel compileWithQoS:options:error:]`, captured live:

```
qos     = 21
options = { kANEFAneInstanceHint = 1; kANEFProcedureVariantHint = 1; }
```

### This resolves the sec.8 quantisation question

`qwen35_ane_compile_linear` takes an fp32 weight and re-quantises it to **int8
with one fp16 scale per output channel** (`ws` is `[128, 1, 1, 1]` for a
`[128, 64]` weight; the 384-byte scale blob holds exactly 128 fp16 values).

The ANE never sees the model's 4-bit gs64 blockwise structure. That is the
mechanism behind the ~4.3x error measured in sec.7 — not fp16 accumulation as
first supposed, but a lossy int8 per-channel re-quantisation of already
-dequantised weights.

### Solved — by the q38 ANE engine

The standalone-compile failure above is solved in
`~/AppleLLM/q38_native_engine/runtime/q38_ane_engine.py`. The missing step is
that `_ANEInMemoryModel` exposes **`localModelPath`**, a hashed temp dir, and
the MIL and weight blobs must be materialised there *before* `compileWithQoS:`:

```python
local = _desc(_msg(model, "localModelPath"))
shutil.rmtree(local, ignore_errors=True)              # espresso leftovers shadow model.mil
os.makedirs(os.path.join(local, "weights"), exist_ok=True)
open(os.path.join(local, "model.mil"), "wb").write(mil_text)
```

That is the process state this doc's harness never reproduced — it called
compile without ever writing `model.mil` to disk. Three further corrections
from that engine:

- `saveModelFiles` is the **espresso** path and yields `InvalidMILProgram` for
  MIL models. Do not use it.
- `optionsPlist` may be passed as **nil**, not an empty NSData.
- Blob headers are **128 bytes with payload at 0x80**, not the 64 the MIL's
  `offset=uint64(64)` suggests.
- `hexStringIdentifier` is also exposed on the model.

One useful negative result did come out of it: **descriptor creation does not
validate the program.** `modelWithMILText:` returns non-nil for obviously
malformed MIL — wrong arity, wrong shapes, undefined ops. Only
`compileWithQoS:options:error:` validates. Any prober must reach the compile
step to mean anything.


## 12. Weights can be runtime inputs — MoE on ANE is not blocked

This supersedes the framing in secs. 4-5 that ANE weights are compile-time
constants with a baked DMA schedule. They need not be.
`generate_dynamic_linear_mil` in `q38_ane_engine.py` declares the weight as a
**second model input** and reshapes it into conv weight layout inside the graph:

```
func main<ios18>(tensor<fp16, [1, I, 1, S]> x,
                 tensor<fp16, [1, O, 1, I]> wimg) {
  tensor<int32, [4]> shp = const()[val=tensor<int32, [4]>([O, I, 1, 1])];
  tensor<fp16, [O, I, 1, 1]> w = reshape(shape=shp, x=wimg)[name="wr"];
  tensor<fp16, [1, O, 1, S]> y = conv(weight=w, x=x, ...)[name="conv"];
} -> (y);
```

The weight arrives as a feature map through an IOSurface, exactly like the
activations. `AneDynamicLinear` compiles once per `(input_dim, output_dim,
seq_len)` and then pages arbitrary `[O, I]` matrices in via `write_weight()`
without recompiling.

**Consequence for MoE:** one compiled program per expert *shape*, not per
expert. Routing selects which weights to page into the surface. An expert miss
costs `pread` + dequant + `write_weight` instead of a ~50 ms recompile. That is
what `q38_ane_moe.py` does for DeepSeek-V4-Flash with routed experts on SSD.

Gotcha recorded there: the compiled symbol order is typically `(wimg, x)` —
bind IO from `kANEFModelInputSymbolsArrayKey`, **not** MIL argument order.

**But it does not pay on this chip.** From `q38_ane_moe.py`: "ANE only wins the
compile tax, not tok/s vs Metal on this chip." Consistent with sec. 7 — the ANE
region measured ~13.2 TFLOP/s against a GPU half that finished sooner, and
decode is bandwidth-bound regardless of which engine issues the matmul. The
mechanism works; the economics do not.
