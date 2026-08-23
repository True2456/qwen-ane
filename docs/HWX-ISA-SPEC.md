# Apple Neural Engine (M5 Max / `h17`) Direct Execution & Model Specification

This specification documents the low-level model descriptors, ObjC API execution sequence, and hardware interface recovered on Apple Silicon M5 Max (`h17`).

---

## 1. Direct Execution Pipeline Architecture

Execution of custom ANE graphs bypasses CoreML entirely through the private `AppleNeuralEngine.framework` runtime:

```
┌──────────────────────────────────────────────────────────────┐
│                    Layer 3: Model Descriptor                 │
│               _ANEInMemoryModelDescriptor                    │
│   - MIL Program Text + Binary Weight Blobs (DEADBEEF 0x80)   │
│   - Options Plist: kANEFKeepModelMemoryWiredKey, Hints       │
└──────────────────────────────┬───────────────────────────────┘
                               │ compileWithQoS:
┌──────────────────────────────▼───────────────────────────────┐
│               Layer 2: Compiled ANE Model Object             │
│                          _ANEModel                           │
│   - programHandle (64-bit hardware handle, e.g. 0x52938642d2b)│
│   - queueDepth: 127 (hardware evaluation queue depth)        │
│   - modelAttributes: ANEFModelDescription + NetworkStatusList│
└──────────────────────────────┬───────────────────────────────┘
                               │ doEvaluateDirectWithModel:
┌──────────────────────────────▼───────────────────────────────┐
│              Layer 1: Direct Client & Silicon Mailbox        │
│                         _ANEClient                           │
│   - IOSurface-backed Zero-Copy Input / Output IO Buffers     │
│   - Asynchronous Execution & IOReport Interrupt Channels     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Low-Level Model Attributes & Memory Strides

The hardware requires precise tensor stride alignment in `modelAttributes`:

### A. Input / Output Tensor Descriptors (`NetworkStatusList`)
For a tensor of shape `[1, Channels, 1, Width]` with `Float16` elements (2 bytes/element):

| Property | Value Formula | Description |
|---|---|---|
| `Width` | $S$ (e.g. 32) | Sequence / spatial dimension |
| `Channels` | $C$ (e.g. 10240) | Channel / feature dimension |
| `Type` | `Float16` | 16-bit floating point |
| `RowStride` | $2 \times S$ bytes (e.g. 64) | Byte stride between consecutive spatial rows |
| `PlaneStride` | $2 \times S$ bytes (e.g. 64) | Byte stride across planes |
| `BatchStride` | $2 \times C \times S$ bytes | Total buffer size in bytes |
| `DepthStride` | $2 \times C \times S$ bytes | Depth stride matching batch size |
| `Interleave` | `1` | Planar channel-major ordering |

---

## 3. Weight Blob Binary Container (`0xDEADBEEF`)

Weights passed to the ANE compiler require a 128-byte binary envelope:

```
Offset       Size (Bytes)   Value / Content
────────────────────────────────────────────────────────────
0x0000       4              0x00000001 (Magic 1)
0x0004       4              0x00000002 (Magic 2)
0x0008       56             0x00...00  (Reserved Padding)
0x0040       4              0xDEADBEEF (Sub-header Signature)
0x0044       4              0x00000001 (Version)
0x0048       4              uint32 payload_size_in_bytes
0x004c       4              0x00000000 (Reserved)
0x0050       4              0x00000080 (Payload Start Offset = 128)
0x0054       44             0x00...00  (Header Padding)
0x0080+      payload_size   Raw Tensor Data (FP16 / INT4 packed)
```

**BLOBFILE offset semantics (empirically pinned by `ane-as` offset-probe):**
the `offset=uint64(64)` argument in MIL `BLOBFILE(...)` is **not** a byte
address into this container. Compiling the same graph with offsets
`0 / 32 / 128` fails outright with `InvalidMILProgram`; only `64` compiles,
regardless of payload size. Treat `uint64(64)` as a mandatory schema constant
in every synthesized MIL. The compiler resolves tensor data via the envelope's
own payload pointer (`0x0050`), not the MIL offset.

---

## 4. Validated Direct Objective-C API Sequence

```objc
// 1. Acquire direct connection to ANE daemon
_ANEClient* client = [_ANEClient sharedConnection];

// 2. Wrap IOSurfaces with zero-copy ANE objects
_ANEIOSurfaceObject* inObj  = [_ANEIOSurfaceObject objectWithIOSurface:inputSurf];
_ANEIOSurfaceObject* outObj = [_ANEIOSurfaceObject objectWithIOSurface:outputSurf];

// 3. Create request
_ANERequest* req = [_ANERequest requestWithInputs:@[inObj]
                                     inputIndices:@[@0]
                                          outputs:@[outObj]
                                    outputIndices:@[@0]
                                    weightsBuffer:nil
                                        perfStats:nil
                                   procedureIndex:0];

// 4. Submit directly to hardware (measured 0.150 ms dispatch overhead)
NSError* error = nil;
BOOL ok = [client doEvaluateDirectWithModel:aneModel
                                    options:@{}
                                    request:req
                                        qos:21
                                      error:&error];
```

---

## 5. Measured Performance (M5 Max, `ane-as --iters 300`, Aug 2026)

Dispatch overhead is noisy run-to-run (scheduler + thermal state); quote p50,
not means. Numbers from the native C++ suite (`make test-ane-as`, no Python):

| Workload | p50 | min | p90 | max |
|---|---|---|---|---|
| Direct dispatch (smoke conv) | 0.095 ms | 0.065 ms | 0.220 ms | 3.5 ms |
| Depthwise causal conv C=64, S=32, K=4 | 0.092 ms | 0.065 ms | 0.115 ms | 3.4 ms |
| Depthwise causal conv C=10240, S=32, K=4 | 0.099 ms | 0.076 ms | 0.182 ms | 3.4 ms |

MIL compile: 16-24 ms one-time per program.

Notes:
- Occasional ~3.4 ms stalls appear in every run; means are misleading.
- Numerical verification vs an fp32-accumulate scalar reference lands at
  RelErr 5-6e-4 for the depthwise convs, consistent with fp16 internal
  accumulation in the ANE kernel. PASS bar in `ane-as` is 1e-3 (warn 5e-3).
- Single-op dispatch is overhead-dominated (~0.1 ms mailbox round trip); ANE
  economics only favor fused multi-op graphs. `queueDepth: 127` asynchronous
  submission is the obvious next lever (not yet exercised here).
- Known environment quirk: binaries under `~/Desktop/...` get SIGKILL'd by the
  ANE daemon on this machine; run the `/tmp/rindi-ane-as` staging copy (the
  Makefile target already does).

---

## 6. Dispatch Concurrency, Real-Time Path, Fused Graphs & INT4 (Aug 2026)

Probed empirically by `ane-as` tests [4]-[7]:

**Real-time path** (`evaluateRealTimeWithModel:`): identical latency to
`doEvaluateDirectWithModel:` (p50 0.09-0.10 ms both). It is a scheduling
priority hint, not a fast path. No throughput benefit.

**Client-side concurrency does NOT pipeline dispatches.** With 1/2/4/8
threads each issuing synchronous direct dispatches on independent requests,
aggregate throughput stays flat at ~4-5 kHz while per-request latency grows
linearly (0.22 -> 1.93 ms/dispatch at 8 threads = pure queueing). The daemon
mailbox serializes evaluations; concurrency cannot hide the ~100 us round
trip. (`_ANEClient.doEnqueueSetsWithModel:outputSet:` + `_ANEOutputSetEnqueue`
isOpenLoop and `_ANEChainingRequest` exist for true async, but require the
program-surface IO mode - unexplored.) Consequence: fused programs are the
only way to amortize dispatch overhead.

**Fused gated depthwise conv works at single-dispatch cost.**
`y = conv(x,w1) (*) sigmoid(conv(g,w2))` (2 inputs, 2 weights, 5 ops)
compiles and runs in one dispatch: p50 0.093 ms @C=64 / 0.120 ms @C=10240 -
i.e. five ops for the price of one mailbox round trip. RelErr vs fp32
reference 5.2e-4 / 1.0e-3 (fp16 sigmoid noise; bar 3e-3).

**Input binding is REVERSED**: `_ANERequest` initWithInputs array positions
map to MIL function parameters right-to-left (proven via hypothesis testing:
with @[a,b] the hardware computed conv(param1<-b), conv(param2<-a); RelErr
1.5 vs 8.6e-4 discriminated cleanly). `ane_request_create_2in` reverses on
the caller's behalf so surfaces are passed in MIL parameter order.

**Sub-fp16 weight encodings: NEGATIVE RESULT (exhaustive).** The MIL text
compiler rejects every probed compressed-weight form with `InvalidMILProgram`:
`tensor<int4/uint4/int3/int2,...>` + `cast`, `constexpr_affine_to_dense`,
`dequantize`, and `constexpr_lut_to_dense` (both uint8 and int32 index
tensors). Verdicts are grammar-level compile rejections, not packing
ambiguities - probe weights used uniform bitstreams (all-0xFF bytes) whose
decode is packing-order invariant. **fp16 BLOBFILE constants are the only
weight encoding accepted through the text-MIL pipeline.** The section 3
"INT4 packed" envelope comment remains UNVERIFIED from this path.

Implications for low-bit weights (2/3/4-bit, EXL3/QTIP trellis):
- Not expressible in text MIL at all. EXL3 trellis codebooks additionally are
  not representable as ANY per-weight dequant op - they need a real dequant
  pass (CPU/Metal) before ANE consumption.
- ANE hardware itself does execute palettized/compressed weights, but that
  support is reached through the coremltools protobuf path (offline compile),
  not text MIL. Viable hybrid: compress offline with coremltools, ship the
  .mlmodelc, load at runtime via `ane_model_load_compiled` (already in the
  bridge) - keeps inference Python-free.

---

## 7. Weight SRAM / Packed-Weight Execution Investigation (Aug 2026)

Question: can packed 2/3/4-bit weights be fetched from DRAM and dequantized
into the ANE's weight SRAM, running MACs at fp16? (This is what ANE does for
palettized CoreML models - internal accumulation is fp16 regardless.)

**Container format DECODED** (from coremltools-compiled `.mlmodelc`, verified
byte-level). The weight blob is a multi-region envelope:

```
[0] u32   blob_count          (p2 file: 1, p4 file: 2)
[4] u32   version (= 2)
per region, back-to-back:
  u32     0xDEADBEEF
  u32     code               (1 = dense fp16, 3 = packed indices observed)
  u64     payload_size
  u64     payload_offset (absolute, from file start)
  payload...
```

MIL `BLOBFILE(offset=N)` therefore addresses each region's DEADBEEF
sub-header - reconciling the previously mysterious constants (indices @64,
palette @256 in the same file). Our single-blob builder (`ane_make_blob`)
writes count=1/code=1 with payload@128, which is why offset=64 always worked.

Apple's own backend emits **text MIL** with
`constexpr_lut_to_dense()[indices=..., lut=..., shape=...]` referencing those
regions - i.e. the op exists and hardware executes it via this container.
k-means palettization shrinks a 512 B conv weight to 352 B (4-bit) / 192 B
(2-bit) including palettes.

**But execution through the native in-memory text-MIL pipeline is BLOCKED:**
- All synthetic LUT/intN forms: rejected (`InvalidMILProgram`) or graceful
  compile failure, including a byte-faithful reconstruction of Apple's own
  two-blob container.
- Byte-exact Apple-authored artifact (verbatim model.mil + its weight.bin):
  ANECCompile rejects it too; the subsequent evaluation of the stale model
  handle crashes upstream in `-[_ANEClient reportEvaluateFailure]`
  (`-[_ANEInMemoryModel getUUID]: unrecognized selector`). Hardened the
  bridge dispatch with @try so upstream exceptions return NO cleanly.

**Conclusion**: packed-weight fetch/dequant-in-SRAM is real silicon behavior
but is only reachable through the full offline compile path (coremltools ->
espresso -> ANE-exported cache bundle `__.bin`/`__s.bin`, loadable natively
via `_ANEModel initWithModelAtURL:`). Text-MIL synthesis remains an fp16-only
channel. EXL3/QTIP trellis additionally requires external dequantization
under all routes.

---

## 8. 1-Bit Palettization on ANE - Execution vs Coherence (Aug 2026)

Sweep on an ANE-resident 5-layer 64ch 128x128 residual conv stack
(compute-plan verified `preferred_compute_device = MLNeuralEngineComputeDevice`;
k-means palettization, iOS18 target). Two distinct error axes:

| variant | quant_err (vs fp16-weight model) | exec_err (ANE vs CPU, same weights) |
|---|---|---|
| fp16 baseline   | -      | 0.0119 (cross-device noise floor) |
| 4-bit palette   | 0.150  | **0.0111** (= noise floor) |
| 2-bit palette   | 0.661  | **0.0130** (= noise floor) |
| 1-bit palette   | 0.752  | **0.0114** (= noise floor) |

Findings:
- **Execution**: ANE runs 1-bit palettized weights with error identical to the
  fp16 baseline's own cross-device variance - the dequant-in-weight-SRAM path
  is precision-neutral down to 1 bit. Hardware answer: yes.
- **Coherence**: post-hoc 1-bit destroys fidelity - 75% relative output error
  after only 5 layers (15% at 4-bit). Untrained-network k-means at 1 bit does
  NOT give coherent outputs; LLM decode would be especially fragile since
  near-tie argmax decisions cascade (cf. P12 norm-convention fork).
  Coherent 1-bit requires quantization-aware training (BitNet/BNN style),
  not post-hoc palettization.
- Practical floor for post-hoc compression: 4-bit (with 2-bit viable only for
  insensitive stages).
- Caveats found en route: per-channel-scale palettization (iOS18) emits
  `constexpr_blockwise_shift_scale` which crashes Apple's MPSGraph verifier
  on this OS; CoreML silently prefers CPU for small models - ALWAYS verify
  ANE residency via MLComputePlan before quoting ANE numbers.
