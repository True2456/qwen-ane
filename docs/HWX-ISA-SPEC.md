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
