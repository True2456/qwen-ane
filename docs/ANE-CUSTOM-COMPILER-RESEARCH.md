# Research & Architecture: Building a Custom ANE Compiler / ISA Assembler

This document details the reverse-engineered internals of `AppleNeuralEngine.framework`, the `.hwx` binary container structure on Apple Silicon (M5 Max), and the exact technical roadmap required to emit raw ANE bytecode directly to hardware, bypassing `ANECCompile`.

---

## 1. The Execution Stack & Injection Points

```
┌───────────────────────────────────────────────────────────────────┐
│                    Layer 4: High-Level Client                     │
│               pure_ane.py / q38_ane_engine.py                     │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│              Layer 3: Objective-C Private Framework               │
│               _ANEInMemoryModel / _ANERequest                     │
│  - Descriptor: modelWithMILText:weights:optionsPlist:             │
│  - Methods: compileWithQoS:, loadWithQoS:, evaluateWithQoS:      │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                    [ BYPASS TARGET: ANECCompile ]
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│            Layer 2: ANE Client Daemon & Model Package             │
│                  _ANEClient / _ANEModel                           │
│  - Compiled Package: model.hwx + weight payloads                  │
│  - Direct Load: [_ANEClient loadModel:options:qos:error:]         │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
┌─────────────────────────────────▼─────────────────────────────────┐
│           Layer 1: Kernel Driver & Hardware Controller            │
│         AppleNeuralEngine.kext / _ANEDeviceController             │
│  - Direct Program Handle: programHandle (uint64_t)                │
│  - Direct DMA Mailbox to ANE Core SRAM                            │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. Objective-C Introspection: Bypassing `ANECCompile`

From runtime introspection of `AppleNeuralEngine.framework` on macOS (M5 Max):

### Key Classes:
* **`_ANEInMemoryModel`**: Orchestrates high-level MIL text $\to$ `.hwx` compilation.
* **`_ANEModel`**: Represents a compiled ANE program package stored on disk at `modelURL` / `localModelPath`.
* **`_ANEClient`**: Direct interface to `aned` (the ANE daemon). Implements:
  ```objc
  - (BOOL)loadModel:(_ANEModel *)model options:(NSDictionary *)opts qos:(unsigned int)qos error:(NSError **)err;
  - (BOOL)evaluateWithModel:(_ANEModel *)model options:(NSDictionary *)opts request:(_ANERequest *)req qos:(unsigned int)qos error:(NSError **)err;
  - (BOOL)doEvaluateDirectWithModel:(_ANEModel *)model options:(NSDictionary *)opts request:(_ANERequest *)req qos:(unsigned int)qos error:(NSError **)err;
  ```
* **`_ANEDeviceController`**: Low-level kernel driver bridge wrapping `ANEDeviceStruct` and assigning the 64-bit hardware `programHandle`.

### The Direct Binary Injection Path:
Instead of passing MIL text through `_ANEInMemoryModel`, a custom compiler can:
1. Synthesize the compiled `.hwx` binary container directly.
2. Instantiate an `_ANEModel` pointing to that `.hwx` container using:
   ```objc
   [_ANEModel alloc] initWithModelAtURL:customModelURL key:key identifierSource:1 cacheURLIdentifier:cacheId modelAttributes:attrs standardizeURL:YES];
   ```
3. Call `[_ANEClient loadModel:options:qos:error:]` directly.

---

## 3. Dissecting the ANE Binary Container (`model.hwx`)

The compiled ANE artifact (`model.hwx`) consists of three primary segments:

### A. The Container Header
* **Magic / Chip Tag:** 4-byte identifier (`HWX0` / `ANEF`) and target processor signature (`H17P` for M5, `H16P` for M4, `H13P` for M1).
* **Section Table:** Offsets and sizes for the Task Descriptors, Register Map, and Weight Table.

### B. Task Descriptors & Sequence Engine
* **Instruction Sequencer:** Controls the order of operations between SRAM input buffers, kernel execution tiles, and output DMA engines.
* **Tiling Parameters:**
  * Spatial width tile count ($S$ dimension, in multiples of 32/64).
  * Channel chunk strides ($C$ dimension, in multiples of 16/32).
  * Input/Output base address registers in unified memory (`IOSurfaceRef`).

### C. Execution Core Registers (Tile Control)
* **ALU Mode Register:**
  * `0x00`: FP16 Mode (512 MACs/cycle/core).
  * `0x01`: Dual-Lane INT8 Mode (1024 MACs/cycle/core $\to$ **42 TOPS**).
  * `0x02`: Dynamic FP16 Shift/Scale Mode.
* **Accumulator Bitdepth:** Configures whether partial sums accumulate into FP32 registers or INT32 registers.

---

## 4. Implementation Steps for a Custom ANE Assembler

To build a standalone ANE assembler (`ane_as`):

1. **Model Snapshot & Diffing:**
   * Compile identical small operations ($1\times 1$ convs, adds, reshapes) across varying shapes and precisions using `saveModelFiles`.
   * Hex-diff the resulting `model.hwx` binaries to map out the exact bitfields for ALU Mode, Channel/Spatial Strides, and Register addresses.
2. **Bytecode Builder:**
   * Construct a lightweight Python/C bytecode generator that emits valid `model.hwx` packages.
3. **Direct Driver Submission:**
   * Load the generated `model.hwx` through `_ANEClient` and execute via `_ANERequest`.
