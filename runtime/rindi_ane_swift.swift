// rindi_ane_swift.swift — C-ABI bridge into Apple's official CoreAI runtime.
//
// Loads .aimodel bundles (coreai-torch export + coreai-build AOT) and runs
// them on CPU/GPU/ANE via SpecializationOptions. This replaces our broken
// hand-written MIL private-API path on macOS 27: official toolchain bundles
// verify fine, ours do not.
//
// Contract (C ABI):
//   void* rindi_ane_load(const char* bundlePath, int preferANE);
//       -> opaque handle or NULL (details via rindi_ane_last_error)
//   long rindi_ane_run(void* h, const uint16_t* xin, long inCount,
//                      uint16_t* outs, long outCap);
//       -> total output elements written (>0), or negative error code
//   void  rindi_ane_free(void* h);
//   const char* rindi_ane_last_error(void);
//
// Input/output are fp16 bit patterns (uint16). Outputs are concatenated in
// function output-name order.

import CoreAI
import Foundation

final class RindiBox {
    let fn: InferenceFunction
    let inputName: String
    let outputNames: [String]
    init(fn: InferenceFunction, inputName: String, outputNames: [String]) {
        self.fn = fn
        self.inputName = inputName
        self.outputNames = outputNames
    }
}

enum RindiErr {
    static var last: String = ""
    static let lock = NSLock()
    static func set(_ s: String) {
        lock.lock(); last = s; lock.unlock()
    }
}

@_cdecl("rindi_ane_last_error")
public func rindi_ane_last_error() -> UnsafePointer<CChar>? {
    // Leak one buffer per call site is acceptable for a probe; keep a static.
    struct Holder { static var p: UnsafeMutablePointer<CChar>? = nil }
    let msg = { RindiErr.lock.lock(); defer { RindiErr.lock.unlock() }; return RindiErr.last }()
    if let p = Holder.p { p.deallocate() }
    let p = strdup(msg.isEmpty ? "(no error)" : msg)!
    Holder.p = p
    return UnsafePointer(p)
}

final class RindiSlot: @unchecked Sendable {
    var result: Result<Any, Error>? = nil
    init() {}
}

private func runSync<T>(_ body: @escaping @Sendable () async throws -> T) throws -> T {
    let sem = DispatchSemaphore(value: 0)
    let slot = RindiSlot()
    Task {
        do {
            let res = try await body()
            slot.result = .success(res)
        } catch {
            slot.result = .failure(error)
        }
        sem.signal()
    }
    sem.wait()
    guard let r = slot.result else {
        throw NSError(domain: "rindi", code: 99, userInfo: [NSLocalizedDescriptionKey: "Task failed to set result"])
    }
    switch r {
    case .success(let v): return v as! T
    case .failure(let e): throw e
    }
}

@_cdecl("rindi_ane_load")
public func rindi_ane_load(_ path: UnsafePointer<CChar>, _ preferANE: Int32) -> UnsafeMutableRawPointer? {
    let pathString = String(cString: path)
    let url = URL(fileURLWithPath: pathString)
    return autoreleasepool {
        do {
            return try runSync {
                let kind: ComputeUnitKind = preferANE != 0 ? .neuralEngine : .cpu
                let opts = SpecializationOptions(preferredComputeUnitKind: kind)
                let model = try await AIModel.specialize(contentsOf: url,
                    options: opts, cache: .default, cachePolicy: .persistent)
                guard let f = try await model.loadFunction(named: "main") else {
                    throw NSError(domain: "rindi", code: 1,
                        userInfo: [NSLocalizedDescriptionKey:
                            "function 'main' not found; have \(model.functionNames)"])
                }
                let desc = f.descriptor
                guard let inName = desc.inputNames.first else {
                    throw NSError(domain: "rindi", code: 2,
                        userInfo: [NSLocalizedDescriptionKey: "function has no inputs"])
                }
                let box = RindiBox(fn: f, inputName: inName,
                                   outputNames: Array(desc.outputNames))
                return Unmanaged.passRetained(box).toOpaque()
            }
        } catch {
            RindiErr.set("load: \(error)")
            return nil
        }
    }
}

@_cdecl("rindi_ane_free")
public func rindi_ane_free(_ h: UnsafeMutableRawPointer) {
    Unmanaged<RindiBox>.fromOpaque(h).release()
}

@_cdecl("rindi_ane_run")
public func rindi_ane_run(_ h: UnsafeMutableRawPointer,
                          _ xin: UnsafePointer<UInt16>, _ rows: Int, _ cols: Int,
                          _ outs: UnsafeMutablePointer<UInt16>?, _ outCap: Int) -> Int {
    let box = Unmanaged<RindiBox>.fromOpaque(h).takeUnretainedValue()
    let inCount = rows * cols
    // Copy input into an NDArray up front (caller buffer may be freed after).
    var ndIn = NDArray(shape: [rows, cols], scalarType: .float16,
                       strides: [cols, 1])
        do {
            try ndIn.mutableRawView().withUnsafeMutableBytes { dst, shapeSpan, strideSpan in
                memcpy(dst, UnsafeRawPointer(xin), inCount * 2)
            }
        } catch let e {
            RindiErr.set("input view: \(e)")
            return -1
        }
    do {
        let total: Int = try runSync {
            var ndCopy = ndIn
            var outputs = try await box.fn.run(inputs: [box.inputName: ndCopy])
            _ = ndCopy // keep alive
            var written = 0
            for nm in box.outputNames {
                guard var val = outputs.remove(nm) else { continue }
                guard let arr = val.ndArray else {
                    throw NSError(domain: "rindi", code: 3,
                        userInfo: [NSLocalizedDescriptionKey:
                            "output \(nm) is not an NDArray"])
                }
                var a = arr
                let n = a.shape.reduce(1, *)
                let bytes = n * MemoryLayout<UInt16>.size
                if let outBuf = outs {
                    if written + n > outCap {
                        throw NSError(domain: "rindi", code: 4,
                            userInfo: [NSLocalizedDescriptionKey:
                                "outCap \(outCap) < needed \(written + n)"])
                    }
                    try a.mutableRawView().withUnsafeMutableBytes { ptr, _, _ in
                        if getenv("RINDI_ANE_DEBUG") != nil {
                            let p = ptr.assumingMemoryBound(to: UInt16.self)
                            var s = 0.0
                            for i in 0..<n { s += Double(Float16(bitPattern: p[i])) }
                            let vals = (0..<4).map { String(Float16(bitPattern: p[$0])) }.joined(separator: ",")
                            FileHandle.standardError.write(Data("[shim] \(nm): n=\(n) first4=\(vals) sum=\(s)\n".utf8))
                        }
                        memcpy(outBuf + written, ptr, bytes)
                    }
                }
                written += n
            }
            return written
        }
        return total
    } catch {
        RindiErr.set("run: \(error)")
        return -2
    }
}
