// Link against the production Rindi-NativeMLX build, not an mlx_lm port.
// Uses the exact SwitchGLU / weightedExpertSum called by Qwen4Exp.swift.
import Foundation
import MLX
import MLXNN
import MLXLMCommon

for path in CommandLine.arguments.dropFirst() {
    let tensors = try loadArrays(url: URL(fileURLWithPath: path))
    let model = SwitchGLU(inputDims: 2560, hiddenDims: 640, numExperts: 10)
    quantize(model: model, groupSize: 64, bits: 4)
    let weights = tensors.filter { $0.key.hasSuffix(".weight") || $0.key.hasSuffix(".scales") || $0.key.hasSuffix(".biases") }
        .mapValues { $0.dtype == .uint32 ? $0 : $0.asType(.float32) }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    let x = tensors["x"]!.asType(.float32).reshaped(1, 1, 2560)
    let indices = MLXArray(0..<10).reshaped(1, 1, 10)
    let scores = tensors["scores"]!.asType(.float32)
    var y = weightedExpertSum(model(x, indices), scores)
    let g = matmul(x, tensors["shared.gate"]!.asType(.float32))
    let u = matmul(x, tensors["shared.up"]!.asType(.float32))
    let shared = matmul(silu(g) * u, tensors["shared.down"]!.asType(.float32))
    let selector = sigmoid(matmul(x, tensors["shared.selector"]!.asType(.float32)))
    y = y + shared * selector
    let expected = tensors["expected_fp32"]!.asType(.float32)
    let error = sqrt(((y - expected) * (y - expected)).sum() / (expected * expected).sum())
    eval(error)
    let relative = error.item(Float.self)
    print("SWIFT_REFERENCE \(URL(fileURLWithPath: path).lastPathComponent) relative_L2=\(relative)")
    if !relative.isFinite || relative > 0.00001 {
        fatalError("production Swift SwitchGLU differs from FP32 runtime by more than 1e-5 relative L2")
    }
}
