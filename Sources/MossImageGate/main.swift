// MossImageGate — parity gates against the oracle goldens (mlxengine-image/WIP/moss-image-oracle/
// goldens, fp32 CPU torch) and real generations. Every Metal-context gate is a CLI mode (the SPM
// test product's metallib is the fragile context).
//
//   MossImageGate sched | tok | vae [--cpu] | te [--dtype f32|bf16] [--quant-te int8] [--cpu]
//                 dit [--variant SFT|Base] [--dtype f32|bf16] [--cpu]
//                 e2e [--variant SFT|Base] [--dtype f32|bf16] [--cpu]
//                 generate --prompt "…" [--variant SFT|Base] [--size 1024x1024] [--steps 50]
//                          [--cfg 3] [--seed 42] [--no-skip] [--quant int8] [--quant-te int8] [--out file.png]
//                 membench [--variant SFT|Base] [--quant int8] [--quant-te int8] [--sizes 1024x1024,…]
// Paths: --weights <dir with MOSS-Image-SFT/ and MOSS-Image-Base/>, --goldens <oracle goldens dir>.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MossImage
import UniformTypeIdentifiers

// MARK: - args

nonisolated(unsafe) var args = Array(CommandLine.arguments.dropFirst())
let mode = args.isEmpty ? "help" : args.removeFirst()

// fp32 parity runs with TF32 OFF (set before the first MLX op). M5's TF32 fp32 matmuls cost
// ~1e-3 relL2 per forward (DiT 4e-4…2.7e-3, TE 2e-3…5e-3) against ≤ 5e-6 without; an explicit
// MLX_ENABLE_TF32 in the environment still wins. `generate` keeps the production default.
if mode != "generate", (args.firstIndex(of: "--dtype").map { args[$0 + 1] } ?? "f32") == "f32" {
    setenv("MLX_ENABLE_TF32", "0", 0)
}
func opt(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func flag(_ name: String) -> Bool { args.contains(name) }

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent()
let weightsRoot = URL(fileURLWithPath: opt("--weights")
    ?? packageRoot.appendingPathComponent("../../weights").standardized.path)
let goldensRoot = URL(fileURLWithPath: opt("--goldens")
    ?? packageRoot.appendingPathComponent("../moss-image-oracle/goldens").standardized.path)
let variant = opt("--variant") ?? "SFT"
let dtype: DType = (opt("--dtype") ?? "f32") == "bf16" ? .bfloat16 : .float32
let quantBits: Int? = opt("--quant").flatMap { $0 == "int8" ? 8 : $0 == "int4" ? 4 : nil }
let teQuant = opt("--quant-te").flatMap(MossImageTextEncoderQuant.init(rawValue:)) ?? .bf16
let sftSnapshot = weightsRoot.appendingPathComponent("MOSS-Image-SFT")
let variantSnapshot = weightsRoot.appendingPathComponent("MOSS-Image-\(variant)")

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func golden(_ group: String, _ name: String) throws -> [String: MLXArray] {
    try MLX.loadArrays(url: goldensRoot.appendingPathComponent(group).appendingPathComponent("\(name).safetensors"))
}

struct Stats: CustomStringConvertible {
    let relL2: Float, cos: Float, maxAbs: Float, meanAbs: Float
    var description: String {
        String(format: "relL2 %.3e  cos %.7f  maxAbs %.3e  meanAbs(ref) %.4f", relL2, cos, maxAbs, meanAbs)
    }
}

func compare(_ a: MLXArray, _ ref: MLXArray) -> Stats {
    let x = a.asType(.float32).flattened(), y = ref.asType(.float32).flattened()
    precondition(x.size == y.size, "shape \(a.shape) vs \(ref.shape)")
    let d = x - y
    let rel = (sqrt(sum(d * d)) / sqrt(sum(y * y))).item(Float.self)
    let cos = (sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))).item(Float.self)
    return Stats(relL2: rel, cos: cos, maxAbs: abs(d).max().item(Float.self), meanAbs: abs(y).mean().item(Float.self))
}

nonisolated(unsafe) var failures = 0
func gate(_ label: String, _ s: Stats, relL2 limit: Float) {
    let ok = s.relL2 <= limit
    if !ok { failures += 1 }
    print("\(ok ? "PASS" : "FAIL")  \(label): \(s)  (gate relL2 ≤ \(limit))")
}

func withDevice<T>(_ body: () throws -> T) rethrows -> T {
    flag("--cpu") ? try Device.withDefaultDevice(.cpu, body) : try body()
}

func writePNG(_ rgb: (pixels: [UInt8], width: Int, height: Int), to url: URL) throws {
    let data = Data(rgb.pixels) as CFData
    guard let provider = CGDataProvider(data: data),
          let image = CGImage(width: rgb.width, height: rgb.height, bitsPerComponent: 8, bitsPerPixel: 24,
                              bytesPerRow: rgb.width * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw MossImageError.invalidInput("PNG encode failed") }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw MossImageError.invalidInput("PNG write failed") }
}

let promptNames = ["cat", "empty", "note", "layout"]

// MARK: - modes

func runSched() throws {
    let g = try golden("shared", "sched")
    for steps in [50, 30, 20, 8, 4] {
        let s = MossImageScheduler()
        let sig = s.sigmas(steps: steps)
        let ts = s.timesteps(sigmas: sig)
        let refSig = g["s\(steps).sigmas"]!.asArray(Float.self)
        let refTs = g["s\(steps).timesteps"]!.asArray(Float.self)
        let exact = sig == refSig && ts == refTs
        if !exact { failures += 1 }
        print("\(exact ? "PASS" : "FAIL")  sched s\(steps): \(sig.count - 1) steps, bit-exact \(exact)"
            + (exact ? "" : "\n  swift \(sig)\n  ref   \(refSig)"))
    }
}

func runTok() async throws {
    let g = try golden("shared", "tok")
    let tokenizer = try await MossImagePromptEncoder.loadTokenizer(snapshot: sftSnapshot)
    for name in promptNames {
        let prompt = try promptText(name)
        let ids = tokenizer.encode(text: MossImagePromptEncoder.render(prompt), addSpecialTokens: false)
        let ref = g["\(name).ids"]!.asArray(Int32.self).map(Int.init)
        let ok = ids == ref
        if !ok { failures += 1 }
        print("\(ok ? "PASS" : "FAIL")  tok \(name): \(ids.count) ids" + (ok ? "" : "\n  swift \(ids)\n  ref   \(ref)"))
    }
}

func promptText(_ name: String) throws -> String {
    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: goldensRoot.appendingPathComponent("shared/tok.json"))) as! [String: [String: Any]]
    return meta[name]!["prompt"] as! String
}

func runVAE() throws {
    let g = try golden("shared", "vae")
    let vae = try MossImageWeights.loadVAE(directory: sftSnapshot.appendingPathComponent("vae"))
    for route in ["default", "exact"] {
        if route == "exact" { vae.convRoute = .conv3d }
        let img = withDevice { clip(vae.decodePackedLatents(g["packed"]!), min: -1, max: 1) }
        gate("vae decode (\(route) conv route\(flag("--cpu") ? ", cpu" : ""))", compare(img[0], g["image"]!),
             relL2: 1e-2)
    }
}

func runTE() throws {
    let g = try golden("shared", "te")
    let t0 = Date()
    let encoder = try MossImageWeights.loadTextEncoder(
        directory: sftSnapshot.appendingPathComponent("text_encoder"), dtype: dtype, quantBits: teQuant.bits)
    print(String(format: "      te load %.1fs, weights %.2f GB (MLX active)", Date().timeIntervalSince(t0),
                 Double(MLX.GPU.activeMemory) / 1e9))
    let tokens = try golden("shared", "tok")
    let tag = (dtype == .float32 ? "fp32" : "bf16") + (teQuant == .bf16 ? "" : " \(teQuant.rawValue)")
    for name in promptNames {
        let ids = tokens["\(name).ids"]!.reshaped(1, -1)
        let t0 = Date()
        let (h, captured) = withDevice { encoder.lastHiddenState(ids, captureLayers: [1, 4, 16]) }
        eval(h)
        let secs = Date().timeIntervalSince(t0)
        for li in [1, 4, 16] {
            print("      te \(name) layer\(li): \(compare(captured[li]![0], g["\(name).hidden\(li).fp32"]!))")
        }
        // int8: no torch int8 reference exists; the gate is cos ≥ 0.999 with relL2 ≤ 5e-2 (≈ 2.5× the
        // worst torch bf16 floor), and each prompt is printed against that floor.
        let limit: Float = teQuant == .int8 ? 5e-2 : dtype == .float32 ? 1e-3 : 3e-2
        let s = compare(h[0], g["\(name).fp32"]!)
        if teQuant == .int8, s.cos < 0.999 { failures += 1; print("FAIL  te \(name) cos \(s.cos) < 0.999") }
        gate("te \(name) last_hidden \(tag) (\(String(format: "%.2fs", secs)))", s, relL2: limit)
        if dtype == .bfloat16 || teQuant != .bf16 {
            print("      te \(name) torch-bf16 floor vs fp32: \(compare(g["\(name).bf16"]!, g["\(name).fp32"]!))")
        }
    }
}

/// Every layer of the encoder vs HF hidden_states (oracle dump_te_layers.py, 'cat' prompt).
func runTELayers() throws {
    let g = try golden("shared", "te_layers_cat")
    let encoder = try MossImageWeights.loadTextEncoder(
        directory: sftSnapshot.appendingPathComponent("text_encoder"), dtype: dtype)
    let ids = try golden("shared", "tok")["cat.ids"]!.reshaped(1, -1)
    let (h, captured) = withDevice { encoder.lastHiddenState(ids, captureLayers: Set(1 ... 31)) }
    let full = Set((0 ..< encoder.configuration.hiddenLayers).filter { encoder.configuration.isFullAttention($0) })
    for li in 1 ... 31 {
        print("layer \(li - 1) (\(full.contains(li - 1) ? "attn" : "gdn ")): \(compare(captured[li]![0], g["h\(li)"]!))")
    }
    print("last_hidden: \(compare(h[0], g["h32"]!))")
    for li in [1, 2, 3] {  // per-token relL2 where the drift first appears
        let a = captured[li]![0], r = g["h\(li)"]!
        let per = (0 ..< a.dim(0)).map { t -> String in
            let d = a[t] - r[t]
            return String(format: "%.1e", (sqrt(sum(d * d)) / sqrt(sum(r[t] * r[t]))).item(Float.self))
        }
        print("layer \(li - 1) per-token relL2: \(per.joined(separator: " "))")
        let rnorm = (0 ..< r.dim(0)).map { t in String(format: "%.1f", sqrt(sum(r[t] * r[t])).item(Float.self)) }
        print("layer \(li - 1) per-token |ref|: \(rnorm.joined(separator: " "))")
    }
}

func runDiT() throws {
    let g = try golden(variant, "dit")
    let te = try golden("shared", "te")
    let model = try MossImageWeights.loadTransformer(
        directory: variantSnapshot.appendingPathComponent("transformer"), dtype: dtype)
    if let quantBits { MossImageWeights.quantizeTransformer(model, bits: quantBits) }
    let latent = g["latent"]!
    let limit: Float = quantBits == 4 ? 0.15 : quantBits == 8 ? 0.05 : dtype == .float32 ? 1e-3 : 3e-2
    // RoPE table and modulation first (cheap, isolate the scalar paths).
    let rope = model.rope(height: latent.dim(2), width: latent.dim(3), textLength: te["cat.fp32"]!.dim(0))
    gate("rope table", compare(rope[0], g["rope.cat"]!), relL2: 1e-6)
    let (mod, head) = model.modulation(timestep: 500)
    gate("t500 modulation", compare(mod[0], g["t500.mod"]!), relL2: limit)
    gate("t500 head modulation", compare(head[0], g["t500.head"]!), relL2: limit)
    for prompt in ["cat", "empty"] {
        let ctx = te["\(prompt).fp32"]!.expandedDimensions(axis: 0)
        for t in [999.0, 500.0, 37.0] {
            for skip in [Set<Int>(), [22]] {
                let tag = "\(prompt).t\(Int(t))" + (skip.isEmpty ? "" : ".skip22")
                var captured: [String: MLXArray] = [:]
                let t0 = Date()
                let out = withDevice {
                    model(latent, timestep: t, context: ctx, skipLayers: skip) { name, v in
                        if prompt == "cat", t == 500, skip.isEmpty,
                           ["text_refined", "block0", "block21", "block41"].contains(name) { captured[name] = v }
                    }
                }
                eval(out)
                let secs = Date().timeIntervalSince(t0)
                for (name, v) in captured.sorted(by: { $0.key < $1.key }) {
                    print("      dit \(tag) \(name): \(compare(v[0], g["\(tag).\(name)"]!))")
                }
                gate("dit \(variant) \(tag) \(dtype == .float32 ? "fp32" : "bf16") (\(String(format: "%.2fs", secs)))",
                     compare(out[0], g["\(tag).out"]!), relL2: limit)
            }
        }
    }
}

func runE2E() throws {
    let name = "e2e_cat_8step"
    let g = try golden(variant, name)
    let te = try golden("shared", "te")
    let model = try MossImageWeights.loadTransformer(
        directory: variantSnapshot.appendingPathComponent("transformer"), dtype: dtype)
    let vae = try MossImageWeights.loadVAE(directory: sftSnapshot.appendingPathComponent("vae"))
    let pipe = MossImagePipeline(transformer: model, promptEncoder: nil, vae: vae)
    var p = MossImageGenerationParameters()
    p.steps = 8
    let noise = g["noise"]!
    p.height = noise.dim(2) * 16
    p.width = noise.dim(3) * 16
    let c = MossImageConditioning(positive: te["cat.fp32"]!.expandedDimensions(axis: 0).asType(dtype),
                                  negative: te["empty.fp32"]!.expandedDimensions(axis: 0).asType(dtype))
    // fp32: the trajectory amplifies per-forward error ~400× by step 6 (σ 0.791 → 0.671 → 0.463).
    // The oracle's own fp32 run with a random 5e-6 relative perturbation per forward drifts 7.2e-4
    // (oracle e2e_sensitivity.py); the port's ≤ 5e-6 systematic error lands at 2.0e-3 (SFT) /
    // 1.5e-4 (Base). Gate 5e-3 latents, 2e-3 image. torch fp32 on MPS lands at 2.3e-3 / 1.3e-3 —
    // the same drift from a different device. bf16: torch's own bf16 trajectory (MPS) drifts 0.283 /
    // 0.173 from the fp32 oracle and the port 0.209 / 0.131, so the bf16 gate is torch's own drift.
    let limit: Float = dtype == .float32 ? 5e-3 : 0.30
    let t0 = Date()
    let latents = try withDevice {
        try pipe.denoise(conditioning: c, parameters: p, latents: noise) { i, n, l in
            print("      e2e step \(i): \(compare(l[0], g["step\(i).latents"]!))")
        }
    }
    let image = withDevice { pipe.decode(latents) }
    gate("e2e \(variant) 8-step latents (\(String(format: "%.1fs", Date().timeIntervalSince(t0))))",
         compare(latents[0], g["step7.latents"]!), relL2: limit)
    gate("e2e \(variant) 8-step image", compare(image[0], g["image"]!), relL2: dtype == .float32 ? 2e-3 : 0.18)
    let out = packageRoot.appendingPathComponent("outputs/e2e_\(variant)_\(dtype == .float32 ? "fp32" : "bf16").png")
    try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    try writePNG(MossImagePipeline.toRGB8(image), to: out)
    print("      wrote \(out.path)")
}

func runGenerate() async throws {
    guard let prompt = opt("--prompt") else { throw MossImageError.invalidInput("--prompt required") }
    var p = MossImageGenerationParameters()
    if let size = opt("--size") {
        let wh = size.split(separator: "x").compactMap { Int($0) }
        p.width = wh[0]
        p.height = wh[1]
    }
    p.steps = Int(opt("--steps") ?? "50")!
    p.guidanceScale = Float(opt("--cfg") ?? "3")!
    p.seed = UInt64(opt("--seed") ?? "42")!
    if let neg = opt("--negative") { p.negativePrompt = neg }
    if flag("--no-skip") { p.uncondSkipLayers = [] }
    let modelDType: DType = (opt("--dtype") ?? "bf16") == "f32" ? .float32 : .bfloat16
    let t0 = Date()
    let pipe = try await MossImagePipeline.load(
        transformerSnapshot: variantSnapshot, sharedSnapshot: sftSnapshot, dtype: modelDType, textEncoderQuant: teQuant)
    if let quantBits { MossImageWeights.quantizeTransformer(pipe.transformer, bits: quantBits) }
    log("loaded in \(String(format: "%.1fs", Date().timeIntervalSince(t0)))")
    let t1 = Date()
    let c = try pipe.encodeAndRelease(prompt: prompt, parameters: p)
    log("encoded in \(String(format: "%.2fs", Date().timeIntervalSince(t1))) (\(c.positive.dim(1)) tokens)")
    let t2 = Date()
    let latents = try pipe.denoise(conditioning: c, parameters: p) { i, n, _ in
        log("step \(i + 1)/\(n)  \(String(format: "%.1fs", Date().timeIntervalSince(t2)))")
    }
    let denoiseSecs = Date().timeIntervalSince(t2)
    let t3 = Date()
    let image = pipe.decode(latents)
    let decodeSecs = Date().timeIntervalSince(t3)
    let out = URL(fileURLWithPath: opt("--out") ?? packageRoot.appendingPathComponent(
        "outputs/gen_\(variant)_\(p.width)x\(p.height)_s\(p.steps)_seed\(p.seed).png").path)
    try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
    try writePNG(MossImagePipeline.toRGB8(image), to: out)
    print(String(format: "generate %@ %dx%d %d steps: denoise %.1fs (%.2fs/step), decode %.2fs, peak %.2f GB -> %@",
                 variant, p.width, p.height, p.steps, denoiseSecs, denoiseSecs / Double(p.steps), decodeSecs,
                 Double(MLX.GPU.peakMemory) / 1e9, out.path))
}

/// The package's memory path at one size: resident = DiT (bf16 / --quant) + fp32 VAE after load;
/// then per request: load the bf16 encoder, encode positive + empty negative, release it, two
/// denoise steps (both CFG forwards), decode. Activation = peak − resident (MLX pool numbers).
func runMembench() async throws {
    let sizes = (opt("--sizes") ?? "1024x1024,1456x1456").split(separator: ",").map {
        $0.split(separator: "x").compactMap { Int($0) }
    }
    MLX.GPU.clearCache()
    let transformer = try MossImageWeights.loadTransformer(
        directory: variantSnapshot.appendingPathComponent("transformer"), dtype: .bfloat16)
    if let quantBits { MossImageWeights.quantizeTransformer(transformer, bits: quantBits) }
    let vae = try MossImageWeights.loadVAE(directory: sftSnapshot.appendingPathComponent("vae"))
    MLX.GPU.clearCache()
    let resident = Double(MLX.GPU.activeMemory) / 1e9
    print(String(format: "membench %@ %@ te-%@: resident %.2f GB", variant, opt("--quant") ?? "bf16",
                 teQuant.rawValue, resident))
    for wh in sizes {
        MLX.GPU.resetPeakMemory()
        var p = MossImageGenerationParameters()
        p.width = wh[0]; p.height = wh[1]; p.steps = 2
        let prompt = (try? promptText("layout")) ?? "A cat"
        // As the package: the encoder lives only inside this closure.
        let c = try await { () async throws -> MossImageConditioning in
            let t0 = Date()
            let encoder = try await MossImagePromptEncoder.load(
                snapshot: sftSnapshot, dtype: .bfloat16, textEncoderQuant: teQuant)
            print(String(format: "  encoder load %.1fs", Date().timeIntervalSince(t0)))
            return try MossImagePipeline(transformer: transformer, promptEncoder: encoder, vae: vae)
                .encode(prompt: prompt, parameters: p)
        }()
        MLX.GPU.clearCache()
        let encodePeak = Double(MLX.GPU.peakMemory) / 1e9
        MLX.GPU.resetPeakMemory()
        let pipe = MossImagePipeline(transformer: transformer, promptEncoder: nil, vae: vae)
        let t0 = Date()
        let latents = try pipe.denoise(conditioning: c, parameters: p)
        let stepSecs = Date().timeIntervalSince(t0) / 2
        _ = pipe.decode(latents)
        let peak = Double(MLX.GPU.peakMemory) / 1e9
        let overall = max(peak, encodePeak)
        print(String(format: "  %dx%d: peak %.2f GB (encode %.2f, denoise+decode %.2f), activation %.2f GB, %.2f s/step",
                     wh[0], wh[1], overall, encodePeak, peak, overall - resident, stepSecs))
        MLX.GPU.clearCache()
    }
}

switch mode {
case "membench": try await runMembench()
case "sched": try runSched()
case "tok": try await runTok()
case "vae": try runVAE()
case "te": try runTE()
case "te-layers": try runTELayers()
case "dit": try runDiT()
case "e2e": try runE2E()
case "generate": try await runGenerate()
default:
    print("modes: sched | tok | vae | te | dit | e2e | generate (see header)")
}
if failures > 0 {
    print("\(failures) gate(s) FAILED")
    exit(1)
}
