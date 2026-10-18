// Init-time configuration (C9) for the MOSS-Image packages. One configuration type serves both
// checkpoints; the wrapping package (MossImageSFTPackage / MossImageBasePackage) fixes the variant.
//
// Two weight roles, because Base and SFT share every file except the DiT byte-for-byte (LFS sha256
// checked 2026-10-17): the variant's `transformer/` comes from its own repo, and the Qwen3.5-9B
// text encoder, tokenizer and FLUX.2 small-decoder VAE always come from the SFT repo, so a machine
// holding both tiers stores the 16.8 GB encoder once.
//
// Weights materialize from the mlx-community mirrors (fleet durability policy: a namespace we
// publish to, pinned by commit). Every weight / config / tokenizer file in them is byte-identical
// to the upstream OpenMOSS revision (LFS sha256 compared after upload, 2026-10-17).

import Foundation
import MLXToolKit
import MossImage

public enum MossImageVariant: String, Codable, Sendable, CaseIterable {
    case sft, base

    /// The mlx-community mirror the weights materialize from.
    public var repo: String {
        switch self {
        case .sft: return "mlx-community/MOSS-Image-SFT-bf16"
        case .base: return "mlx-community/MOSS-Image-Base-bf16"
        }
    }

    /// Mirror commit (pinned; the mirror is byte-identical to `upstreamRevision`).
    public var revision: String {
        switch self {
        case .sft: return "ea698e99312015505884559ebad94b4a5a0d0753"
        case .base: return "56e5c1f98640356efce285957df9a883a53064fa"
        }
    }

    public var upstreamRepo: String {
        switch self {
        case .sft: return "OpenMOSS-Team/MOSS-Image-SFT"
        case .base: return "OpenMOSS-Team/MOSS-Image-Base"
        }
    }

    /// Upstream commits the port was gated against (oracle goldens, AB-T-0307).
    public var upstreamRevision: String {
        switch self {
        case .sft: return "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"
        case .base: return "1ceede80badddd6a7812069619edaa5633587192"
        }
    }
}

public struct MossImageConfiguration: PackageConfiguration, ModelStorable, QuantConfigured, FootprintConfigured,
    WeightSourcing {
    public var variant: MossImageVariant
    /// DiT tier: `.bf16` or `.int8` (the block and text-refiner Linears quantized on load). int4
    /// is not declared — see MossImageFootprints.
    public var quant: Quant
    /// Explicit root holding this variant's `transformer/`. Empty = store.
    public var snapshotPath: String
    /// Explicit root holding `text_encoder/`, `tokenizer/`, `vae/` (an SFT snapshot). Empty = store.
    public var sharedSnapshotPath: String
    public var defaultSteps: Int
    public var defaultGuidanceScale: Double
    /// Keep the 16.8 GB bf16 Qwen3.5-9B encoder resident between requests (big-RAM tiers). Off: it
    /// loads per request and is released before the denoise loop.
    public var keepEncoderResident: Bool
    /// Qwen3.5-9B encoder tier, orthogonal to the DiT `quant`. `.int8` quantizes its Linears on load
    /// (9.4 GB instead of 16.8 GB). The encode is the peak on every lane up to the 1456² cap (0.3.0, row-band VAE
    /// decode), so it declares its own activation, `peakActivationBytesHint` (MossImageFootprints).
    public var textEncoderQuant: MossImageTextEncoderQuant
    public var modelsRootDirectory: URL?

    /// The text encoder, tokenizer and VAE source (shared by both variants).
    public static let sharedRepo = MossImageVariant.sft.repo

    public init(
        variant: MossImageVariant = .sft,
        quant: Quant = .bf16,
        snapshotPath: String = "",
        sharedSnapshotPath: String = "",
        defaultSteps: Int = 50,
        defaultGuidanceScale: Double = 3.0,
        keepEncoderResident: Bool = false,
        textEncoderQuant: MossImageTextEncoderQuant = .bf16,
        modelsRootDirectory: URL? = nil
    ) {
        self.variant = variant
        self.quant = quant
        self.snapshotPath = snapshotPath
        self.sharedSnapshotPath = sharedSnapshotPath
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.keepEncoderResident = keepEncoderResident
        self.textEncoderQuant = textEncoderQuant
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant, quant, snapshotPath, sharedSnapshotPath, defaultSteps, defaultGuidanceScale, keepEncoderResident,
             textEncoderQuant
    }

    /// `textEncoderQuant` is optional on decode, so configurations saved before it existed still load (bf16).
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        variant = try c.decode(MossImageVariant.self, forKey: .variant)
        quant = try c.decode(Quant.self, forKey: .quant)
        snapshotPath = try c.decode(String.self, forKey: .snapshotPath)
        sharedSnapshotPath = try c.decode(String.self, forKey: .sharedSnapshotPath)
        defaultSteps = try c.decode(Int.self, forKey: .defaultSteps)
        defaultGuidanceScale = try c.decode(Double.self, forKey: .defaultGuidanceScale)
        keepEncoderResident = try c.decode(Bool.self, forKey: .keepEncoderResident)
        textEncoderQuant = try c.decodeIfPresent(MossImageTextEncoderQuant.self, forKey: .textEncoderQuant) ?? .bf16
        modelsRootDirectory = nil
    }

    /// The DiT tier keys the resident bytes (QuantFootprint); the encoder tier moves the activation.
    public var residentBytesHint: UInt64? { nil }
    public var peakActivationBytesHint: UInt64? {
        textEncoderQuant == .int8 ? MossImageFootprints.int8EncoderPeakActivationBytes : nil
    }

    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "transformer", repo: variant.repo, revision: variant.revision,
                         matching: ["transformer/*", "model_index.json", "scheduler/*", "LICENSE"]),
            WeightSource(role: "shared", repo: Self.sharedRepo, revision: MossImageVariant.sft.revision,
                         matching: ["text_encoder/*", "tokenizer/*", "vae/*", "LICENSE"]),
        ]
    }

    static func has(_ path: String, _ marker: String) -> Bool {
        !path.isEmpty && FileManager.default.fileExists(
            atPath: URL(fileURLWithPath: path).appendingPathComponent(marker).path)
    }

    /// Explicit paths satisfy their own role; everything else resolves through the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        let store = ModelStore(root: storeRoot)
        return weightSources.filter { source in
            let marker = source.role == "transformer" ? "transformer" : "text_encoder"
            if source.role == "transformer", Self.has(snapshotPath, marker) { return false }
            if source.role == "shared", Self.has(sharedSnapshotPath, marker) && Self.has(sharedSnapshotPath, "vae") {
                return false
            }
            guard let dir = Self.storeDirectory(store, repo: source.repo, revision: source.revision, marker: marker)
            else { return true }
            return !fm.fileExists(atPath: dir.appendingPathComponent(marker).path)
        }
    }

    static func storeDirectory(_ store: ModelStore, repo: String, revision: String?, marker: String) -> URL? {
        let fm = FileManager.default
        if let flat = store.directory(for: repo), fm.fileExists(atPath: flat.appendingPathComponent(marker).path) {
            return flat
        }
        if let snap = store.snapshotDirectory(for: repo, revision: revision),
           fm.fileExists(atPath: snap.appendingPathComponent(marker).path) {
            return snap
        }
        return store.directory(for: repo)
    }

    /// Root holding this variant's `transformer/`.
    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        return Self.storeDirectory(ModelStore(root: storeRoot), repo: variant.repo, revision: variant.revision,
                                   marker: "transformer")
    }

    /// Root holding `text_encoder/`, `tokenizer/`, `vae/`.
    public func resolvedSharedDirectory(storeRoot: URL?) -> URL? {
        if !sharedSnapshotPath.isEmpty { return URL(fileURLWithPath: sharedSnapshotPath) }
        return Self.storeDirectory(ModelStore(root: storeRoot), repo: Self.sharedRepo,
                                   revision: MossImageVariant.sft.revision, marker: "text_encoder")
    }
}
