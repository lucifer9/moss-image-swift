// swift-tools-version: 6.2
// moss-image-swift — Swift/MLX port of OpenMOSS-Team/MOSS-Image-Base and MOSS-Image-SFT (Apache-2.0;
// shipping approved AB-D-0150): a 5B single-stream DiT (42 blocks, one shared adaLN modulation,
// SiTU-GLU FFN, 4-axis RoPE) over a frozen Qwen3.5-9B text encoder (last hidden state, chat template,
// thinking enabled) and the FLUX.2 VAE with BFL's small decoder.
// Reference = the upstream `pipeline_moss_image.py` (diffusers custom pipeline, 2026-10-16);
// goldens from mlxengine-image/WIP/moss-image-oracle (fp32 CPU torch). Task AB-T-0307.

import PackageDescription

let package = Package(
    name: "MossImage",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "MossImage", targets: ["MossImage"]),
        // MLXEngine wrappers: MossImageSFTPackage (PackageID moss-image-sft, the default tier) and
        // MossImageBasePackage (moss-image-base) — textToImage, Apache-2.0, bf16/int8/int4 DiT tiers.
        .library(name: "MLXMossImage", targets: ["MLXMossImage"]),
        .executable(name: "MossImageGate", targets: ["MossImageGate"]),
    ],
    dependencies: [
        // 0.32.3 carries the NAX split-K GEMM fix (mlx#3810): the DiT FFN down-projection has K = 8192.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
        // MLXLMCommon.gatedDeltaUpdate (the GDN Metal kernel) for the Qwen3.5 encoder.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // ≥ 0.4.0 reads the decoder widths from vae/config.json — the FLUX.2 small decoder; ≥ 0.6.0 adds the
        // memory-bounded row-band decode (`decodePackedLatents(_:tile:)`, AB-R-0577).
        .package(url: "https://github.com/xocialize/flux2-vae-mlx-swift", from: "0.6.0"),
        // MLXEngine contract (MLXToolKit) + conformance gates, for the wrapper only.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.65.4"),
    ],
    targets: [
        .target(
            name: "MossImage",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Flux2VAE", package: "flux2-vae-mlx-swift"),
            ],
            path: "Sources/MossImage"
        ),
        .target(
            name: "MLXMossImage",
            dependencies: [
                "MossImage",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXMossImage"
        ),
        .executableTarget(
            name: "MossImageGate",
            dependencies: ["MossImage", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXRandom", package: "mlx-swift")],
            path: "Sources/MossImageGate"
        ),
        .testTarget(
            name: "MossImageTests",
            dependencies: ["MossImage", .product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXNN", package: "mlx-swift")],
            path: "Tests/MossImageTests"
        ),
        .testTarget(
            name: "MLXMossImageTests",
            dependencies: [
                "MLXMossImage",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXMossImageTests"
        ),
    ]
)
