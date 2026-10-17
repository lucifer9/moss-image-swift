// Declared memory footprints (efficiency contract 1.14.0 split: resident + peak activation).
// MEASURED with `MossImageGate membench` (M5 Max, mlx-swift 0.32.3, Release, 2026-10-17), the
// package's own memory path: DiT (bf16 / quantized on load) + fp32 VAE resident; per request the
// bf16 Qwen3.5-9B encoder loads, encodes positive + empty negative inside a scope, is released,
// then denoise (both CFG forwards) and decode.
//
//   tier   resident   peak 1024²   peak 1456² (cap)   denoise+decode 1024² / 1456²
//   bf16   10.28 GB   26.19 GB     26.19 GB           17.40 / 24.66 GB
//   int8    5.61 GB   21.52 GB     21.52 GB           12.73 / 19.99 GB
//   int4    3.11 GB   19.03 GB     19.03 GB           10.24 / 17.50 GB
//
// The peak is the ENCODE phase on every tier and size inside the envelope: resident + the 16.8 GB
// bf16 encoder (activation 15.92 GB, size-independent up to the 1456² cap, where denoise + decode
// still sits 1.5 GB under it). Declared activation 16.5 GB (+3.6%); resident rounded up.
//
// int4 is measured but NOT declared: per-pass cos 0.94–0.96 vs the fp32 oracle (gate ≥ 0.99), and a
// 1024² / 50-step render diverges from bf16 at the same seed (18.6 dB PSNR — a coherent, text-correct,
// different image; int8 is 27.3 dB, the same image). It saves 2.5 GB of peak (the bf16 encoder
// dominates) and is not faster (2.33 vs 2.27 s/step). Revisit with an int8 encoder (AB-T-0307 notes).
// [MLX-pool numbers, not in-app phys_footprint — the in-app re-baseline (AB-T-0019 pattern) is owed.]
//
// int8 ENCODER lane (`textEncoderQuant: .int8`, either DiT tier): the encoder's Linears quantized on
// load, 9.39 GB instead of 16.8 GB (embed_tokens, the norms, conv1d, A_log, dt_bias and the GDN
// in_proj_a/b stay bf16/fp32). Quantized layer by layer from the lazy loads, so the bf16 copy is never
// resident (TE gate process peak 10.0 GB). Gate: relL2 2.1–3.3e-2 / cos ≥ 0.99944 vs the fp32 oracle
// (torch bf16 floor 1.2–2.1e-2); the 1024² fridge render is 27.13 dB vs bf16/bf16 (int8 DiT alone
// 27.34), note text exact.
//
//   lane          resident   peak 1024² (encode / denoise+decode)   peak 1456² (cap)
//   bf16 DiT      10.28 GB   20.09 GB (20.09 / 17.40)               24.66 GB (denoise)
//   int8 DiT       5.61 GB   15.42 GB (15.42 / 12.73)               19.99 GB (denoise)
//
// With the int8 encoder the encode phase stops being the peak at the cap: the 1456² denoise is
// (activation 14.39 GB on both DiT tiers). Declared activation 14.9 GB (+3.6%) as a
// `peakActivationBytesHint`, so the lane saves 1.6 GB of declared activation — the 6.1 GB it saves
// at 1024² only shows in admission with a lower envelope cap.

import MLXToolKit

enum MossImageFootprints {
    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 10_500_000_000, peakActivationBytes: 16_500_000_000),
        QuantFootprint(quant: .int8, residentBytes: 5_800_000_000, peakActivationBytes: 16_500_000_000),
    ]

    /// The int8-encoder lane's activation (FootprintConfigured hint): the 1456² denoise, 14.39 GB.
    static let int8EncoderPeakActivationBytes: UInt64 = 14_900_000_000
}
