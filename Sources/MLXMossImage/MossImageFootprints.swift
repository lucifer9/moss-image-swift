// Declared memory footprints (efficiency contract 1.14.0 split: resident + peak activation).
//
// 0.3.0 (row-band VAE decode, flux2-vae-mlx-swift ≥ 0.6.0, AB-R-0577): the decode stops being the cap peak, so the
// ENCODE is the peak on every lane at every size, and the encoder tier sets the activation again. In-app phys_footprint
// (Pixel Agency Image Engine, Release, against this tree; WIP/moss-image-inapp/run-lanes.sh, TAGP v030):
//
//   lane (in-app phys)      resident   peak 1024²          peak 1456² (cap)    declared 0.3.0
//   bf16 DiT + bf16 TE      10.44 GB   26.45 GB (encode)   26.45 GB (encode)   10.5 + 16.7 = 27.2 GB
//   bf16 DiT + int8 TE      10.44 GB   20.39 GB (encode)   20.39 GB (encode)   10.5 + 10.4 = 20.9 GB
//   int8 DiT + bf16 TE       5.77 GB   21.80 GB (encode)   21.80 GB (encode)    5.8 + 16.7 = 22.5 GB
//   int8 DiT + int8 TE       5.77 GB   15.73 GB (encode)   15.73 GB (encode)    5.8 + 10.4 = 16.2 GB
//
// - bf16 encoder: 16.03 GB over resident (int8 DiT; 16.01 on bf16) + 3.6 % = 16.61 → 16.7 GB, the quant-keyed
//   `peakActivationBytes`.
// - int8 encoder: 9.96 GB over resident (9.95 on bf16) + 3.6 % = 10.32 → 10.4 GB, the `peakActivationBytesHint`.
// - The 1456² decode no longer moves the ledger peak on any lane; its 1 s samples read 14.4–14.8 GB (bf16 DiT) and
//   8.6–9.0 GB (int8 DiT), against 17.8 GB untiled. MLX pool (membench): the 1456² decode is 2.86 GB over resident,
//   down from 14.38 GB.
// - A 24 GB Mac's ~19 GB budget now admits int8/int8 (16.2 GB declared; it was 23.3).
//
// Superseded history follows.
//
// 0.2.1 (AB-T-0317, in-app AB-R-0575): declared activation 17.5 GB on EVERY lane, from the Pixel Agency Image
// Engine's phys_footprint, not the MLX pool. At the 1456² cap the per-request peak is the VAE decode, 16.84 GB over
// resident on both DiT tiers and with either encoder (the decode doesn't see the encoder): 27.28 GB phys on the bf16
// DiT, 22.61 GB on int8. 16.84 GB + 3.6 % = 17.45 → 17.5 GB. Below the cap the bf16-encoder encode stays the peak
// (in-app 26.45–26.53 GB bf16, 21.82–21.87 GB int8 at 1024², AB-R-0570), still under the declaration. The 0.2.1
// in-app re-run (bf16/bf16 + int8/int8, AB-T-0317) put bf16/bf16 at 27.28 GB at the cap, as with the int8 encoder.
// The int8 encoder lowers the peak only below the cap (6.1 GB at 1024²), so it no longer declares its own activation: the
// 0.2.0 `peakActivationBytesHint` (14.9 GB, the pool's 1456² denoise) was 1.9 GB under the real cap peak.
//
//   lane (in-app phys)      resident   peak 1024²          peak 1456² (cap)    declared 0.2.1
//   bf16 DiT + bf16 TE      10.43 GB   26.45 GB (encode)   27.28 GB (decode)   10.5 + 17.5 = 28.0 GB
//   bf16 DiT + int8 TE      10.44 GB   20.38 GB (encode)   27.28 GB (decode)   10.5 + 17.5 = 28.0 GB
//   int8 DiT + bf16 TE       5.6 GB    21.87 GB (encode)   ≈22.6 GB (decode)    5.8 + 17.5 = 23.3 GB  (cap not run)
//   int8 DiT + int8 TE       5.77 GB   15.73 GB (encode)   22.62 GB (decode)    5.8 + 17.5 = 23.3 GB
//
// The `membench` MLX-pool figure at the cap (24.66 GB on bf16) understates phys there by 2.6 GB; the 0.2.0 notes
// below are kept as the pool record.
//
// 0.2.0 pool record. MEASURED with `MossImageGate membench` (M5 Max, mlx-swift 0.32.3, Release, 2026-10-17), the
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
// at 1024² only shows in admission with a lower envelope cap. [Superseded in 0.2.1: the in-app cap peak is the
// decode, 16.84 GB phys over resident, so the hint was 1.9 GB short.]

import MLXToolKit

enum MossImageFootprints {
    /// The bf16 encoder's activation on either DiT tier.
    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 10_500_000_000, peakActivationBytes: 16_700_000_000),
        QuantFootprint(quant: .int8, residentBytes: 5_800_000_000, peakActivationBytes: 16_700_000_000),
    ]
    /// The int8 encoder's activation on either DiT tier (`MossImageConfiguration.peakActivationBytesHint`).
    static let int8EncoderPeakActivationBytes: UInt64 = 10_400_000_000
}
