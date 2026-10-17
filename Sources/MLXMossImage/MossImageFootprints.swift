// Declared memory footprints (efficiency contract 1.14.0 split: resident + peak activation).
// PROVISIONAL — weight-size arithmetic only, replaced by the MossImageGate membench measurement
// before this package is registered anywhere:
//   resident = DiT (bf16 10.2 GB · int8 ≈ 5.6 · int4 ≈ 3.2, quantized block Linears only) + VAE fp32 0.25 GB
//   activation = the per-request Qwen3.5-9B encoder (16.8 GB bf16, released before denoise) +
//                denoise/decode working set at the 1456² cap.

import MLXToolKit

enum MossImageFootprints {
    static let declared: [QuantFootprint] = [
        QuantFootprint(quant: .bf16, residentBytes: 10_500_000_000, peakActivationBytes: 24_000_000_000),
        QuantFootprint(quant: .int8, residentBytes: 6_000_000_000, peakActivationBytes: 24_000_000_000),
        QuantFootprint(quant: .int4, residentBytes: 3_600_000_000, peakActivationBytes: 24_000_000_000),
    ]
}
