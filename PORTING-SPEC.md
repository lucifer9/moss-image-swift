# moss-image-swift — Porting spec: MOSS-Image-Base + MOSS-Image-SFT → Swift/MLX

Task AB-T-0307 · shipping decision AB-D-0150 · oracle `../moss-image-oracle` (fp32 CPU torch goldens)

## 1. Licence and data

- **Weights:** Apache-2.0 for both checkpoints (OpenMOSS-Team, released 2026-10-16).
- **Components:** the Qwen3.5-9B text encoder (Apache-2.0) and BFL's FLUX.2-small-decoder (Apache-2.0) ship
  inside both snapshots.
- **Corpus:** the tech report (Table 8) *enumerates* the training data: ~3.2B records, ~98% ReLAION /
  OmniCorpus / DataComp web scrape, with AI-generated images filtered out and no rights statement.
- **Decision:** the operator approved shipping on that account, **AB-D-0150**. This is an explicit exception to
  AB-L-0082 for these two checkpoints, not posture A (AB-D-0105 covers the undisclosed-corpus case).
- **Port code:** MIT.

## 2. Reference facts (upstream `pipeline_moss_image.py`, byte-identical in both repos)

Base and SFT are byte-identical except `transformer/*.safetensors` (LFS sha256 compared 2026-10-17). The text
encoder, tokenizer and VAE are therefore loaded from the SFT snapshot for both.

### 2.1 Transformer `MOSSImageTransformer` (463 tensors, 10.16 GB bf16, ≈5.1B)

- **Shape:** dim 3072, 42 blocks, 24 heads × 128, FFN 8192, latent_channels 128 (FLUX.2 32 ch × 2×2 patch,
  /16 grid), text_dim 4096.
- **Single stream:** the image tokens, row-major over the /16 grid, come first, then the refined text tokens.
  Each sample runs at its true text length: no padding and no mask.
- **Block:**
  - RMSNorm in fp32, cast back **before** the weight multiply.
  - ONE shared adaLN modulation, `shared_block_modulation` (dim → 6·dim), read by every block.
  - QK-RMSNorm per head.
  - Bias-free q/k/v/o and a fused `ffn_up_gate`.
  - Attention-out and FFN-out gated by gate_a / gate_f.
- **SiTU-GLU FFN:** `(4·tanh(g/4)·σ(g)) · (25·tanh(h/25))` (β = 4, 25; the report omits them).
- **Text refiner:** Linear 4096 → 3072, then 2 text-only blocks (biased qkv/proj/ffn, no modulation) on the
  text slice of the RoPE.
- **4-axis RoPE (r, f, h, w):** 16 bands per axis (head_dim / 8), base 1e4, fp32 phases.
  - Table layout is [sin(r,f,h,w) | cos(r,f,h,w)], applied as rotate-half over 64 + 64.
  - Image tokens: r = 0, f = text_length, h and w centred (`i − (H−1)/2`).
  - Text tokens: f = index.
  - `rope.bands` is in the checkpoint but is a derived constant; it is rebuilt, never loaded.
- **Timestep:** a float64 sinusoid (256 dims, cos | sin) → fp32 → `time_embedding` (Linear, SiLU, Linear). Its
  Sequential keys "0" / "2" are renamed to `linear_1` / `linear_2` at load, because `unflattened` reads digits
  as array indices.
- **Head:** `final_norm_video` on the image tokens · (1 + head[1]) + head[0], then `final_linear_video`.

### 2.2 Text encoder: Qwen3.5-9B `Qwen3_5Model`, `last_hidden_state`

- **Rendering:** the chat template with `add_generation_prompt=True, enable_thinking=True` gives
  `<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n`.
  - Swift renders this string directly; ids match the oracle on all four fixture prompts, including the empty
    negative.
- **Encoding:** one unpadded forward per prompt, ≤ 4096 tokens, feature after the final norm. The vision tower
  is unused.
- **Layers:** 24 Gated-DeltaNet layers plus 8 gated full-attention layers (3, 7, …, 31).
- **Full attention:** q_proj emits [query | gate] per head; partial RoPE over 64 of 256 dims, θ 1e7. Interleaved
  M-RoPE with identical t/h/w ids is plain RoPE.
- **Norms:**
  - Zero-centred RMSNorm `(1 + w)`, in fp32, for the input, post-attention, final, q and k norms.
  - The GDN gated norm uses `w` directly.
- **GDN:**
  - Causal depthwise conv (k 4) → silu.
  - q and k cast to **fp32**, then FLA's `l2norm`: `x · rsqrt(Σx² + 1e-6)`, with eps **on the sum**.
  - q scaled by Dk^-½; key heads `repeat_interleave`'d onto the 32 value heads; the rule runs in fp32.

### 2.3 Sampler

- **Steps:** Euler flow matching, default 50.
- **`MOSSImageScheduler` grid:**
  - `grid(n, shift) = shift·v / (1 + (shift−1)·v)` over `linspace(1, shift/(1000+shift−1), n)`.
  - The shift-5 inference grid is snapped to the nearest point of the shift-1 training grid of 1000, with
    consecutive duplicates removed.
  - Timesteps = σ·1000, with a terminal 0.
  - `torch.linspace` fp32 on CPU is a **fused multiply-add**. Plain mul-then-add differs at 239 of 1000 grid
    points; Swift uses `addingProduct` and is bit-exact.
- **CFG 3:** the negative (empty-prompt) branch runs with **block 22 skipped**, two forwards per step.
  - The combine and the Euler update are fp32, and the initial noise is fp32.
  - Optional positive-branch SLG is ported but off.
- **Decode:**
  - bn de-norm in packed 128-channel space, then unpatchify 2×2.
  - The FLUX.2 small decoder (widths [96, 192, 384, 384]) in fp32, then clamp.
  - `(x/2 + 0.5)·255`, rounded.

## 3. Package layout

| Target | Contents |
|---|---|
| `MossImage` | Core (`Adapted/Qwen35Encoder`, `MossImageTransformer`, `MossImageScheduler`, `MossImagePromptEncoder`, `MossImagePipeline`, `MossImageWeights`). Depends on mlx-swift 0.32.3, mlx-swift-lm 3.32.3 (`gatedDeltaUpdate` only), swift-transformers and flux2-vae-mlx-swift. |
| `MLXMossImage` | MLXEngine packages `MossImageSFTPackage` (`moss-image-sft`) and `MossImageBasePackage` (`moss-image-base`): T2I, bf16 / int8 DiT tiers (int4 measured, not declared), bf16 / int8 encoder (`textEncoderQuant`), envelope 256²–1456² by area. |
| `MossImageGate` | Parity gates (`sched`, `tok`, `vae`, `te`, `te-layers`, `dit`, `e2e`) and `generate` / `membench`. |

**flux2-vae-mlx-swift:** the decoder widths are now read from `vae/config.json` (`decoder_block_out_channels`).
- The full decoder stays the default, so Klein, Lens and ERNIE are unchanged.
- Tagged v0.4.0; this package depends on it by URL.

## 4. Gate results (M5 Max, mlx-swift 0.32.3)

fp32 parity runs with **TF32 off**: the gate sets `MLX_ENABLE_TF32=0` itself. M5's TF32 fp32 matmuls cost
≈1e-3 relL2 per forward; without TF32 the port is at ≤ 5e-6.

| Gate | Result |
|---|---|
| sched (4, 8, 20, 30, 50 steps) | bit-exact |
| tok (4 prompts incl. empty, CJK, JSON layout) | ids exact |
| VAE small decoder, fp32 | relL2 1.2e-6 (exact conv route) · 3.6e-6 (default route); TF32 on: 1.8e-4 · 3.5e-3 |
| TE fp32, all 32 layers ('cat') | layer 0: 7e-7 … last_hidden: 4.8e-6 |
| TE bf16 vs fp32 oracle | 1.2–2.1e-2, the same as torch's own bf16 floor (1.2–2.1e-2) |
| DiT SFT fp32, 12 cases (t 999 / 500 / 37 × cat / empty × skip-22) | relL2 1.1e-6 … 5.3e-6 |
| DiT Base fp32, same cases | passes with TF32 on, at the TF32 level (4e-4 … 1.9e-3) |
| DiT SFT bf16 vs fp32 oracle | relL2 1.2e-2 … 2.7e-2 (torch bf16 floor not yet measured) |
| RoPE table · modulation · head modulation | 1.7e-8 · 5.9e-8 · 6.4e-8 |
| e2e 8-step 512², SFT fp32 | latents 2.0e-3, image 1.1e-3 (torch fp32 on MPS vs CPU: 2.3e-3 / 1.3e-3) |
| e2e 8-step 512², Base fp32 | latents 1.5e-4, image 5.2e-5 |
| e2e 8-step 512², SFT bf16 vs fp32 oracle | latents 0.209, image 0.131 — **inside** torch's own bf16 drift (MPS: 0.283 / 0.173) |
| DiT int8 (bf16 load, quantized), 12 cases | cos 0.9992–0.9999, relL2 1.2e-2 … 4.0e-2 → declared |
| DiT int4, 12 cases | cos 0.936–0.999 (t500 / t37 fail the ≥ 0.99 gate) → not declared |
| TE int8 (bf16 load, Linears quantized), 4 prompts | relL2 2.1e-2 … 3.3e-2, cos ≥ 0.99944 (gate relL2 ≤ 5e-2, cos ≥ 0.999) → declared |

**e2e threshold.** The 8-step trajectory amplifies per-forward error ~400× by step 6 (σ 0.791 → 0.671 →
0.463). The oracle's own fp32 run, with a random 5e-6 relative perturbation per forward, drifts 7.2e-4
(`e2e_sensitivity.py`). The gate is therefore 5e-3 on latents and 2e-3 on the image.

**Two bugs found and fixed by the gates:**
1. **GDN l2norm eps.** mlx-swift-lm's `Qwen35GatedDeltaNet` (copied from mlx-lm) normalises q and k as
   `rmsNorm(x, eps 1e-6) · Dk^-½`, which puts the eps on the *mean* (128e-6 on the sum) instead of FLA/HF's
   1e-6 on the sum. Against HF that is a uniform ~1e-3 relL2 from the first GDN layer whose q/k norms are
   small (layer 1). Fixed here; the upstream mlx-swift-lm and mlx-lm still carry it.
2. **Time-MLP keys.** `time_embedding.0/.2` crashed `update(parameters:)` (digit keys → array). Renamed at
   load.

**First real render.** SFT, 1024², 50 steps, seed 42, the model card's fridge-note prompt:
- The three Chinese lines render exactly.
- 2.27 s/step (two forwards), 113 s denoise, 0.71 s decode.
- MLX peak 26.2 GB with the encoder co-resident during encoding.

**Footprints** (`MossImageGate membench`, the package's own memory path, MLX pool):

| Tier | Resident | Peak 1024² | Peak 1456² (cap) | Denoise + decode 1024² / 1456² | s/step 1024² (50-step run) |
|---|---|---|---|---|---|
| bf16 | 10.28 GB | 26.19 GB | 26.19 GB | 17.40 / 24.66 GB | 2.27 |
| int8 | 5.61 GB | 21.52 GB | 21.52 GB | 12.73 / 19.99 GB | 2.45 |
| int4 (not declared) | 3.11 GB | 19.03 GB | 19.03 GB | 10.24 / 17.50 GB | 2.33 |

- The peak is the encode phase everywhere inside the envelope: the per-request 16.8 GB bf16 encoder.
- Declared: resident 10.5 GB (bf16) and 5.8 GB (int8); activation 16.5 GB.
- A caller-scope encoder binding kept the encoder alive through denoising (+6 GB at 1024²). It was caught by
  membench and fixed by scoping the encoder to the conditioning call.

**Renders** (1024², 50 steps, seed 42):
- int8 matches bf16 at 27.3 dB PSNR (the same image).
- int4 lands at 18.6 dB: a coherent, text-correct but different image.
- Base: text correct, less polished composition.
- A JSON layout prompt (headline, centre circle, date line) follows its boxes exactly.

### 4.1 int8 text encoder (`textEncoderQuant: .int8`)

**Recipe.** `MossImageWeights.quantizeTextEncoder` — affine, group 64, 8 bits, on every Linear of the
Qwen3.5-9B encoder except the GDN decay-gate inputs `in_proj_a` / `in_proj_b` (32 outputs each, 0.5 MB per
layer). `embed_tokens`, every RMSNorm, `conv1d`, `A_log` and `dt_bias` are not Linears and are never touched.
- `loadTextEncoder(quantBits:)` binds the lazy safetensors loads, quantizes, then evaluates one decoder
  layer at a time. The bf16 copy is never resident in full: the TE gate's process peak is 10.0 GB.
- Weights: **9.39 GB** (was 16.8 GB). `embed_tokens` is 2.0 GB of that (248 320 × 4096 bf16).
- Load + quantize: 9–12 s per request.

**Gate** (`MossImageGate te --dtype bf16 --quant-te int8`, vs the fp32 oracle; no torch int8 reference
exists, so the gate is relL2 ≤ 5e-2 ≈ 2.5× the worst torch bf16 floor, and cos ≥ 0.999):

| Prompt | int8 + bf16 act | int8 + fp32 act | our bf16 | torch bf16 floor |
|---|---|---|---|---|
| cat | 2.56e-2 · cos 0.99967 | 1.62e-2 | 1.60e-2 | 1.60e-2 |
| empty | 3.34e-2 · cos 0.99944 | 2.13e-2 | 1.85e-2 | 2.07e-2 |
| note (CJK) | 2.09e-2 · cos 0.99978 | 1.43e-2 | 1.21e-2 | 1.18e-2 |
| layout (JSON) | 2.57e-2 · cos 0.99967 | 1.53e-2 | 2.09e-2 | 1.35e-2 |

- The quantization error alone (fp32 activations) is the size of torch's own bf16 floor. With bf16
  activations it stacks to 1.3–2.2× the floor.
- Quantizing `in_proj_a/b` as well (`MOSS_TE_QUANT_ALL=1`) changes nothing measurable (2.0–3.3e-2). They
  stay bf16 because they cost nothing.

**End to end** (fridge prompt, 1024², 50 steps, seed 42, PSNR vs the bf16-DiT / bf16-encoder render):

| DiT / encoder | PSNR | Note text |
|---|---|---|
| int8 / bf16 (before) | 27.34 dB | exact |
| bf16 / int8 | 31.09 dB | exact |
| int8 / int8 | 27.13 dB (35.85 dB vs int8 / bf16) | exact |

The encoder costs less than the DiT tier does: int8 / int8 is the same image as int8 / bf16. All three Chinese lines
render exactly in every arm (`outputs/note_crops_te_int8.png`).

**Footprints** (membench, int8 encoder):

| DiT | Resident | Peak 1024² (encode / denoise+decode) | Peak 1456² (cap) | Activation at the cap |
|---|---|---|---|---|
| bf16 | 10.28 GB | 20.09 GB (20.09 / 17.40) | 24.66 GB (denoise) | 14.39 GB |
| int8 | 5.61 GB | 15.42 GB (15.42 / 12.73) | 19.99 GB (denoise) | 14.39 GB |

- With the int8 encoder the encode phase is no longer the peak at the cap; the 1456² denoise is.
- The lane is declared as `FootprintConfigured.peakActivationBytesHint` = **14.9 GB** (+3.6%) over either DiT
  tier, and resident stays keyed on the DiT quant.
- So the declared activation drops only 16.5 → 14.9 GB. The 6.1 GB the encoder saves at 1024² would show in
  admission only with a lower envelope cap. At 1024², int8 / int8 peaks at 15.42 GB
  (was 21.52 GB) and bf16 / int8 at 20.09 GB (was 26.19 GB).

### 4.2 In-app re-declaration (0.2.1, AB-T-0317)

Pixel Agency Image Engine 0.2.13 (Release), built against a local mlx-image-runtime on this package; MES
`image.generate`, the bookstore poster at seed 7, 50 steps, CFG 3; `phys_footprint` ledger peak
(`WIP/moss-image-inapp/run-lanes.sh`, AB-R-0575 method).

| Lane (DiT / encoder) | Resident | Peak 1024² | Peak 1456² (cap) | Declared 0.2.1 |
|---|---|---|---|---|
| bf16 / bf16 | 10.43 GB | 26.45 GB (encode) | 27.28 GB (decode) | 10.5 + 17.5 = 28.0 GB |
| bf16 / int8 (AB-R-0575) | 10.44 GB | 20.38 GB | 27.28 GB | 28.0 GB |
| int8 / bf16 (AB-R-0570) | 5.6 GB | 21.87 GB | not run (≈22.6 GB) | 5.8 + 17.5 = 23.3 GB |
| int8 / int8 | 5.77 GB | 15.73 GB (encode) | 22.62 GB (decode) | 23.3 GB |

- At the cap the VAE decode is the peak: 16.84 GB phys over resident on every lane. The decode doesn't depend on
  the encoder: bf16/bf16 and bf16/int8 both peak at 27.28 GB.
- The MLX pool (membench) understated it by 2.6 GB, so 0.2.0 under-declared every lane at the cap.
- **Declared activation is 17.5 GB on every lane** (16.84 GB + 3.6 %). The int8 encoder's 14.9 GB
  `peakActivationBytesHint` is gone, and every lane uses the quant-keyed footprint.
- Outputs are byte-identical to 0.2.0 (declaration-only change); every string in the 1456² bf16 render is exact.
- The int8 encoder still lowers the peak by 6.1 GB below the cap. Admission would see it only with per-size footprints
  or a lower cap.

### 4.3 Row-band VAE decode (0.3.0)

`MossImagePipeline.vaeTile` defaults to `.standard`: flux2-vae-mlx-swift ≥ 0.6.0's `decodePackedLatents(_:tile:)`
(64 MB row bands, whole-tensor GroupNorm stats, one-row conv halo; AB-R-0577). `nil` decodes untiled. The banded path
is generic over the decoder widths, so BFL's small decoder (96/192/384/384) runs on it unchanged.

**Parity** (`MossImageGate vae` / `vae-tiled`, fp32, TF32 off):
- Oracle golden (512×384), 1 MB bands: relL2 1.2e-6 on both conv routes (untiled: 3.6e-6 default, 1.2e-6 exact).
- Tiled vs untiled, the golden latent tiled to size: default route 124.4 dB at 1024² and 124.6 dB at 1456² (relL2
  5e-6, 99.99 % of 8-bit pixels identical). Exact route: 137.1 and 135.7 dB. No row stands out (worst-row maxAbs
  4.6e-5 / 4.9e-5), so there are no seams.
- MLX pool over the resident VAE: 1024² 7.13 → 1.63 GB, 1456² 14.39 → 2.89 GB; decode 0.68 → 1.00 s and
  1.49 → 1.93 s.

**membench** (pool, 2 steps): the 1456² decode drops 24.66 → 13.14 GB (bf16 DiT) and 19.99 → 8.47 GB (int8). The
encode is now the peak at both sizes: 26.19 GB bf16/bf16, 15.42 GB int8/int8.

**In-app** (Image Engine 0.2.13 clone, Release, against a local mlx-image-runtime on this tree; `run-lanes.sh`,
TAGP v030, `quiet.sh` before each lane, no foreign Metal process):

| Lane (DiT / encoder) | Resident | Peak 1024² | Peak 1456² (cap) | Declared 0.3.0 |
|---|---|---|---|---|
| bf16 / bf16 | 10.44 GB | 26.45 GB (encode) | 26.45 GB (encode) | 10.5 + 16.7 = 27.2 GB |
| bf16 / int8 | 10.44 GB | 20.39 GB | 20.39 GB | 10.5 + 10.4 = 20.9 GB |
| int8 / bf16 | 5.77 GB | 21.80 GB | 21.80 GB | 5.8 + 16.7 = 22.5 GB |
| int8 / int8 | 5.77 GB | 15.73 GB | 15.73 GB | 5.8 + 10.4 = 16.2 GB |

- The ledger peak doesn't rise during any 1456² run. The decode's 1 s samples read 14.4–14.8 GB (bf16 DiT) and
  8.6–9.0 GB (int8), against 17.8 GB untiled (0.2.1).
- Declared activation: the bf16 encoder 16.7 GB (16.03 + 3.6 %), quant-keyed; the int8 encoder 10.4 GB
  (9.96 + 3.6 %), as `peakActivationBytesHint` again.
- Renders vs 0.2.1: 60.9 / 64.5 dB (bf16/bf16, 1024² / 1456²), 61.1 / 64.5 dB (int8/int8), max |Δ| ≤ 4/255,
  spread over every row (TF32 is on in production). Every string in both 1456² renders is exact.

## 5. Open

- 16 GB Macs: the int8 / int8 lane declares 5.8 + 10.4 = 16.2 GB (0.3.0), over a 16 GB Mac's budget. A further ~1 GB
  would come from quantizing `embed_tokens`; pre-quantized encoder weights would not move the peak.
- Ship pre-quantized int8 encoder weights: that removes the 9–12 s per-request quantize and the 16.8 GB
  download.
- Mirror and publish the weights; registry rows; in-app validation.
