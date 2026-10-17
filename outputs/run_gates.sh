#!/bin/zsh
# MOSS-Image parity gates (AB-T-0307), each behind a quiet-GPU wait (hold-gpu-under-contention kit).
cd "$(dirname "$0")/.."
K=/Volumes/Satechi/Development/mlxengine-todo/probes/creative-upscale/kit
G=.build/release/MossImageGate
run() { local name=$1; shift; zsh $K/gpu_quiet.sh 4 >&2; echo "=== $name: $G $*" ; $G "$@" > outputs/gates/$name.log 2>&1; echo "exit $?" >> outputs/gates/$name.log; grep -E "PASS|FAIL|exit" outputs/gates/$name.log; }
[ -n "$ONLY_REMAINING" ] || { run vae vae; run te_f32 te --dtype f32; run te_bf16 te --dtype bf16; }
MLX_ENABLE_TF32=0 run te_f32_notf32 te --dtype f32
run dit_sft_f32 dit --variant SFT --dtype f32
run dit_sft_bf16 dit --variant SFT --dtype bf16
run dit_base_f32 dit --variant Base --dtype f32
run e2e_sft_f32 e2e --variant SFT --dtype f32
run e2e_sft_bf16 e2e --variant SFT --dtype bf16
run e2e_base_f32 e2e --variant Base --dtype f32
MLX_ENABLE_TF32=0 run dit_sft_f32_notf32 dit --variant SFT --dtype f32
