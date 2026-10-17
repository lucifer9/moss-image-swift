#!/bin/zsh
# MOSS-Image footprints (membench, MLX pool) + quant-tier DiT gates, each behind a quiet-GPU wait.
cd "$(dirname "$0")/.."
K=/Volumes/Satechi/Development/mlxengine-todo/probes/creative-upscale/kit
G=.build/release/MossImageGate
run() { local name=$1; shift; zsh $K/gpu_quiet.sh 4 >&2; echo "=== $name: $*"; $G "$@" > outputs/gates/$name.log 2>&1; echo "exit $?" >> outputs/gates/$name.log; grep -E "membench|peak|PASS|FAIL|exit" outputs/gates/$name.log | cut -c1-160; }
run membench_bf16 membench --variant SFT
run membench_int8 membench --variant SFT --quant int8
run membench_int4 membench --variant SFT --quant int4
run dit_sft_int8 dit --variant SFT --dtype bf16 --quant int8
run dit_sft_int4 dit --variant SFT --dtype bf16 --quant int4
