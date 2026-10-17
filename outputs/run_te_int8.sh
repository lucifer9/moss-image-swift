#!/bin/zsh
# int8 Qwen3.5-9B encoder (AB-T-0307): the fridge render with the int8 encoder over the int8 and bf16 DiT,
# each behind a quiet-GPU wait. Compare against gen_sft_fridge_1024_s50{,_int8}.png (bf16 encoder).
cd "$(dirname "$0")/.."
K=/Volumes/Satechi/Development/mlxengine-todo/probes/creative-upscale/kit
G=.build/release/MossImageGate
P=$(printf 'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"')
r() { zsh $K/gpu_quiet.sh 4 >&2; "$G" generate "$@" 2>&1 | grep -E "loaded|encoded|generate"; }
r --variant SFT --quant int8 --quant-te int8 --prompt "$P" --size 1024x1024 --steps 50 --seed 42 --out outputs/gen_sft_fridge_1024_s50_int8_te8.png
r --variant SFT --quant-te int8 --prompt "$P" --size 1024x1024 --steps 50 --seed 42 --out outputs/gen_sft_fridge_1024_s50_te8.png
echo done
