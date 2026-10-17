#!/bin/zsh
cd "$(dirname "$0")/.."
K=/Volumes/Satechi/Development/mlxengine-todo/probes/creative-upscale/kit
G=.build/release/MossImageGate
P=$(printf 'A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:\n"记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。"')
L='{"summary":{"caption":"A clean minimalist conference poster on a light grey background. A bold black headline at the top, a large orange circle in the centre, and a small date line at the bottom."},"items":[{"type":"text","text":"PIXEL AGENCY","bbox":[-420,-380,-300,380]},{"type":"object","description":"a large flat orange circle","bbox":[-220,-220,220,220]},{"type":"text","text":"OCTOBER 17, 2026","bbox":[340,-240,400,240]}],"bbox_format":"yxyx","bbox_coord_space":"abs_centralized"}'
r() { zsh $K/gpu_quiet.sh 4 >&2; "$G" generate "$@" 2>/dev/null | tail -1; }
r --variant SFT --quant int8 --prompt "$P" --size 1024x1024 --steps 50 --seed 42 --out outputs/gen_sft_fridge_1024_s50_int8.png
r --variant SFT --quant int4 --prompt "$P" --size 1024x1024 --steps 50 --seed 42 --out outputs/gen_sft_fridge_1024_s50_int4.png
r --variant Base --prompt "$P" --size 1024x1024 --steps 50 --seed 42 --out outputs/gen_base_fridge_1024_s50.png
r --variant SFT --prompt "$L" --size 1024x1024 --steps 50 --seed 7 --out outputs/gen_sft_layout_poster_1024_s50.png
echo done
