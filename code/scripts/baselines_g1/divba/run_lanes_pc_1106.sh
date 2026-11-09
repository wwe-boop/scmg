#!/bin/bash
# zehan-scmg-g1-divba-lanespc-1106: per_clip bench of the five 2000-step Div/BA lanes trained in dlc116vua3f481li
# (hos-base + final causal recipe + w_safe 300; ctrl / w_spec 0.3 / w_spec 1.0 / w_onset 5 / w_spec 0.3 + w_onset 5).
export REL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/rel/20261106-130816-g1-divba-pc OUT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/lanes_pc COMMON_EXTRA='--seed-mode per_clip' CKS=. GPUS='0 0 0 0 1 1 1 1'
L=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/lanes/dev
export CONFIGS="ctrl|10||$L/ctrl/final.pt
spec03|10||$L/spec03/final.pt
spec1|10||$L/spec1/final.pt
onset5|10||$L/onset5/final.pt
spcon|10||$L/spcon/final.pt"
exec /bin/bash /mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/dlc/dlc_infer_v2.sh
