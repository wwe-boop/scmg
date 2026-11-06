#!/bin/bash
# zehan-scmg-g1-divba-ztpc-1106: zero-training Div/BA candidates on czm6k dev under the per_clip seed protocol
# (exp/divba ebb03ae, --seed-mode per_clip); control rerun. D-spk dev only. Criteria: CRITERIA.md (b0941c1).
export REL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/rel/20261106-130816-g1-divba-pc OUT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/zt_pc COMMON_EXTRA='--seed-mode per_clip' GPUS='0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3'
export CONFIGS='ctrl|10|
st16|16|
ns09|10|--noise-scale 0.9
g05t05|10|--sde-eta 0.5 --sde-tmax 0.5
g1t05|10|--sde-eta 1.0 --sde-tmax 0.5
g1t03|10|--sde-eta 1.0 --sde-tmax 0.3
g2t03|10|--sde-eta 2.0 --sde-tmax 0.3'
exec /bin/bash /mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/dlc/dlc_infer_v2.sh
