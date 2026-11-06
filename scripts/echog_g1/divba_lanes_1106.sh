#!/bin/bash
# zehan-scmg-g1-divba-lanes-1106: Div/BA training candidates, 2000-step dev lanes from hos-base with the final causal recipe
# (causal_front + causal_spk + causal_text, w_safe 300, self_force_p 0, training seed 0), D-spk dev bench offline + psync
# (incremental feat89, speaker refresh grades) x inference seeds 0/1. Joint criteria fixed before (scmg 2726906).
# GPU0: ctrl -> spcon (w_spec 0.3 + w_onset 5) -> spec1 (w_spec 1.0); GPU1: spec03 (w_spec 0.3) -> onset5 (w_onset 5).
set -uo pipefail
L=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/dlc-logs/${DLC_JOB_ID:-manual}; mkdir -p $L
exec > >(tee -a $L/run.log) 2>&1
REL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/divba-1106/rel/20261106-100059-g1-divba-lanes
echo "99f74b088defe8a6ae4be7da04a92d0728738ef123489eeb4b0c79ba1bd8772a  $REL/scripts/echog_g1/causal_g1.sh" | sha256sum -c - || exit 3
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; Dd=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
export PY=/usr/local/bin/python BENCH_PY=/usr/local/bin/python EVAL_PY=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python
export EG=$Dd OFF=$W/official MMAE=$Dd/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export SPG=4 WORKERS=4 OMP_NUM_THREADS=2 USE_TF=0 SPK_REFRESH=grades CAUSAL_FEAT=inc
unset STEPS SCMG_LOCAL_MIRROR
O=$B/divba-1106/lanes
cd $REL && echo "release commit $(cat GIT_COMMIT) host $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
( while true; do echo "$(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | paste -sd';')"; sleep 60; done ) > $L/MONITOR.log 2>&1 &
MON=$!
lane() { GPU=$1 SIDE=dev G1_BENCH=1 SEEDS="0 1" BENCH_PROTOS="offline psync" EXTRA_SC="$3" bash scripts/echog_g1/causal_g1.sh lane $O $2 2000; echo "$(date -Is) LANE_END $2"; }
( lane 0 ctrl '"w_safe": 300'; lane 0 spcon '"w_safe": 300, "w_spec": 0.3, "w_onset": 5.0'; lane 0 spec1 '"w_safe": 300, "w_spec": 1.0' ) & P0=$!
( lane 1 spec03 '"w_safe": 300, "w_spec": 0.3'; lane 1 onset5 '"w_safe": 300, "w_onset": 5.0' ) & P1=$!
wait $P0 $P1
kill $MON; tail -n 30 $O/causal.log; ls $O/dev/*/bench/*/benchmark_seed_*.json; echo "$(date -Is) JOB_DONE"
