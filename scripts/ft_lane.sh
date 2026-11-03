#!/bin/bash
# 11-03: fine-tune lane from an existing checkpoint on our own GPUs, then probes + evals (text path / hold-fix validation).
#   ft_lane.sh <gpu> <out root> <lane> <init ckpt> <cfg-json (new switches)> <sc-json> [cfg-w=2.0]
# order: train -> planner_probe (text/audio reliance) -> semantic own/notext/swap (+score) -> rollout_probe -> FGD eval seed 0
set -uo pipefail
gpu=$1; R=$2; o=$3; init=$4; cf=$5; scj=$6; w=${7:-2.0}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE; export PYTHONPATH=$CODE USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
VAE=$B/trained/vae-r3/final.pt; TOKC=$D/tok-qwen25-v36_1; SEMMAN=$B/semantic-recognizer-v1/manifest-n200; ST=${STEPS:-4000}
source "$CODE/scripts/artifacts.sh"
$PY -m scmg.token_cache "$init" --entry ft_lane.sh --cfg "$cf" || exit 1
mkdir -p $R/$o
if ! valid_json "$R/$o/DONE.json" train; then
  $PY -m scmg.train --out $R/$o --init $init --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} \
    --tok-cache $TOKC --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $SEMMAN/MANIFEST.json --semsynth-share ${SEMSHARE:-0.4} --batch ${BATCH:-32} --stage base --steps $ST --lr ${LR:-1e-4} --warmup $((ST / 20)) \
    --eval-every $((ST / 8)) --snap-every 0 --cfg "$cf" --sc "$scj" >> $R/$o/log.txt 2>&1 || { echo "$(date -Is) TRAIN FAILED $o"; exit 1; }
  valid_json "$R/$o/DONE.json" train || exit 1
  rm -f $R/$o/resume.pt
fi
json_run "$R/$o/planner_probe.json" planner stdout $PY scripts/planner_probe.py "$VAE" "$o=$R/$o/best.pt" 2>> "$R/$o/probe.log" || exit 1
so=$R/$o/semantic
semantic_run "$so" "$o" "$R/$o/best.pt" "$w" own,notext,swap 200 >> "$R/$o/semantic.log" 2>&1 || exit 1
json_run "$R/$o/rollout_probe.json" rollout stdout $PY scripts/rollout_probe.py "$VAE" "$o=$R/$o/best.pt@$w" 2>> "$R/$o/probe.log" || exit 1
ROLLOUT_WINDOW_LM_TEXT=1 json_run "$R/$o/rollout_probe_win.json" rollout stdout $PY scripts/rollout_probe.py "$VAE" "$o=$R/$o/best.pt@$w" 2>> "$R/$o/probe.log" || exit 1
for protocol in legacy window; do
  suffix=""; args=()
  if [ "$protocol" = window ]; then suffix=-win; args+=(--window-lm-text); fi
  od=$R/$o/eval-best$suffix
  if ! valid_json "$od/RESULTS.json" eval; then
    SCMG_SAVE_N=43 "$ARTIFACT_PY" "$CODE/scripts/artifacts.py" run "$od" eval directory $PY -m scmg.evaluate       --ckpt "$R/$o/best.pt" --vae "$VAE" --stats "$D/STATS_v36_1.json" --text-cache "$D/text-e5-v36_1"       --tok-cache "$TOKC" --cfg-w "$w" --n 43 --horizons inf,0 --ndiv 20 --mmap "${args[@]}" >> "$R/$o/eval.log" 2>&1 || exit 1
  fi
done
echo "$(date -Is) LANE_DONE $o"
