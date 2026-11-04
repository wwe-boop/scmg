#!/bin/bash
# Quick LLM-text lane: fine-tune r4 base with the sbag protocol, only the text cache is swapped (e5 -> LLM tokens).
#   r4_llm_lane.sh <gpu> <out root> <lane> [steps=2000]
# cfg/sc/batch/lr/bagger match scripts/r4_lane.sh's sbag lane + Config.text_dim for the LLM feature width.
# After training every 500-step snapshot and final.pt get the ROLLOUT_ONLY stillness curve (offline + engine, seeds 0/1).
set -uo pipefail
gpu=$1; R=$2; o=$3; ST=${4:-2000}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE
export PYTHONPATH=$CODE${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA} USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
unset SCMG_TEXT_CAUSAL                 # LLM features are natively prefix-causal (PREP native_causal=true)
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
INIT=${INIT:-$B/dlc-scmg-train-r4-1102/base/best.pt}; VAE=$B/trained/vae-r3/final.pt
SEMMAN=$B/semantic-recognizer-v1/manifest-n200
TEXT=${TEXT_CACHE:-$B/models/text-llm-qwen3-0.6b-v36_1}
TD=${TEXT_DIM:-1024}
BAGGER=${BAGGER_DIR:-$B/r4-1104/bagger-pai/bagger-data}
CFG="{\"adsc\": true, \"text_dim\": $TD}"
SC='{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0}'
mkdir -p $R/$o/curve; log(){ echo "$(date -Is) $*" >> $R/$o/lane.log; }

log "TEXT_CACHE $TEXT dim $TD"
$PY -m scmg.train --out $R/$o --init $INIT --stats $D/STATS_v36_1.json --text-cache $TEXT --vae $VAE --workers ${WORKERS:-6} \
  --tok-cache $D/tok-qwen25-v36_1 --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
  --holdout $SEMMAN/MANIFEST.json --semsynth-share ${SEMSHARE:-0.4} --batch ${BATCH:-64} --stage base --steps $ST --lr ${LR:-1e-4} \
  --warmup 200 --eval-every 500 --snap-every 500 --save-every 500 --seed ${TRAIN_SEED:-0} --cfg "$CFG" --sc "$SC" \
  --bagger-dir "$BAGGER" --bagger-share ${BAGGER_SHARE:-0.33} >> $R/$o/log.txt 2>&1
log "TRAIN_EXIT $?"

for f in $(ls $R/$o/snap_*.pt 2>/dev/null | sort -V) $R/$o/final.pt; do
  [ -f "$f" ] || continue
  lab=$(basename $f .pt)
  for sd in 0 1; do
    ROLLOUT_SEED=$sd ROLLOUT_TEXT_CACHE=$TEXT ROLLOUT_WINDOW_LM_TEXT=1 \
      $PY scripts/rollout_probe.py $VAE $lab=$f@${CFGW:-2.5} >> $R/$o/curve/offline.jsonl 2>> $R/$o/curve/probe.log
    ROLLOUT_SEED=$sd ROLLOUT_TEXT_CACHE=$TEXT ROLLOUT_TEXT_MODE=x2 ROLLOUT_X2_LAG=0.15 ROLLOUT_H=2 \
      $PY scripts/rollout_probe.py $VAE $lab=$f@${CFGW:-2.5} >> $R/$o/curve/engine.jsonl 2>> $R/$o/curve/probe.log
  done
  log "CURVE_DONE $lab"
done
touch $R/$o/DONE.json; log "LANE_DONE"
