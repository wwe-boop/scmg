#!/bin/bash
# bagger_rounds.sh GPU null|reverse [--dry-run]
# COUNT, ROUNDS, STEPS, SHARE, BATCH, WORKERS are configurable. Each lane uses ONE GPU.
# Each round collects from the previous round's FINAL, aggregates all preceding
# collections, but reinitialises training from the SAME initial checkpoint.
set -euo pipefail
source "$(dirname "$0")/bagger_env.sh"
export CUDA_VISIBLE_DEVICES=$1
cond=$2
[[ "$cond" == null || "$cond" == reverse ]] || { echo 'cond must be null or reverse'; exit 2; }
dry=${3:-}
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
ROOT=${BAGGER_ROOT:-$B/bagger-1103}
INIT=${INIT:-$B/dlc-scmg-train-r7c-1102-voxabl/vox-randlm/best.pt}
VAE=$B/trained/vae-r3/final.pt
COUNT=${COUNT:-2000}; ROUNDS=${ROUNDS:-3}; STEPS=${STEPS:-4000}; SHARE=${SHARE:-0.33}
CFG=${CFG:-'{"vox_prefsq_norm": true}'}
# Match ft_lane.sh defaults on original examples. Corrective examples retain
# their collected history even if self-forcing fires for the real examples.
SC=${SC:-'{}'}
LANE=$ROOT/bagger-$cond
if [[ "$dry" == --dry-run ]]; then
  "$PY" scripts/bagger_collect.py --ckpt "$INIT" --out "$LANE/round-1/data" --cond "$cond" --count "$COUNT" --dry-run
  printf 'rounds=%s steps=%s batch=%s lr=1e-4 warmup=200 share=%s frames=256 cfg=%s sc=%s init=%s\n' "$ROUNDS" "$STEPS" "${BATCH:-32}" "$SHARE" "$CFG" "$SC" "$INIT"
  exit 0
fi
mkdir -p "$LANE"
prev=$INIT; aggregate=
for ((round=1; round<=ROUNDS; round++)); do
  R=$LANE/round-$round; mkdir -p "$R"
  if [[ ! -f "$R/data/MANIFEST.json" ]]; then
    "$PY" scripts/bagger_timed.py --ledger "$LANE/timing.jsonl" --stage collect-$round -- \
      "$PY" scripts/bagger_collect.py --ckpt "$prev" --out "$R/data" --cond "$cond" --count "$COUNT" --seed "$((1000*round))" \
      >> "$R/collect.log" 2>&1
  fi
  aggregate=${aggregate:+$aggregate:}$R/data
  if [[ ! -f "$R/train/DONE.json" ]]; then
    "$PY" scripts/bagger_timed.py --ledger "$LANE/timing.jsonl" --stage train-$round -- \
      "$PY" -m scmg.train --out "$R/train" --init "$INIT" --stats "$D/STATS_v36_1.json" \
      --text-cache "$D/text-e5-v36_1" --vae "$VAE" --workers "${WORKERS:-0}" --mmap \
      --tok-cache "$D/tok-qwen25-v36_1" --tok-intent "$B/models/intent-qwen25-v36_1" \
      --intent-frames "$B/models/intent-frames-v36_1" --holdout "$B/semantic-recognizer-v1/manifest-n200/MANIFEST.json" \
      --batch "${BATCH:-32}" --stage base --steps "$STEPS" --lr 1e-4 --warmup 200 --eval-every "${EVAL_EVERY:-500}" \
      --snap-every 0 --cfg "$CFG" --sc "$SC" --bagger-dir "$aggregate" --bagger-share "$SHARE" \
      >> "$R/train.log" 2>&1
  fi
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["step"]==int(sys.argv[2]),d' "$R/train/DONE.json" "$STEPS"
  # Task-specific instruction explicitly requests deletion of OUR resume.pt after training.
  if [[ -f "$R/train/resume.pt" ]]; then rm -- "$R/train/resume.pt"; fi
  prev=$R/train/final.pt
  for protocol in default x2; do
    result=$R/rollout-$protocol.jsonl
    if [[ ! -s "$result" ]]; then
      mode=final; [[ "$protocol" == x2 ]] && mode=x2
      ROLLOUT_TEXT_MODE=$mode "$PY" scripts/bagger_timed.py --ledger "$LANE/timing.jsonl" --stage eval-$round-$protocol -- \
        "$PY" scripts/rollout_probe.py "$VAE" "bagger-$cond-r$round=$prev@2.0" > "$result.tmp" 2>> "$R/probe.log"
      "$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert all(k in x for k in ("free","oracle"))' "$result.tmp"
      mv "$result.tmp" "$result"
    fi
  done
  "$PY" scripts/bagger_report.py --root "$ROOT" --out "$ROOT/RESULTS.json"
done
printf '%s LANE_DONE %s\n' "$(date -Is)" "$cond"
