#!/bin/bash
# 11-04 baseline protocol (user: quick-validate the base model first, offline AND engine-streaming).
#   baseline_eval.sh <gpu> <out root> <label> <ckpt> <protocol: offline|engine> <part: fgd|probes> [cfg-w=2.0]
# offline: final text (training windows for VoxCPM ids), H=inf;  engine: text streams word by word and EARLY (legacy x2,
# rate 15 tok/s, lag .15 s, causal e5, deploy timing), H=2 (160 ms).  Both: trim_start + cfg_null_fix, 10 steps, seeds 0/1.
set -uo pipefail
gpu=$1; R=$2; lab=$3; ck=$4; P=$5; part=$6; w=${7:-2.0}
CODE=$(cd "$(dirname "$0")/.." && pwd); cd $CODE
export PYTHONPATH=$CODE${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA} USE_TF=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=$gpu
export SCMG_TEXT_CAUSAL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/text-e5-causal-v36_1
PY=${PY:-python}; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
VAE=$B/trained/vae-r3/final.pt; TOKC=$D/tok-qwen25-v36_1; SEMMAN=$B/semantic-recognizer-v1/manifest-n200
O=$R/$lab/$P; mkdir -p $O
WL=--window-lm-text; [ "${NOVOX:-0}" = 1 ] && WL=""          # chunk/main planners have no LM ids to window
if [ $P = offline ]; then EV=(--horizons inf $WL); GS=(--horizon inf); RO="ROLLOUT_WINDOW_LM_TEXT=1"; HT=Hinf
else EV=(--horizons 2 $WL --text-mode x2 --x2-rate 15 --x2-lag 0.15 --deploy-timing); GS=(--horizon 2 --text-mode x2 --x2-rate 15 --x2-lag 0.15 --deploy-timing); RO="ROLLOUT_TEXT_MODE=x2 ROLLOUT_X2_LAG=0.15 ROLLOUT_H=2"; HT=H2; fi
log(){ echo "$(date -Is) $*" >> $O/runner.log; }
SEEDS="0 1"; SEMN=200; [ "${QUICK:-0}" = 1 ] && { SEEDS=0; SEMN=60; }   # quick tier: seed 0, 60 semantic clips
if [ $part = fgd ]; then
  for s in 0 1; do
    od=$O/eval-s$s; [ -f $od/RESULTS.json ] && continue
    SCMG_SEED0=$s SCMG_SAVE_N=43 $PY -m scmg.evaluate --ckpt $ck --vae $VAE --stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 \
      --tok-cache $TOKC --cfg-w $w --n 43 --ndiv 0 --mmap --trim-start --cfg-null-fix "${EV[@]}" --out $od >> $O/eval-s$s.log 2>&1 \
      && log "EVAL_DONE s$s" || { log "EVAL_FAILED s$s"; exit 1; }
  done
else
  for s in $SEEDS; do
    f=$O/rollout-s$s.json; grep -q '^{' $f 2>/dev/null && continue
    env $RO ROLLOUT_SEED=$s ROLLOUT_FIX=1 $PY scripts/rollout_probe.py $VAE $lab=$ck@$w > $f.tmp 2>> $O/rollout.log && grep -q '^{' $f.tmp && mv $f.tmp $f \
      && log "ROLLOUT_DONE s$s" || { log "ROLLOUT_FAILED s$s"; exit 1; }
  done
  for s in $SEEDS; do
    so=$O/semantic-s$s; [ -f $so/DONE ] && continue
    $PY scripts/gen_semantic.py $SEMMAN $so $lab=$ck@$w --variants own,notext,swap --n $SEMN --trim-start --cfg-null-fix --seed $s "${GS[@]}" >> $O/semantic.log 2>&1 || { log "SEMGEN_FAILED s$s"; exit 1; }
    ok=1; for v in own notext swap; do
      $PY scripts/semantic_eval.py score --gen-dir $so/$lab/$v --manifest $SEMMAN/MANIFEST.json --out $so/score-$v.json >> $O/semantic.log 2>&1 && grep -q top1 $so/score-$v.json || ok=0
    done; [ $ok = 1 ] && touch $so/DONE && log "SEMANTIC_DONE s$s" || { log "SEMSCORE_FAILED s$s"; exit 1; }
  done
fi
log "PART_DONE $part"
