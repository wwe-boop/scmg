#!/bin/bash
# 11-05 text utilisation of ECHO-G vs SCMG under ONE protocol, on a fixed val_common subset (default: the 300-clip RoboGesture
# subset): same audio, same inference seed, text = own / none / swapped (donor = length neighbour, swap_map.json shared by
# both models, word times rescaled to this clip's duration); plus own text with inference seed 1 (sampling floor).
#   ECHO-G: official SGDiT sampler, 8 steps, guidance 1 (as published); none = its null text (1 zero token, time distance 0)
#   SCMG  : test-side sf0 (fix-base warm start), offline protocol, CFG 2.5, 10 steps; none = --modality audio
# Outputs per run: official benchmark on the subset (FGD / dDiv / dBA / ...), semantic recognition (v1 recognisers,
# --allow-missing on the subset events), text sensitivity (text_sensitivity.py: robot_repr joint angles, upper body and all
# joints, d(own, other text) / d(own seed 0, own seed 1)).  val_common use is appended to TEST_USAGE.jsonl.
#   textuse_g1.sh <out dir (with swap_map.json + subset_stems.txt)>     env EGPU, SGPU, PY (echog env)
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; SB=$B/semantic-baseline-1105/g1
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
ST=$O/subset_stems.txt; MAP=$O/swap_map.json; CK=${SCMG_CKPT:-$B/echog-g1-sf-1105/test/sf0/final.pt}; DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt
export PY EG=$D OFF=$W/official REF_DIR=$D/motion_39d_30fps WAV_DIR=$D/audio SPLIT=$ST MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
log() { echo "$(date -Is) $*" >> $O/textuse.log; }
use() { echo "{\"time\": \"$(date -Is)\", \"split\": \"val_common subset $(grep -c . $ST) ($(basename $ST))\", \"model\": \"$1\", \"setting\": \"$2\", \"purpose\": \"report: text-utilisation probe ECHO-G vs SCMG (protocol fixed before the run)\", \"host\": \"$(hostname)\"}" >> $O/TEST_USAGE.jsonl; }
[ -f $ST ] && [ -f $MAP ] || { echo "need $ST and $MAP"; exit 2; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
(  # ECHO-G: own / notext / swap seed 0, own seed 1
  use echog "own notext swap seed0 + own seed1"
  for spec in "own notext swap:0" "own:1"; do
    CUDA_VISIBLE_DEVICES=${EGPU:-0} PYTHONPATH=$W/official/src $PY scripts/echog_g1/echog_textuse.py --checkpoint $W/weights/best.pt \
      --config $W/weights/configs/sgdit_audio_text.yaml --cond-dir $D/condition_30fps --meta $D --stems $ST --swap-map $MAP \
      --out $O/echog --modes ${spec%%:*} --seed ${spec##*:} >> $O/echog.log 2>&1 || log "ECHOG_FAILED $spec"
  done; log "ECHOG_SAMPLED"
) &
(  # SCMG
  use scmg "both seed0/seed1, --modality audio, --text-swap (shared map)"
  for spec in "own:0:" "own:1:" "notext:0:--modality audio" "swap:0:--text-swap --text-swap-map $MAP"; do
    IFS=: read lab sd ex <<< "$spec"; P=$O/scmg/${lab}_seed_$(printf %03d $sd); mkdir -p $P
    CUDA_VISIBLE_DEVICES=${SGPU:-0} SCMG_J=39 SCMG_NO_GLITCH=1 $PY scripts/echog_g1/infer_test.py --ckpt $CK --vae $DEC \
      --stats $C/scmg/STATS_echog_g1_v1.json --view $C/echog-g1-v1/test --text-cache $C/scmg/text-e5-echog-g1-v1-test --out $P \
      --stems $ST --seed $sd --steps 10 --cfg-w 2.5 --protocol offline $ex >> $O/scmg.log 2>&1 || log "SCMG_FAILED $spec"
  done; log "SCMG_SAMPLED"
) &
wait
for m in echog scmg; do for r in own_seed_000 own_seed_001 notext_seed_000 swap_seed_000; do
  f=$O/benchmark_${m}_$r.json; [ -f $f ] && continue
  OMP_NUM_THREADS=${EVAL_OMP:-4} bash scripts/echog_g1/bench.sh eval $O/$m/$r $ST ${m}_$r $f >> $O/eval.log 2>&1 && log "EVAL_DONE $m $r" || log "EVAL_FAILED $m $r"
done; done
S=$O/sem; mkdir -p $S
for rec in $SB/rec-sem-v1 $SB/rec-sem-v1-seed1 $SB/rec-sem-v1-seed2; do
  rn=$(basename $rec)
  [ -s $S/$rn--gt--test.json ] || CUDA_VISIBLE_DEVICES=${SGPU:-0} $PY scripts/semantic/g1_recognizer.py score --data $SB/data-sem --ckpt $rec/model.pt \
    --name gt --split test --allow-missing --stems $ST --out $S/$rn--gt--test.json --save-probs >> $O/sem.log 2>&1
  for m in echog scmg; do for r in own_seed_000 notext_seed_000 swap_seed_000; do
    o=$S/$rn--${m}-${r%_seed_000}--test.json; [ -s $o ] && continue
    CUDA_VISIBLE_DEVICES=${SGPU:-0} $PY scripts/semantic/g1_recognizer.py score --data $SB/data-sem --ckpt $rec/model.pt --name ${m}-${r%_seed_000} \
      --pt-dir $O/$m/$r --split test --allow-missing --out $o --save-probs >> $O/sem.log 2>&1 || log "SEM_FAILED $rn $m $r"
  done; done
done
$PY scripts/semantic/sem_report.py --scores $S --rec-prefix rec-sem-v1 --split test \
  --pairs echog-notext:echog-own,echog-swap:echog-own,scmg-notext:scmg-own,scmg-swap:scmg-own --out $S/REPORT.json >> $O/sem.log 2>&1 && log "SEM_DONE"
$PY scripts/echog_g1/text_sensitivity.py --out $O/TEXT_SENS.json \
  echog_swap=$O/echog/own_seed_000,$O/echog/swap_seed_000,$O/echog/own_seed_001 echog_notext=$O/echog/own_seed_000,$O/echog/notext_seed_000,$O/echog/own_seed_001 \
  scmg_swap=$O/scmg/own_seed_000,$O/scmg/swap_seed_000,$O/scmg/own_seed_001 scmg_notext=$O/scmg/own_seed_000,$O/scmg/notext_seed_000,$O/scmg/own_seed_001 \
  >> $O/textuse.log 2>&1 && log "TEXT_SENS_DONE"
log "TEXTUSE_DONE"
