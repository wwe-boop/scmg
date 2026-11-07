#!/bin/bash
# 11-07 re-measurement of text use, ECHO-G vs SCMG, under the per_clip protocol (user / coordinator 11-06: the 11-05 SCMG numbers of
# textuse_g1.sh shared one noise draw across all clips of a run). Same fixed val_common subset (300-clip RoboGesture subset) and the
# same shared swap map as modality-ablation-1105/echog-textuse; inference seeds 0 / 1 / 2 for own / notext / swap on BOTH models:
#   SCMG  : czm6k (final recipe, test side), offline protocol, CFG 2.5, 10 steps, --seed-mode per_clip; notext = --modality audio
#   ECHO-G: official SGDiT sampler as published (8 steps, guidance 1; per-clip noise by construction); notext = its null text
# Outputs: official benchmark per run (subset), recognition with the v1 recognisers (train split only) scored as <model>-<mode>-pc-s<k>,
# sem_multiseed.py (3 seeds pooled, recording-clustered bootstrap: own - notext, own - swap), text sensitivity per seed.
# val_common use is appended to TEST_USAGE.jsonl BEFORE any sampling.
#   textuse_pc_g1.sh <out dir> <dir with subset_stems.txt + swap_map.json>     env EGPU, SGPU, PY (echog env), SCMG_CKPT
set -uo pipefail
O=$1; SRC=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; SB=$B/semantic-baseline-1105/g1
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
cp $SRC/subset_stems.txt $SRC/swap_map.json $O/; ST=$O/subset_stems.txt; MAP=$O/swap_map.json
CK=${SCMG_CKPT:-$B/causal-train-1105/runs-local/test/czm6k/final.pt}; DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt
export PY EG=$D OFF=$W/official REF_DIR=$D/motion_39d_30fps WAV_DIR=$D/audio SPLIT=$ST MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
log() { echo "$(date -Is) $*" >> $O/textuse.log; }
use() { echo "{\"time\": \"$(date -Is)\", \"split\": \"val_common subset $(grep -c . $ST) ($(basename $ST))\", \"model\": \"$1\", \"setting\": \"$2\", \"purpose\": \"report: per_clip re-measurement of the text-use probe ECHO-G vs SCMG (protocol fixed before the run; replaces the shared-noise 11-05 SCMG numbers)\", \"host\": \"$(hostname)\"}" >> $O/TEST_USAGE.jsonl; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) ckpt $CK"
use echog "own notext swap x seeds 0 1 2 (official sampler)"
use scmg "czm6k offline, own notext swap x seeds 0 1 2, --seed-mode per_clip"
(  for sd in 0 1 2; do
     CUDA_VISIBLE_DEVICES=${EGPU:-0} PYTHONPATH=$W/official/src $PY scripts/echog_g1/echog_textuse.py --checkpoint $W/weights/best.pt \
       --config $W/weights/configs/sgdit_audio_text.yaml --cond-dir $D/condition_30fps --meta $D --stems $ST --swap-map $MAP \
       --out $O/echog --modes own notext swap --seed $sd >> $O/echog.log 2>&1 || log "ECHOG_FAILED s$sd"
   done; log "ECHOG_SAMPLED" ) &
(  for sd in 0 1 2; do for spec in "own:" "notext:--modality audio" "swap:--text-swap --text-swap-map $MAP"; do
     lab=${spec%%:*}; ex=${spec#*:}; P=$O/scmg/${lab}_seed_$(printf %03d $sd); mkdir -p $P
     CUDA_VISIBLE_DEVICES=${SGPU:-0} SCMG_J=39 SCMG_NO_GLITCH=1 $PY scripts/echog_g1/infer_test.py --ckpt $CK --vae $DEC \
       --stats $C/scmg/STATS_echog_g1_v1.json --view $C/echog-g1-v1/test --text-cache $C/scmg/text-e5-echog-g1-v1-test --out $P \
       --stems $ST --seed $sd --seed-mode per_clip --steps 10 --cfg-w 2.5 --protocol offline $ex >> $O/scmg.log 2>&1 || log "SCMG_FAILED $lab s$sd"
   done; done; log "SCMG_SAMPLED" ) &
wait
specs=()
for m in echog scmg; do for r in own notext swap; do for sd in 0 1 2; do
  d=$O/$m/${r}_seed_$(printf %03d $sd); [ -d $d ] || continue
  f=$O/benchmark_${m}_${r}_seed_$(printf %03d $sd).json
  [ -f $f ] || { OMP_NUM_THREADS=${EVAL_OMP:-4} bash scripts/echog_g1/bench.sh eval $d $ST ${m}_${r}_s$sd $f >> $O/eval.log 2>&1 || log "EVAL_FAILED $m $r s$sd"; }
  specs+=($m-$r-pc-s$sd=$d)
done; done; done
S=$O/sem; mkdir -p $S
for rec in $SB/rec-sem-v1 $SB/rec-sem-v1-seed1 $SB/rec-sem-v1-seed2; do
  rn=$(basename $rec)
  [ -s $S/$rn--gt--test.json ] || CUDA_VISIBLE_DEVICES=${SGPU:-0} $PY scripts/semantic/g1_recognizer.py score --data $SB/data-sem --ckpt $rec/model.pt \
    --name gt --split test --allow-missing --stems $ST --out $S/$rn--gt--test.json --save-probs >> $O/sem.log 2>&1
  for spec in "${specs[@]}"; do lab=${spec%%=*}; d=${spec#*=}; o=$S/$rn--$lab--test.json; [ -s $o ] && continue
    CUDA_VISIBLE_DEVICES=${SGPU:-0} $PY scripts/semantic/g1_recognizer.py score --data $SB/data-sem --ckpt $rec/model.pt --name $lab \
      --pt-dir $d --split test --allow-missing --out $o --save-probs >> $O/sem.log 2>&1 || log "SEM_FAILED $rn $lab"
  done
done
# recogniser file prefix: rec-sem-v1 / rec-sem-v1-seed1 / -seed2 -> rename to a common prefix for sem_multiseed
for f in $S/rec-sem-v1--*; do ln -sf $(basename $f) $S/recv1-seed0--${f#$S/rec-sem-v1--}; done
for k in 1 2; do for f in $S/rec-sem-v1-seed$k--*; do ln -sf $(basename $f) $S/recv1-seed$k--${f#$S/rec-sem-v1-seed$k--}; done; done
$PY scripts/semantic/sem_multiseed.py --scores $S --rec-prefix recv1-seed --seeds 0 1 2 --split test --out $O/MULTISEED.json \
  --sources echog-own-pc echog-notext-pc echog-swap-pc scmg-own-pc scmg-notext-pc scmg-swap-pc \
  --pairs scmg-own-pc:scmg-notext-pc,scmg-own-pc:scmg-swap-pc,echog-own-pc:echog-notext-pc,echog-own-pc:echog-swap-pc,scmg-own-pc:echog-own-pc >> $O/sem.log 2>&1 && log "MULTISEED_DONE"
for sd in 0 1 2; do s1=$(( (sd + 1) % 3 )); $PY scripts/echog_g1/text_sensitivity.py --out $O/TEXT_SENS_s$sd.json \
  echog_swap=$O/echog/own_seed_00$sd,$O/echog/swap_seed_00$sd,$O/echog/own_seed_00$s1 echog_notext=$O/echog/own_seed_00$sd,$O/echog/notext_seed_00$sd,$O/echog/own_seed_00$s1 \
  scmg_swap=$O/scmg/own_seed_00$sd,$O/scmg/swap_seed_00$sd,$O/scmg/own_seed_00$s1 scmg_notext=$O/scmg/own_seed_00$sd,$O/scmg/notext_seed_00$sd,$O/scmg/own_seed_00$s1 \
  >> $O/textuse.log 2>&1; done; log "TEXTUSE_PC_DONE"
