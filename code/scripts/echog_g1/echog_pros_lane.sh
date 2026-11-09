#!/bin/bash
# 11-07 §7.55 item 2 on ECHO-G (criteria: docs/criteria/SEMLINE_7_55.md, item 2 ECHO-G section, written before this run): the official
# G1 checkpoint fine-tuned 2000 steps with its Wav2Vec2 audio REPLACED by the 8 prosody dims of the SCMG view (echog_semt.py --audio
# prosody; new audio projection starts at the official null-audio output), same recipe / data / evaluation as echog_semt_lane.sh;
# compared with the egctrl lane of that run (its score files are linked once that run is done, not recomputed).
#   echog_pros_lane.sh <out dir> <egsemt dir (finished)>       env PY, W
set -uo pipefail
O=$1; E=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; W=${W:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official}
R=$B/echog-stream-1105/data_root; A=$B/echog-g1-hos-assets; ST=$A/dev_stems.txt; G=$B/semantic-baseline-1105/g1
D4=$B/baseline-repro-1104/echog/data_root; V=$B/views/echog-g1-v1-hos
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
export PY EG=$D4 OFF=$W/official REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$ST MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH=$W/official/src
CFG=$W/official/configs/sgdit_audio_text.yaml; INIT=$W/weights/best.pt; MAP=$E/dev_swap_map.json
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/echog_g1/echog_semt.py test --init $INIT --config $CFG --data-root $R --view $V --tmp $O/test-tmp > $O/test.log 2>&1
grep -q ALL_OK $O/test.log && log "TEST_ALL_OK" || { log "TEST_FAILED"; exit 1; }
n=egpros; mkdir -p $O/$n
[ -f $O/$n/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY scripts/echog_g1/echog_semt.py train --init $INIT --config $CFG --data-root $R \
  --out $O/$n/train --steps 2000 --batch 12 --accum 4 --workers 8 --seed 0 --audio prosody --view $V > $O/$n/train.log 2>&1 || { log "TRAIN_FAILED"; exit 1; }
log "TRAINED $(cat $O/$n/train/DONE.json)"
for sd in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$sd $PY scripts/echog_g1/echog_semt.py sample --ckpt $O/$n/train/final.pt --data-root $R --stems $ST \
    --out $O/$n --modes own notext swap --swap-map $MAP --seed $sd >> $O/$n/sample_s$sd.log 2>&1 || log "SAMPLE_FAILED s$sd" &
done; wait; log "SAMPLED"
for r in own notext swap; do for sd in 0 1 2; do d=$O/$n/${r}_seed_00$sd; f=$O/benchmark_${n}_${r}_seed_00$sd.json
  [ -f $f ] || echo "bash scripts/echog_g1/bench.sh eval $d $ST ${n}_${r}_s$sd $f >> $O/eval.log 2>&1 || echo EVAL_FAILED $r s$sd >> $O/lane.log"
done; done | xargs -P 4 -I{} bash -c {}
log "EVAL_DONE"
specs=(); for r in own notext swap; do for sd in 0 1 2; do specs+=($n-$r-pc-s$sd=$O/$n/${r}_seed_00$sd); done; done
until grep -q LANE_DONE $E/lane.log; do sleep 120; done; log "EGSEMT_DONE"     # the control's score files exist from here
S=$O/sem; mkdir -p $S; ln -sf $E/sem/*--egctrl-* $S/ 2>/dev/null
i=0; for rec in $G/rec-sem-hos-seed0 $G/rec-sem-hos-seed1 $G/rec-sem-hos-seed2; do
  ( PY=$PY bash scripts/semantic/sem_score_dirs.sh $i $G/data-sem-hos "$rec" $S dev "${specs[@]}" >> $O/sem.log 2>&1 ) & i=$((i + 1)); done; wait
$PY scripts/semantic/sem_multiseed.py --scores $S --rec-prefix rec-sem-hos-seed --seeds 0 1 2 --split dev --out $O/MULTISEED.json \
  --sources egctrl-own-pc egctrl-notext-pc egctrl-swap-pc egpros-own-pc egpros-notext-pc egpros-swap-pc \
  --pairs egpros-own-pc:egctrl-own-pc,egpros-own-pc:egpros-notext-pc,egctrl-own-pc:egctrl-notext-pc >> $O/sem.log 2>&1 && log "MULTISEED_DONE"
for sd in 0 1 2; do s1=$(( (sd + 1) % 3 )); $PY scripts/echog_g1/text_sensitivity.py --out $O/TEXT_SENS_s$sd.json \
  egpros_swap=$O/$n/own_seed_00$sd,$O/$n/swap_seed_00$sd,$O/$n/own_seed_00$s1 egpros_notext=$O/$n/own_seed_00$sd,$O/$n/notext_seed_00$sd,$O/$n/own_seed_00$s1 \
  egctrl_swap=$E/egctrl/own_seed_00$sd,$E/egctrl/swap_seed_00$sd,$E/egctrl/own_seed_00$s1 >> $O/lane.log 2>&1; done
log "LANE_DONE"
