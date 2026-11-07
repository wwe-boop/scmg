#!/bin/bash
# 11-07 §7.55 item 1, ECHO-G side (criteria: docs/criteria/SEMLINE_7_55.md, ECHO-G section, written before this run).
# One pod, 2 GPUs: GPU0 = control (official G1 checkpoint fine-tuned 2000 steps, no module), GPU1 = + SemTimeEG (echog_semt.py);
# then each model samples the held-out-speaker dev set (400 clips) with the official sampler (8 steps, guidance 1, per-clip noise)
# in text modes own / notext / swap (dev swap map = the SCMG infer_test.py derangement) x inference seeds 0 / 1 / 2; official
# benchmark per run, recognition with rec-sem-hos-seed0..2 (hos train split only), 3-seed pooled pair_win (sem_multiseed.py,
# recording-clustered bootstrap), text sensitivity per seed.  Dev only: no test-side data are read.
#   echog_semt_lane.sh <out dir> <dev swap map>       env PY (echog env), W (echog-official root with official/ and weights/)
set -uo pipefail
O=$1; MAP=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; W=${W:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official}
R=$B/echog-stream-1105/data_root; A=$B/echog-g1-hos-assets; ST=$A/dev_stems.txt; G=$B/semantic-baseline-1105/g1
D4=$B/baseline-repro-1104/echog/data_root
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
export PY EG=$D4 OFF=$W/official REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$ST MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH=$W/official/src
CFG=$W/official/configs/sgdit_audio_text.yaml; INIT=$W/weights/best.pt
cp $MAP $O/dev_swap_map.json; MAP=$O/dev_swap_map.json
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) init $INIT"
CUDA_VISIBLE_DEVICES=0 $PY scripts/echog_g1/echog_semt.py test --init $INIT --config $CFG --data-root $R --tmp $O/test-tmp > $O/test.log 2>&1
grep -q ALL_OK $O/test.log && log "TEST_ALL_OK" || { log "TEST_FAILED"; exit 1; }
lane() {  # gpu name extra
  local g=$1 n=$2; shift 2
  [ -f $O/$n/train/DONE.json ] || CUDA_VISIBLE_DEVICES=$g $PY scripts/echog_g1/echog_semt.py train --init $INIT --config $CFG --data-root $R \
    --out $O/$n/train --steps ${STEPS_FT:-2000} --batch 12 --accum 4 --workers 8 --seed 0 "$@" > $O/$n/train.log 2>&1 || { log "TRAIN_FAILED $n"; return 1; }
  log "TRAINED $n $(cat $O/$n/train/DONE.json)"
  for sd in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/echog_g1/echog_semt.py sample --ckpt $O/$n/train/final.pt --data-root $R --stems $ST \
      --out $O/$n --modes own notext swap --swap-map $MAP --seed $sd >> $O/$n/sample.log 2>&1 || log "SAMPLE_FAILED $n s$sd"
  done; log "SAMPLED $n"
}
mkdir -p $O/egctrl $O/egsemt
lane 0 egctrl & lane 1 egsemt --sem-time & wait
specs=()
for m in egctrl egsemt; do for r in own notext swap; do for sd in 0 1 2; do
  d=$O/$m/${r}_seed_$(printf %03d $sd); [ -d $d ] || continue
  f=$O/benchmark_${m}_${r}_seed_$(printf %03d $sd).json
  [ -f $f ] || echo "bash scripts/echog_g1/bench.sh eval $d $ST ${m}_${r}_s$sd $f >> $O/eval.log 2>&1 || echo EVAL_FAILED $m $r s$sd >> $O/lane.log"
done; done; done | xargs -P 4 -I{} bash -c {}
for m in egctrl egsemt; do for r in own notext swap; do for sd in 0 1 2; do
  d=$O/$m/${r}_seed_$(printf %03d $sd); [ -d $d ] && specs+=($m-$r-pc-s$sd=$d); done; done; done
log "EVAL_DONE"
S=$O/sem; GPUQ=(0 1); i=0
for rec in $G/rec-sem-hos-seed0 $G/rec-sem-hos-seed1 $G/rec-sem-hos-seed2; do
  ( PY=$PY bash scripts/semantic/sem_score_dirs.sh ${GPUQ[$((i % 2))]} $G/data-sem-hos "$rec" $S dev gt=- "${specs[@]}" >> $O/sem.log 2>&1 ) &
  i=$((i + 1)); [ $((i % 2)) = 0 ] && wait
done; wait
$PY scripts/semantic/sem_multiseed.py --scores $S --rec-prefix rec-sem-hos-seed --seeds 0 1 2 --split dev --out $O/MULTISEED.json \
  --sources egctrl-own-pc egctrl-notext-pc egctrl-swap-pc egsemt-own-pc egsemt-notext-pc egsemt-swap-pc \
  --pairs egsemt-own-pc:egctrl-own-pc,egsemt-own-pc:egsemt-notext-pc,egctrl-own-pc:egctrl-notext-pc,egsemt-own-pc:egsemt-swap-pc,egctrl-own-pc:egctrl-swap-pc \
  >> $O/sem.log 2>&1 && log "MULTISEED_DONE"
for sd in 0 1 2; do s1=$(( (sd + 1) % 3 )); a=(); for m in egctrl egsemt; do
  a+=(${m}_swap=$O/$m/own_seed_00$sd,$O/$m/swap_seed_00$sd,$O/$m/own_seed_00$s1 ${m}_notext=$O/$m/own_seed_00$sd,$O/$m/notext_seed_00$sd,$O/$m/own_seed_00$s1)
  done; $PY scripts/echog_g1/text_sensitivity.py --out $O/TEXT_SENS_s$sd.json "${a[@]}" >> $O/lane.log 2>&1; done
log "LANE_DONE"
