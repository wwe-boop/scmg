#!/bin/bash
# 11-07 pointing token on ECHO-G (criteria: docs/criteria/SEMLINE_7_55.md, "Pointing on ECHO-G", written before this run).
# egpt = the official G1 checkpoint fine-tuned 2000 steps exactly as egctrl / egsemt (train_drop minus the hos dev speakers) with the
# zero-initialised point-token path of echog_semt.py --ctl (ctlcf recipe ported: token dropout 0.5, counterfactual targets 0.5, ray
# loss w 3, module lr x10). Dev 400 clips, official sampler, seeds 0 / 1 / 2: egpt with tokens none / gt / shuffle (text own), egctrl
# (from the egsemt run) with none; official benchmark of egpt none / gt; g1_ctl_eval.py (STG etc.) per seed.
#   echog_point_lane.sh <out dir> <egsemt dir>       env PY, W
set -uo pipefail
O=$1; E=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; W=${W:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official}
R=$B/echog-stream-1105/data_root; A=$B/echog-g1-hos-assets; ST=$A/dev_stems.txt; HV=$B/views/echog-g1-v1-hos
CTL=$B/control-token-1105/ctl-tokens-hos; D4=$B/baseline-repro-1104/echog/data_root
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
export PY EG=$D4 OFF=$W/official REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$ST MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH=$W/official/src ECHOG_OFFICIAL_SRC=$W/official/src SCMG_J=39
CFG=$W/official/configs/sgdit_audio_text.yaml; INIT=$W/weights/best.pt
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/echog_g1/echog_semt.py test --init $INIT --config $CFG --data-root $R --ctl $CTL --tmp $O/test-tmp > $O/test.log 2>&1
grep -q ALL_OK $O/test.log && log "TEST_ALL_OK" || { log "TEST_FAILED"; exit 1; }
mkdir -p $O/egpt $O/egctrl
[ -f $O/egpt/train/DONE.json ] || CUDA_VISIBLE_DEVICES=0 $PY scripts/echog_g1/echog_semt.py train --init $INIT --config $CFG --data-root $R \
  --out $O/egpt/train --steps 2000 --batch 12 --accum 4 --workers 8 --seed 0 --ctl $CTL > $O/egpt/train.log 2>&1 || { log "TRAIN_FAILED"; exit 1; }
log "TRAINED $(cat $O/egpt/train/DONE.json)"
for sd in 0 1 2; do (
  CUDA_VISIBLE_DEVICES=$sd $PY scripts/echog_g1/echog_semt.py sample --ckpt $O/egpt/train/final.pt --data-root $R --stems $ST --out $O/egpt \
    --modes --ctl-dir $CTL --ctl-modes none gt shuffle --seed $sd >> $O/egpt/sample_s$sd.log 2>&1 || log "SAMPLE_FAILED egpt s$sd"
  CUDA_VISIBLE_DEVICES=$sd $PY scripts/echog_g1/echog_semt.py sample --ckpt $E/egctrl/train/final.pt --data-root $R --stems $ST --out $O/egctrl \
    --modes --ctl-dir $CTL --ctl-modes none --seed $sd >> $O/egctrl/sample_s$sd.log 2>&1 || log "SAMPLE_FAILED egctrl s$sd" ) &
done; wait; log "SAMPLED"
for m in none gt; do for sd in 0 1 2; do d=$O/egpt/${m}_seed_00$sd; f=$O/benchmark_egpt_${m}_seed_00$sd.json
  [ -f $f ] || echo "bash scripts/echog_g1/bench.sh eval $d $ST egpt_${m}_s$sd $f >> $O/eval.log 2>&1 || echo EVAL_FAILED $m s$sd >> $O/lane.log"
done; done | xargs -P 6 -I{} bash -c {}
log "EVAL_DONE"
for sd in 0 1 2; do
  $PY scripts/semantic/g1_ctl_eval.py --ctl $CTL --view $HV --out $O/CTL_EVAL_s$sd.json gt=- egpt_none=$O/egpt/none_seed_00$sd \
    egpt_gt=$O/egpt/gt_seed_00$sd egpt_shuffle=$O/egpt/shuffle_seed_00$sd egctrl_none=$O/egctrl/none_seed_00$sd >> $O/ctl_eval.log 2>&1 \
    || log "CTL_EVAL_FAILED s$sd" &
done; wait
log "LANE_DONE"
