#!/bin/bash
# 11-07 item 4: token rows of the control-token gain table on dev 400 (inference only, no training). One model, modes x inference
# seeds 0 / 1 / 2 (per_clip, offline, CFG 2.5, 10 steps), official benchmark per run, recognition with rec-sem-hos-seed0..2 (scores
# copied next to the other per_clip scores for sem_multiseed.py). GPU k = seed k; the modes of a seed run concurrently.
#   ctl_oracle_g1.sh <out dir> <ckpt> <ctl dir> <label>=<ctl mode> ...      env PY
set -uo pipefail
O=$1; CK=$2; CTL=$3; shift 3; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt; A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos; G=$B/semantic-baseline-1105/g1
PY=${PY:-python}; export SCMG_J=39 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official; D4=$B/baseline-repro-1104/echog/data_root
export EG=$D4 OFF=$W/official MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt
export REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$HV VSPLIT=val TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 \
  TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 STATS=$C/scmg/STATS_echog_g1_v1.json CFGW=2.5 BENCH_STEPS=10
log() { echo "$(date -Is) $*" >> $O/lane.log; }
log "START commit $(cat GIT_COMMIT 2>/dev/null || git rev-parse HEAD 2>/dev/null) ckpt $CK"
for sd in 0 1 2; do (
  for spec in "$@"; do lab=${spec%%=*}; m=${spec#*=}; P=$O/$lab/seed_00$sd
    [ -f $O/$lab/benchmark_seed_00$sd.json ] || PY=$PY GPUS=$sd INFER_EXTRA="--save50 --seed-mode per_clip --ctl-tokens $CTL --ctl-mode $m" \
      bash scripts/echog_g1/bench.sh scmg $CK $DEC $P offline $sd $A/dev_stems.txt >> $O/bench_$lab.log 2>&1 || log "BENCH_FAILED $lab s$sd" &
  done; wait; log "SEED_DONE s$sd" ) &
done; wait
sp=(); for spec in "$@"; do lab=${spec%%=*}; for sd in 0 1 2; do sp+=($lab-pc-s$sd=$O/$lab/seed_00$sd); done; done
i=0; for rec in $G/rec-sem-hos-seed0 $G/rec-sem-hos-seed1 $G/rec-sem-hos-seed2; do
  ( PY=$PY PAIRS= bash scripts/semantic/sem_score_dirs.sh $i $G/data-sem-hos "$rec" $O/sem dev "${sp[@]}" >> $O/sem.log 2>&1 ) & i=$((i + 1)); done; wait
log "LANE_DONE"
