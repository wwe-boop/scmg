#!/bin/bash
# 11-07 item 5 on DLC (3 GPUs, seed k on GPU k): lib_g1.sh for sf0 (RePaint insertion) and for the mask-conditioned model (item 5b)
# in parallel, CPFS paths. CBF-QP + executability run afterwards on a CPU host (SAFE unset here).
#   lib_dlc.sh <out root> <mask-cond ckpt>      env PY, LANES (default 'sf0 mc'), GPU_PER_SEED (default 1; 0 = all on GPU)
set -uo pipefail
O=$1; MC=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; L=$B/control-token-1105/lib-assets
W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official; D4=$B/baseline-repro-1104/echog/data_root; A=$B/echog-g1-hos-assets
export ASSETS=$A DEC=$B/echog-g1-decft-1105/decft-w3d0m05/final.pt LIB=$L/gesture-lib-K CTLC=$B/control-token-1105/ctl-tokens-hos-count \
  PRED=$L/llm_PRED.json VIEW=$B/views/echog-g1-v1-hos REC_DATA=$L/data-sem-hos-R REC_GLOB="$L/rec-sem-hosR-seed*" GPU_PER_SEED=${GPU_PER_SEED:-1} GPU=${GPU:-0} PAR=${PAR:-6} \
  EG=$D4 OFF=$W/official MMAE=$D4/eval_assets/mmae/g1_mmae_30body_30fps.npy FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt \
  REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$B/views/echog-g1-v1-hos TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 \
  TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 STATS=$C/scmg/STATS_echog_g1_v1.json OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
for ln in ${LANES:-sf0 mc}; do
  [ $ln = sf0 ] && ( CK=$B/echog-g1-sf-1105/dev/sf0/final.pt bash scripts/semantic/lib_g1.sh $O/sf0 > $O/sf0.log 2>&1 ) &
  [ $ln = mc ] && ( CK=$MC bash scripts/semantic/lib_g1.sh $O/mc > $O/mc.log 2>&1 ) &
done
wait; echo "$(date -Is) LIB_DLC_DONE" >> $O/lane.log
