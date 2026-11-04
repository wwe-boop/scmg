#!/bin/bash
# ECHO-G official benchmark on val_common (or a stem subset), 11-04.
#   bench.sh scmg <ckpt> <vae> <pred dir> <offline|engine> <seed> [stem list]
#       SCMG inference (infer_test.py, sharded over GPUS) with the given protocol, then the official evaluator
#       -> <pred dir>/../benchmark_<basename pred dir>.json
#   bench.sh eval <pred dir> <stem list> <tag> <out.json>
#       official evaluator only (any method, e.g. the ECHO-G sliding-window predictions of echog_slide.py)
# offline = final text, H = inf (the ECHO-G benchmark setting); engine = EX002 engine protocol (x2 word-by-word early text,
# 15 tok/s, lag 0.15 s, causal e5, deploy timing, H = 160 ms). CFG fixed a priori at the r4 value (CFGW=2.5), 10 steps.
# Paths (defaults = this 5090 host; override on DSW/DLC): EG (ECHO-G root with ds/ and official-meta/ or REF_DIR/WAV_DIR/
# SPLIT/MMAE/FGD_CKPT), OFF (official code), VIEW_TEST TEXT_TEST TEXT_CAUSAL_TEST STATS (G1 view assets).
set -euo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg
EG=${EG:-/mnt/bulk/zehan/echog-dataset}; OFF=${OFF:-$HOME/bulk/echog-official-ro/official}
REF_DIR=${REF_DIR:-$EG/ds/motion_39d_30fps}; WAV_DIR=${WAV_DIR:-$EG/ds/audio}; SPLIT=${SPLIT:-$EG/official-meta/splits/val_common.txt}
MMAE=${MMAE:-$EG/official-meta/eval_assets/mmae/g1_mmae_30body_30fps.npy}; FGD_CKPT=${FGD_CKPT:-$EG/weights-ro/evaluation/g1_fgd_encoder.pt}
VIEW_TEST=${VIEW_TEST:-$C/echog-g1-v1/test}; STATS=${STATS:-$C/scmg/STATS_echog_g1_v1.json}
TEXT_TEST=${TEXT_TEST:-$C/scmg/text-e5-echog-g1-v1-test}; TEXT_CAUSAL_TEST=${TEXT_CAUSAL_TEST:-$B/models/text-e5-causal-echog-g1-v1-test}
export SCMG_J=39 SCMG_NO_GLITCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} SCMG_LOCAL_MIRROR=${SCMG_LOCAL_MIRROR:-$HOME/bulk/cpfs-mirror}
unset SCMG_TEXT_CAUSAL                                    # infer_test.py takes the causal cache only from --text-causal
ev() {  # <pred dir> <stem list> <tag> <out>
  PYTHONPATH=$OFF/src $PY $OFF/scripts/eval_g1_motion_cls.py --pred-dir $1 --ref-dir $REF_DIR --val-split $2 --wav-dir $WAV_DIR \
    --mmae-file $MMAE --g1-ae-ckpt $FGD_CKPT --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems \
    --tag $3 --out $4
}
case $1 in
  scmg)
    ck=$2; vae=$3; P=$4; proto=$5; seed=$6; stems=${7:-$SPLIT}
    case $proto in offline) PA=(--protocol offline);; engine) PA=(--protocol engine --text-causal $TEXT_CAUSAL_TEST);; *) echo "protocol $proto"; exit 2;; esac
    GL=(${GPUS:-0}); n=${#GL[@]}; mkdir -p $P
    for i in $(seq 0 $((n - 1))); do
      CUDA_VISIBLE_DEVICES=${GL[$i]} $PY scripts/echog_g1/infer_test.py --ckpt $ck --vae $vae --stats $STATS --view $VIEW_TEST \
        --text-cache $TEXT_TEST --out $P --stems $stems --seed $seed --steps ${STEPS:-10} --cfg-w ${CFGW:-2.5} --split ${VSPLIT:-test} --shard $i/$n "${PA[@]}" &
    done; wait
    ev $P $stems scmg_${proto}_seed$seed $(dirname $P)/benchmark_$(basename $P).json;;
  eval) ev $2 $3 $4 $5;;
  *) echo "usage: bench.sh scmg|eval ..."; exit 2;;
esac
