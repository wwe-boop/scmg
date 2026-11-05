#!/bin/bash
# Official ECHO-G G1 evaluator (same flags as scripts/echog_g1/bench.sh `ev`), CPU only.
#   eval_official.sh <pred dir> <stem list> <tag> <out.json>
# Paths default to the 5090-host read-only mirrors; override EG / OFF / PY on DSW or DLC
# (CPFS: EG=.../baseline-repro-1104/echog/data_root layout differs -> set REF_DIR WAV_DIR MMAE FGD_CKPT explicitly).
set -euo pipefail
PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}
EG=${EG:-/mnt/bulk/zehan/echog-dataset}; OFF=${OFF:-$HOME/bulk/echog-official-ro/official}
REF_DIR=${REF_DIR:-$EG/ds/motion_39d_30fps}; WAV_DIR=${WAV_DIR:-$EG/ds/audio}
MMAE=${MMAE:-$EG/official-meta/eval_assets/mmae/g1_mmae_30body_30fps.npy}; FGD_CKPT=${FGD_CKPT:-$EG/weights-ro/evaluation/g1_fgd_encoder.pt}
mkdir -p "$(dirname "$4")"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=$OFF/src nice -n 10 $PY $OFF/scripts/eval_g1_motion_cls.py --pred-dir "$1" --ref-dir $REF_DIR --val-split "$2" \
  --wav-dir $WAV_DIR --mmae-file $MMAE --g1-ae-ckpt $FGD_CKPT --fps 30 --ba-direction audio_to_motion \
  --enable-foot-metrics --require-all-stems --tag "$3" --out "$4"
