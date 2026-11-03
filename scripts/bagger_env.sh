#!/bin/bash
# Source inside a taskset/nice constrained tmux job. No process management.
export PY=${PY:-/workspace/zehan/env/tts_drift/bin/python}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$CODE:/workspace/zehan/env/scmg-deps${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_ENDPOINT=https://hf-mirror.com
export SCMG_TEXT_CAUSAL=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/text-e5-causal-v36_1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BAGGER_MEMORY_FRACTION=${BAGGER_MEMORY_FRACTION:-.48}
cd "$CODE"
