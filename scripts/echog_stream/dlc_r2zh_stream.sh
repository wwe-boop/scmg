#!/bin/bash
# One DLC pod (11-06): build full-data-r2 (= full-data-r1 + Streamer zh with FunASR char timings, view data-v36_2-streamer-text)
# with the frozen r3 package's prepare_full.py, then fine-tune the EX002 streaming ECHO-G (echog-stream-weights-20261106/
# ex002/final.pt, its own normalisation via --init-stats) on it with prefix-audio training, and evaluate in the same job
# (ex002_stream.sh rows stream_prefix / stream_ramp / stream_frozen x seeds 0 1 + snapshots, EX002 FGD v2).
set -euo pipefail
: "${DLC_JOB_ID:?real DLC job required}"
REL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)   # release root
P=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/reference/dlc-package-r3
E=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog
W=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/echog-official
VIEW=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views/data-v36_2-streamer-text
NG=$(nvidia-smi -L | wc -l)
printf '%s\n' "0f67be85c526451bc7d17d4cfefb44357781fc50789c3aec268c024706f471a5  $P/FILES.sha256" | sha256sum -c -
(cd "$P" && sha256sum -c FILES.sha256 --quiet)
test -s "$REL/GIT_COMMIT" || { echo "GIT_COMMIT missing under $REL"; exit 4; }
# Keep numba/librosa caches on the pod's local disk. The shared env lives on CPFS
# (USERS/zehan/envs/echog); r2 (dlc140g6izut8i1j) hit OSError 116 stale file handle
# when many condition workers concurrently read numba's cache index while importing librosa.
export NUMBA_CACHE_DIR=/tmp/numba-$DLC_JOB_ID
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) R2ZH_START gpus=$NG NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR"

# ---- 1 full-data-r2 (skipped when already complete)
if [ ! -f "$W/full-data-r2/FULL_DATA_COMPLETE.json" ]; then
  ( export PATH="$E/bin:$PATH" PYTHONPATH="$P/official/src:$P/code" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
    export XDG_CACHE_HOME=$W/dlc-cache/$DLC_JOB_ID HF_HOME=$W/dlc-cache/$DLC_JOB_ID/hf TORCH_HOME=$W/dlc-cache/$DLC_JOB_ID/torch
    export TMPDIR=$XDG_CACHE_HOME/tmp OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=disable
    export NUMBA_CACHE_DIR=/tmp/numba-$DLC_JOB_ID
    mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR"
    python "$P/code/gate_check.py" --package "$P"
    python "$P/code/prepare_full.py" --output "$W/full-data-r2" --view "$VIEW" --code "$P/code" --workers "${PREP_WORKERS:-6}" \
      --gpus "$NG" --per-gpu "${PREP_PER_GPU:-6}" --claim-round "$DLC_JOB_ID" )
fi
test -f "$W/full-data-r2/FULL_DATA_COMPLETE.json"
echo "$(date -Is) DATA_DONE"

# ---- 2 streaming fine-tune + evaluation (inherits NUMBA_CACHE_DIR)
cd "$REL"
GL=$(seq -s ' ' 0 $((NG - 1)))
TAG=r2zh INIT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-weights-20261106/ex002/final.pt DATA="$W/full-data-r2" \
  ROWS="stream_prefix stream_ramp stream_frozen" GPUS="$GL" SPG=${SPG:-3} \
  TRAIN_ARGS="--init-stats --prefix-audio /mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/models/wav2vec2-large-xlsr-53-english --prefix-audio-mode batch16 --wav-dir $W/full-data-r2/raw_inputs.jsonl" \
  bash scripts/echog_stream/ex002_stream.sh
echo "$(date -Is) R2ZH_DONE"
