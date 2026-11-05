#!/bin/bash
# DLC entry: G1 main-table efficiency on ONE dedicated A800 (1 Pod x 1 GPU), every method in sequence, batch 1.
#   dlc_entry.sh check   path / env / import check (any host, no GPU work)
#   dlc_entry.sh full    real DLC job only (DLC_JOB_ID set, hostname not dsw*): all methods + memory breakdowns
# Outputs: $OUT_ROOT/$DLC_JOB_ID/ (per-method JSON + logs, MONITOR.log = nvidia-smi + /proc/loadavg every 15 s).
set -uo pipefail
PHASE=${1:?phase}
P=$(cd "$(dirname "$0")" && pwd)
cd $P && sha256sum -c --quiet FILES.sha256 || { echo "FILES.sha256 mismatch"; exit 2; }
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; L=$B/baselines-g1-1105/latency
OUT_ROOT=${OUT_ROOT:-$L/dlc-runs}
ECHOG=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error
need() { for f in "$@"; do [ -e "$f" ] || { echo "MISSING $f"; exit 3; }; done; }
need $ECHOG $L/envs/{causal,semtalk,rg,gmr}/bin/python $L/snap/scmg_causal/scmg/causal_feat.py $L/snap/campplus/campplus.onnx \
     $L/snap/echog_stream/echog_stream/stream_core.py $B/baselines-g1-1105/semtalk/root/weights/weights/best_semtalk_sparse.bin \
     $B/baselines-g1-1105/assets/gmr_weights/emage_audio $B/baselines-g1-1105/assets/gmr_weights/gesturelsm/shortcut_reflow.bin \
     $B/baselines-g1-1105/assets/emage_smplx_seed0 $B/baselines-g1-1105/semtalk/run/smplx_seed0 \
     $B/echog-g1-sf-1105/test/sf0/final.pt $B/echog-stream-1105/g1-confirm/train/final.pt \
     /mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official/models/Qwen3.5-4B /mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture/weights/model.pt
case $PHASE in
  check)
    for e in causal semtalk rg gmr; do CUDA_VISIBLE_DEVICES="" $L/envs/$e/bin/python -c "import torch, numpy; print('$e', torch.__version__)" || exit 4; done
    CUDA_VISIBLE_DEVICES="" $L/envs/gmr/bin/python -c "import mink, smplx, mujoco" && CUDA_VISIBLE_DEVICES="" $L/envs/semtalk/bin/python -c "import clip, funasr" \
      && CUDA_VISIBLE_DEVICES="" $L/envs/causal/bin/python -c "import onnxruntime, torchaudio" && echo CHECK_OK ;;
  full)
    : "${DLC_JOB_ID:?real DLC job required}"
    case $(hostname) in dsw*) echo "refusing to run the full phase on DSW"; exit 78;; esac
    O=$OUT_ROOT/$DLC_JOB_ID; mkdir -p $O
    { echo "host $(hostname) nproc $(nproc) job $DLC_JOB_ID"; nvidia-smi; cat /proc/cpuinfo | grep "model name" | head -1; } > $O/HOST.txt 2>&1
    ( while true; do echo "$(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | paste -sd';')"; sleep 15; done ) > $O/MONITOR.log 2>&1 &
    MON=$!
    GPU=0 bash $P/run_latency.sh $O scmg_offline scmg_psync scmg_engine echog echog_stream echog_stream_nocudnn robogesture semtalk emage lsm \
      gmr_semtalk gmr_lsm gmr_emage
    export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
    $ECHOG $P/lat_echog_mem.py --out $O/mem > $O/logs/echog_mem.log 2>&1 && echo "$(date -Is) DONE echog_mem" >> $O/run.log
    for v in default nocudnn; do $ECHOG $P/w2v_prefix_mem.py --variant $v --out $O/mem > $O/logs/w2v_$v.log 2>&1; done && echo "$(date -Is) DONE w2v_prefix_mem" >> $O/run.log
    for p in offline engine; do SCMG_J=39 $ECHOG $P/lat_kernel_time.py --protocol $p --n 6 --out $O/kernel > $O/logs/kernel_$p.log 2>&1; done && echo "$(date -Is) DONE kernel_time" >> $O/run.log
    $ECHOG $P/lat_table.py $O > $O/TABLE.md 2>$O/logs/table.err
    kill $MON; nvidia-smi >> $O/HOST.txt 2>&1
    echo "$(date -Is) JOB_DONE" >> $O/run.log ;;
  *) echo "phase $PHASE"; exit 2 ;;
esac
