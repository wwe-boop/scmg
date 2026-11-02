#!/bin/bash
# DLC entry of the SCMG training package.   dlc_entry.sh full | selftest
#   full      real DLC job only (DLC_JOB_ID set, hostname not dsw*), 8 GPUs, outputs under OUT
#   selftest  DSW self-test with an explicit run id (SCMG_DSW_RUNID): tiny step counts, 1 GPU, mmap
set -euo pipefail
PHASE=${1:?phase}
P=$(cd "$(dirname "$0")/.." && pwd)                       # package root (contains scmg/ scripts/ FILES.sha256)
VAE=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/trained/vae-r3/final.pt
cd $P && sha256sum -c --quiet FILES.sha256
[ "$PHASE" = full25 ] || { echo "$(cat $P/VAE_SHA256)  $VAE" | sha256sum -c -; }
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
# prefix-causal e5 features for ADSC token streaming (prep_text --causal; 11-02 review fix)
TC=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models/text-e5-causal-v36_1
[ -f $TC/TEXT_FEAT.npy ] && export SCMG_TEXT_CAUSAL=$TC
case $PHASE in
  full|full25|fullvox|adsc|adscvox|voxabl|voxabl2|textuse|voxq)
    : "${DLC_JOB_ID:?real DLC job required}"
    case $(hostname) in dsw*) echo "refusing to run the full phase on DSW"; exit 78;; esac
    SUF=""
    [ "$PHASE" = adsc ] && export ONLY_ADSC=1        # add-on: ADSC lanes on an already running main pipeline (same OUT)
    [ "$PHASE" = adscvox ] && export ONLY_ADSC=1
    if [ "$PHASE" = voxq ]; then export VOXQ=1 TOKC=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1; SUF=-voxq; fi
    if [ "$PHASE" = textuse ]; then export TEXTUSE=1 TOKC=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1; SUF=-textuse; fi
    if [ "$PHASE" = voxabl2 ]; then export VOXABL2=1 TOKC=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1; SUF=-voxabl2; fi
    if [ "$PHASE" = voxabl ]; then                     # VoxCPM module ablation, own output root
      export VOXABL=1 TOKC=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1; SUF=-voxabl
    fi
    if [ "$PHASE" = fullvox ] || [ "$PHASE" = adscvox ]; then                    # VoxCPM-style planner (TSLM Qwen2.5-0.5B + FSQ + RALM + LocEnc)
      SUF=-vox; export TOKC=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1 BASECFG='{"planner": "voxcpm"}'
    fi
    if [ "$PHASE" = full25 ]; then                     # 25 Hz motion latents ablation: own VAE, own output root
      export SCMG_STRIDE=2; SUF=-25hz
      VAE=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/dlc-$(cat $P/RUN_NAME)$SUF/vae25/final.pt
    fi
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
      export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
    fi
    RUN=$(cat $P/RUN_NAME)
    [ "${ONLY_ADSC:-0}" = 1 ] && RUN=$(cat $P/ADSC_TARGET)      # add-on lanes write into the running pipeline's root
    OUT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/dlc-$RUN$SUF
    mkdir -p $OUT; echo "$DLC_JOB_ID $(hostname) $(date -Is)" >> $OUT/JOBS.txt
    NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l) PY=/usr/local/bin/python \
      exec bash $P/scripts/pipeline_dlc.sh $P $OUT $VAE ;;
  selftest)
    : "${SCMG_DSW_RUNID:?explicit DSW run id required}"
    OUT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/selftest/$SCMG_DSW_RUNID; rm -rf $OUT   # CPFS, never the shared /tmp
    export RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT=29500   # what DLC sets on every pod
    [ "${SCMG_STRIDE:-4}" = 2 ] && VAE=$OUT/vae25/final.pt
    [ "${SCMG_VOX:-0}" = 1 ] && export TOKC=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1 BASECFG='{"planner": "voxcpm", "d": 256}'
    NGPU=1 SCALE=0.002 SNAP=40000 NEVAL=1 NFULL=1 NMID=1 BATCH=8 WORKERS=2 EXTRA=--mmap PY=${PY:-/usr/local/bin/python} \
      bash $P/scripts/pipeline_dlc.sh $P $OUT $VAE
    test -f $OUT/PIPELINE_DONE && ls $OUT && cat $OUT/SUMMARY.md && echo SELFTEST_PASS ;;
  *) echo "unknown phase $PHASE"; exit 2 ;;
esac
