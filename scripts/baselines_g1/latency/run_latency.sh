#!/bin/bash
# Latency / memory / RTF of the G1 main-table methods, one method at a time on ONE GPU (batch 1), zehan-server.
#   run_latency.sh <out dir> [methods...]      env GPU=<index> CPUS=<taskset list> N=<clips, 0 = the 50-clip set>
# Methods: scmg_offline scmg_engine scmg_psync echog_stream echog robogesture semtalk emage lsm gmr_semtalk gmr_lsm gmr_emage
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); O=$1; shift; mkdir -p $O/logs
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; L=$B/baselines-g1-1105/latency
ECHOG=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python
RG=${RG_PY:-$L/envs/rg/bin/python}          # venv on the echog runtime + robogesture-py310 site-packages (image independent)
GMR=${GMR_PY:-$L/envs/gmr/bin/python}       # venv on the echog runtime + gmr-repro env/gmr + robogesture site-packages
ST=$L/envs/semtalk/bin/python
CAUSAL=$L/envs/causal/bin/python
P=$B/baseline-repro-1104/gmr-repro/preds-1104
export CUDA_VISIBLE_DEVICES=${GPU:?GPU} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 \
  TMPDIR=${TMPDIR:-/tmp/lat-zehan}; mkdir -p $TMPDIR
TS=${CPUS:+taskset -c $CPUS}; NA=${N:+--n $N}
log() { echo "$(date -Is) $*" | tee -a $O/run.log; }
for m in "${@:-scmg_offline scmg_engine echog robogesture semtalk emage lsm gmr_semtalk gmr_lsm}"; do
  for mm in $m; do
    log "START $mm $(nvidia-smi -i ${GPU%%,*} --query-gpu=memory.used,utilization.gpu --format=csv,noheader) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    case $mm in
      scmg_offline|scmg_engine) SCMG_J=39 $TS $ECHOG $D/lat_scmg.py --protocol ${mm#scmg_} --out $O $NA ;;
      scmg_psync) SCMG_J=39 SCMG_REL=$L/snap/scmg_causal $TS $CAUSAL $D/lat_scmg.py --protocol psync --out $O $NA ;;
      cg_offline_off|cg_offline_on|cg_engine_on|cg_psync_off|cg_psync_on|cg_psync_on_grades|cg_psync_off_grades|cg_psync_on_grades_inc|cg_psync_off_grades_inc)   # 11-06 CUDA-graph before / after (code snap/scmg_cg)
        set -- ${mm//_/ }; P2=$2; CGF=""; [ "$3" = on ] && CGF=--cuda-graph; SR=""; [ "${4:-}" = grades ] && SR="--spk-refresh grades"; [ "${5:-}" = inc ] && SR="$SR --causal-feat inc"; [ -n "${CG_CKPT:-}" ] && SR="$SR --ckpt $CG_CKPT"
        SCMG_J=39 SCMG_REL=${CG_REL:-$L/snap/scmg_cg3} $TS $CAUSAL $D/lat_scmg.py --protocol $P2 $CGF $SR --tag _cg${3}${4:+_$4}${5:+_$5} --out $O $NA ;;
      echog_stream) $TS $ECHOG $D/lat_echog_stream.py --out $O $NA ;;
      echog_stream_nocudnn) ECHOG_W2V_NOCUDNN=1 $TS $ECHOG $D/lat_echog_stream.py --out $O $NA ;;
      echog) $TS $ECHOG $D/lat_echog.py --out $O $NA ;;
      robogesture) $TS $RG $D/lat_robogesture.py --out $O $NA ;;
      semtalk) SEMTALK_ROOT=$B/baselines-g1-1105/semtalk/root ECHOG_DS=$B/baselines-g1-1105/semtalk/root/ds MODELSCOPE_OFFLINE=1 \
               $TS $ST $D/lat_semtalk.py --out $O $NA ;;
      emage|lsm) GMR_WEIGHTS=${GMR_WEIGHTS:-$B/baselines-g1-1105/assets/gmr_weights} $TS $GMR $D/lat_emage_lsm.py --method $mm --out $O $NA ;;
      gmr_semtalk) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" $TS $GMR $D/lat_gmr.py --src $B/baselines-g1-1105/semtalk/run/smplx_seed0 --tag semtalk --out $O $NA ;;
      gmr_lsm) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" $TS $GMR $D/lat_gmr.py --src $P/lsm_shortcut_seed0 --tag lsm --out $O $NA ;;
      gmr_emage) OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="" $TS $GMR $D/lat_gmr.py --src $B/baselines-g1-1105/assets/emage_smplx_seed0 --tag emage --out $O $NA ;;
      *) log "unknown $mm"; continue ;;
    esac > $O/logs/$mm.log 2>&1 && log "DONE $mm" || log "FAILED $mm"
  done
done
log ALL_DONE
