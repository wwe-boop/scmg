#!/bin/bash
# 11-04: official ECHO-G benchmark of the reproduced Table-I baselines (EMAGE+GMR, GestureLSM+GMR), CPU-only DLC job
# (user 11-04: no heavy CPU work on zehan-server).  Predictions were converted to G1 39-d on zehan-server by DeepSeek and
# copied to $G/preds-1104; the conversion of GestureLSM was stopped mid-run, so every .pt is validated first and broken /
# missing stems are re-converted from the SMPL-X source (converter skips existing outputs).
#   gmr_eval_cpu.sh            env NW=<converter workers>
set -uo pipefail
G=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro; P=$G/preds-1104; PY=$G/env/gmr/bin/python
SPLIT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root/splits/val_common.txt
DR=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official; OFF=$W/official
O=$P/results; mkdir -p $O; log() { echo "$(date -Is) $*" | tee -a $O/run.log; }
for d in g1_emage_seed0 g1_lsm_seed0; do
  $PY - $P/$d $SPLIT > $O/check_$d.txt <<'PY'
import sys, torch
from pathlib import Path
d = Path(sys.argv[1]); stems = [l.strip() for l in open(sys.argv[2]) if l.strip()]; bad = []
for s in stems:
    f = d / f'{s}.pt'
    try:
        x = torch.load(f, weights_only=False); r = x['robot_repr']
        assert r.ndim == 2 and r.shape[1] == 39 and r.shape[0] == x['real_num_frames'] and torch.isfinite(r).all()
    except Exception as e:
        bad.append(s); f.unlink(missing_ok=True)
print('\n'.join(bad))
PY
  log "CHECK $d bad=$(grep -c . $O/check_$d.txt)"
done
if grep -q . $O/check_g1_lsm_seed0.txt; then
  for i in $(seq 0 $((${NW:-8} - 1))); do
    env CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY $G/scripts/smplx_npz_to_g1.py --src $P/lsm_shortcut_seed0 \
      --out $P/g1_lsm_seed0 --stems $O/check_g1_lsm_seed0.txt --shard $i/${NW:-8} --generator "lsm shortcut seed0" >> $O/reconvert.log 2>&1 &
  done; wait; log "RECONVERT_DONE $(ls $P/g1_lsm_seed0 | wc -l) files"
fi
grep -q . $O/check_g1_emage_seed0.txt && log "WARN emage has broken files and no source copy here: $(head -3 $O/check_g1_emage_seed0.txt | paste -sd' ')"
for d in g1_emage_seed0 g1_lsm_seed0; do
  # same evaluator flags as scripts/echog_g1/bench.sh (our own G1 rows), data_root assets on CPFS (run_eval.sh pointed at
  # zehan-server-local copies that a DLC pod cannot see)
  OMP_NUM_THREADS=${OMP:-16} CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OFF/src ${EPY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python} $OFF/scripts/eval_g1_motion_cls.py --pred-dir $P/$d --ref-dir $DR/motion_39d_30fps \
    --val-split $SPLIT --wav-dir $DR/audio --mmae-file $DR/eval_assets/mmae/g1_mmae_30body_30fps.npy --g1-ae-ckpt $W/weights/evaluation/g1_fgd_encoder.pt \
    --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems --tag ${d%_seed0}+gmr --out $O/benchmark_$d.json >> $O/eval_$d.log 2>&1 \
    && log "EVAL_DONE $d" || log "EVAL_FAILED $d"
done
log GMR_EVAL_ALL_DONE
