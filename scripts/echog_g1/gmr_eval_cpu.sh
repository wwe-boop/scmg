#!/bin/bash
# 11-04: official ECHO-G benchmark of the reproduced Table-I baselines (EMAGE+GMR, GestureLSM+GMR), CPU-only DLC job
# (user 11-04: no heavy CPU work on zehan-server).  Predictions were converted to G1 39-d on zehan-server by DeepSeek and
# copied to $G/preds-1104; the conversion of GestureLSM was stopped mid-run, so every .pt is validated first and broken /
# missing stems are re-converted from the SMPL-X source (converter skips existing outputs).
#   gmr_eval_cpu.sh            env NW=<converter workers>
set -uo pipefail
G=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro; P=$G/preds-1104; PY=$G/env/gmr/bin/python
SPLIT=$G/data/echog/splits/val_common.txt; [ -f $SPLIT ] || SPLIT=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root/splits/val_common.txt
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
if [ -s $O/check_g1_lsm_seed0.txt ]; then
  for i in $(seq 0 $((${NW:-8} - 1))); do
    env CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY $G/scripts/smplx_npz_to_g1.py --src $P/lsm_shortcut_seed0 \
      --out $P/g1_lsm_seed0 --stems $O/check_g1_lsm_seed0.txt --shard $i/${NW:-8} --generator "lsm shortcut seed0" >> $O/reconvert.log 2>&1 &
  done; wait; log "RECONVERT_DONE $(ls $P/g1_lsm_seed0 | wc -l) files"
fi
[ -s $O/check_g1_emage_seed0.txt ] && log "WARN emage has broken files and no source copy here: $(head -3 $O/check_g1_emage_seed0.txt | paste -sd' ')"
for d in g1_emage_seed0 g1_lsm_seed0; do
  OMP_NUM_THREADS=${OMP:-16} CUDA_VISIBLE_DEVICES="" bash $G/scripts/run_eval.sh $P/$d $SPLIT ${d%_seed0}+gmr $O/benchmark_$d.json >> $O/eval_$d.log 2>&1 \
    && log "EVAL_DONE $d" || log "EVAL_FAILED $d"
done
log GMR_EVAL_ALL_DONE
