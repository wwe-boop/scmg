#!/bin/bash
# post-processing + scoring of the one test run (test_v3): blends, projection (CPU), v3 table, bootstrap
W=$HOME/bulk/interrupt-1105; I=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/interrupt-baseline-1105; cd $W
PYH=/home/xr-25b7/bulk/echog-official-ro/official/src; S=/home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt; PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python
mkdir -p test_v3 && ssh zehan-server "cd $I/test_v3 && tar cf - no_int B3c" | tar xf - -C test_v3
INT=interrupts_test.json; HJ=human_ref_v2_train_drop.json; O=test_v3
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/B3c --prefix-from $O/no_int --out-dir $O/R0causal --interrupts $INT --human $HJ --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/no_int --full-clip --out-dir $O/B2 --interrupts $INT --human $HJ --tr 1.8 --tb 0.5 --w0 1.0
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/no_int --full-clip --out-dir $O/B2fast --interrupts $INT --human $HJ --tr 1.0 --tb 0.5 --w0 1.0
for d in R0causal B2 B2fast B3c; do for sh in 0 1 2 3; do CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYH ECHOG_OFFICIAL_SRC=$PYH OMP_NUM_THREADS=2 $PY $S/safety_project.py \
  --in-dir $O/$d --out-dir $O/${d}_p300 --interrupts $INT --smooth 300 --shard $sh/4 > /dev/null 2>&1 & done; wait; done
U=$O/no_int; E=preds/echog; EG=/mnt/bulk/zehan/echog-dataset
bash ev.sh $INT metrics/test_v3_final.json 1 echog_B1=B1:$E echog_B2=B2:$E echog_B2fast=B2fast:$E ours_B1=B1:$U ours_B2_p=B3:$O/B2_p300:$U ours_B2fast_p=B3:$O/B2fast_p300:$U \
  ours_B3c=B3:$O/B3c:$U ours_B3c_p=B3:$O/B3c_p300:$U R0causal_p=B3:$O/R0causal_p300:$U GT_B2=B2:$EG/ds/motion_39d_30fps > logs/ev_test_v3_final.txt 2>&1
PYTHONPATH=$PYH $PY /home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt/rkd_diff_boot.py metrics/test_v3_final.jsonl R0causal_p metrics/test_v3_final.jsonl ours_B2_p > metrics/test_v3_rkddiff.txt 2>&1
PYTHONPATH=$PYH $PY /home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt/rkd_diff_boot.py metrics/test_v3_final.jsonl R0causal_p metrics/test_v3_final.jsonl echog_B2 >> metrics/test_v3_rkddiff.txt 2>&1
PYTHONPATH=$PYH $PY $S/boot.py --human ${HJ%.json} --jsonl metrics/test_v3_final.jsonl --n 500 > metrics/test_v3_boot.txt 2>&1
echo TEST_V3_POST_DONE >> logs/ev_test_v3_final.txt
