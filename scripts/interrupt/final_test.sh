#!/bin/bash
# ONE test-set run of the selected zero-training retraction (R0) + the same safety projection on the baselines (11-05).
# Selected on dev (hos speakers, 900 interrupts): blend of the B3 continuation (audio end, chunk restart) into the damped
# retraction (T_r 1.4 s, T_b 0.5 s, w0 0.6) + band-pass texture (g 1, tau 0.25 s, smooth 0.1 s), frames < f_i from the
# uninterrupted output, then safety projection (smooth 300, wrist-hip proxy).
#   final_test.sh <work dir>   (needs b3_test/{end,no_int}, preds/echog; writes test_final/, metrics/test_final.*)
set -uo pipefail
W=$1; cd $W; S=$(cd "$(dirname "$0")" && pwd); PYH=/home/xr-25b7/bulk/echog-official-ro/official/src
PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python; INT=$W/interrupts_test.json; HJ=$W/human_ref_v2_train_drop.json; O=$W/test_final
mkdir -p $O; bl() { PYTHONPATH=$PYH $PY $S/blend_retract.py --interrupts $INT --human $HJ "$@"; }
bl --in-dir b3_test/end --prefix-from b3_test/no_int --out-dir $O/R0 --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1
bl --in-dir b3_test/no_int --full-clip --out-dir $O/ours_B2 --tr 1.8 --tb 0.5 --w0 1.0
bl --in-dir b3_test/no_int --full-clip --out-dir $O/ours_B2fast --tr 1.0 --tb 0.5 --w0 1.0
bl --in-dir preds/echog --full-clip --out-dir $O/echog_B2 --tr 1.8 --tb 0.5 --w0 1.0
bl --in-dir preds/echog --full-clip --out-dir $O/echog_B2fast --tr 1.0 --tb 0.5 --w0 1.0
mkdir -p $O/ours_B3; cp b3_test/end/*.pt $O/ours_B3/
bash $W/proj_par.sh $INT _p300 $O/R0 $O/ours_B2 $O/ours_B2fast $O/echog_B2 $O/echog_B2fast $O/ours_B3
U=$W/b3_test/no_int; E=$W/preds/echog
bash $W/ev.sh $INT $W/metrics/test_final.json 1 echog_B1=B1:$E echog_B2=B2:$E echog_B2fast=B2fast:$E echog_B2_p=B3:$O/echog_B2_p300:$E \
  echog_B2fast_p=B3:$O/echog_B2fast_p300:$E ours_B1=B1:$U ours_B2=B2:$U ours_B2fast=B2fast:$U ours_B2_p=B3:$O/ours_B2_p300:$U \
  ours_B2fast_p=B3:$O/ours_B2fast_p300:$U ours_B3=B3:$W/b3_test/end:$U ours_B3_p=B3:$O/ours_B3_p300:$U R0_raw=B3:$O/R0:$U R0=B3:$O/R0_p300:$U
PYTHONPATH=$PYH $PY $S/boot.py --human ${HJ%.json} --jsonl $W/metrics/test_final.jsonl --n 500 > $W/metrics/test_final_boot.txt
echo FINAL_TEST_DONE
