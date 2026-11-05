#!/bin/bash
# ONE test-set run of the dev-selected floor configuration + the interruption-causal B3 baseline (11-06).
#   final_test_floor.sh <lane> <floor value> <post: raw|p300>      runs on zehan-server GPUs 2/3 (inference) + local eval
set -uo pipefail
LANE=$1; FV=$2; POST=$3; W=$HOME/bulk/interrupt-1105; I=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/interrupt-baseline-1105
R=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-g1-floor-1106/test
ssh zehan-server "bash $I/zs_eval.sh 2 $R/$LANE/final.pt $I/test_floor/$LANE test $FV & bash $I/zs_eval.sh 3 /mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-g1-sf-1105/test/sf0/final.pt $I/test_floor/sf0 test 0 & wait"
mkdir -p $W/test_floor && ssh zehan-server "cd $I/test_floor && tar cf - $LANE/no_int $LANE/f$FV sf0/no_int sf0/f0" | tar xf - -C $W/test_floor
cd $W && bash proj_par.sh interrupts_test.json _p300 test_floor/$LANE/f$FV test_floor/sf0/f0 > logs/proj_test_floor.out 2>&1
U=$W/test_floor/$LANE/no_int; U0=$W/test_floor/sf0/no_int
bash $W/ev.sh $W/interrupts_test.json $W/metrics/test_floor.json 1 B3c=B3:test_floor/sf0/f0:$U0 B3c_p=B3:test_floor/sf0/f0_p300:$U0 \
  floor_${LANE}_f${FV}=B3:test_floor/$LANE/f$FV:$U floor_${LANE}_f${FV}_p=B3:test_floor/$LANE/f${FV}_p300:$U
PYTHONPATH=/home/xr-25b7/bulk/echog-official-ro/official/src /home/xr-25b7/envs/part2-exec-20261029/bin/python \
  /home/xr-25b7/projects/gesture/scmg-wt-floor/scripts/interrupt/boot.py --human $W/human_ref_v2_train_drop --jsonl $W/metrics/test_floor.jsonl --n 500 > $W/metrics/test_floor_boot.txt
echo FINAL_FLOOR_DONE
