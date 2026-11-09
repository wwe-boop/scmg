#!/bin/bash
# 11-07 (coordinator): two extra rows of the interruption final run, from the SAME B3c / no_int continuations of psync_r0.sh test:
#   Funnel_p   = funnel_retract.py h22m10s1 (--tr 1.4 --tf 2.2 --m0 1.0 --smooth 0.1, eps 0.02, alpha 0.6; pre-registered, §7.59)
#   Funnel18_p = g18m8s1 (--tf 1.8 --m0 0.8 --smooth 0.1), the latency-balanced point
# + safety_project --smooth 300, eval_interrupt (with R0causal_p in the same call), rkd_diff_boot vs R0causal_p.
# funnel_retract.py = verbatim copy of scmg-wt-funnel/funnel_retract.py (md5 409b6db9d5cc57145aca8a46ea050f6f).
#   psync_funnel_post.sh <tag>     (waits for PSYNC_R0_DONE <tag>; TEST_USAGE registered before by the caller)
set -uo pipefail
TAG=$1; W=$HOME/bulk/interrupt-1105; CODE=$(cd "$(dirname "$0")/../.." && pwd); S=$CODE/scripts/interrupt
PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python; PYH=/home/xr-25b7/bulk/echog-official-ro/official/src
INT=interrupts_test.json; HJ=human_ref_v2_train_drop.json; O=final_psync_$TAG; U=$O/no_int
cd $W
until grep -q "PSYNC_R0_DONE $TAG" logs/psync_$TAG.log; do sleep 60; done
echo "409b6db9d5cc57145aca8a46ea050f6f  $S/funnel_retract.py" | md5sum -c - || exit 3
PYTHONPATH=$PYH $PY $S/funnel_retract.py --in-dir $O/B3c --prefix-from $U --out-dir $O/Funnel --interrupts $INT --human $HJ --tr 1.4 --tf 2.2 --m0 1.0 --smooth 0.1 --eps 0.02 --alpha 0.6 > logs/psync_${TAG}_funnel.log 2>&1
PYTHONPATH=$PYH $PY $S/funnel_retract.py --in-dir $O/B3c --prefix-from $U --out-dir $O/Funnel18 --interrupts $INT --human $HJ --tr 1.4 --tf 1.8 --m0 0.8 --smooth 0.1 --eps 0.02 --alpha 0.6 >> logs/psync_${TAG}_funnel.log 2>&1
for d in Funnel Funnel18; do for sh in 0 1 2 3; do CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYH ECHOG_OFFICIAL_SRC=$PYH OMP_NUM_THREADS=2 $PY $S/safety_project.py \
  --in-dir $O/$d --out-dir $O/${d}_p300 --interrupts $INT --smooth 300 --shard $sh/4 > /dev/null 2>&1 & done; wait; done
bash ev.sh $INT metrics/psync_${TAG}_funnel.json 1 R0causal_p=B3:$O/R0causal_p300:$U Funnel_p=B3:$O/Funnel_p300:$U Funnel18_p=B3:$O/Funnel18_p300:$U > logs/psync_${TAG}_funnel_eval.txt 2>&1
for m in Funnel_p Funnel18_p; do PYTHONPATH=$PYH $PY $S/rkd_diff_boot.py metrics/psync_${TAG}_funnel.jsonl $m metrics/psync_${TAG}_funnel.jsonl R0causal_p >> metrics/psync_${TAG}_funnel_rkddiff.txt 2>&1; done
echo "$(date -Is) FUNNEL_DONE $TAG" >> logs/psync_$TAG.log
