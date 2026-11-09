#!/bin/bash
# The "final system" test run of the interruption column (11-06, coordinator): ONE run on the final-recipe test-side model.
# Everything is frozen (criteria/candidate_r0causal_1106.json, abruptness v3 criteria/metric_change_v3_1106.json):
#   continuation : infer_retract.py, engine protocol, audio end at t_i, chunk restart, text truncated to spoken words,
#                  interruption-causal prefix feat89 / CAM++ (prefix_test), --seed-mode per_clip --seed 0
#   R0causal     : blend_retract.py --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1 --prefix-from <uninterrupted>
#   B2           : blend_retract.py --full-clip --tr 1.8 --tb 0.5 --w0 1.0 (critically damped to R_sil)
#   projection   : safety_project.py --smooth 300 (wrist-hip proxy on) on R0causal, B2, B3c
#   scoring      : eval_interrupt.py (v3 main, v1 / v2 secondary), rkd_diff_boot.py vs B2_p and vs ECHO-G B2, boot.py
#   rows         : R0causal_p, B2_p, B3c, B3c_p (final model) + ECHO-G B1 / B2 / B2fast (official outputs) + GT + B2
#   final_test_r0causal.sh <test-side final ckpt (CPFS path)> <its VAE (CPFS path)> <tag> [gpus on zehan-server, default "0 2 3"]
# Run from xr-xx; inference on zehan-server free GPUs (one thread per process), scoring on xr-xx CPU. Registers itself in
# TEST_USAGE.jsonl BEFORE the inference. NOTE: if the final recipe is trained for the causal-input protocol (infer_test
# --protocol psync / --causal-inputs), the continuation protocol here must be matched first (infer_retract has no
# per-tick causal_feat inputs yet) -- check before running.
set -uo pipefail
CK=$1; VAE=$2; TAG=$3; GPUS=(${4:-0 2 3})
W=$HOME/bulk/interrupt-1105; I=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/interrupt-baseline-1105; O=final_$TAG
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
CODE=/home/xr-25b7/projects/gesture/scmg-wt-floor; S=$CODE/scripts/interrupt
ssh zehan-server "test -f $CK && test -f $VAE" || { echo "checkpoint / VAE not found on CPFS"; exit 2; }
ssh zehan-server "cat >> $I/TEST_USAGE.jsonl" <<EOF
{"time": "$(date -Is)", "split": "val_common subset 600 clips x 3 interrupts (interrupts_test.json)", "model": "$CK", "setting": "FINAL SYSTEM interruption column: R0causal + B2_p + B3c (frozen, final_test_r0causal.sh, code $(cd $CODE && git rev-parse --short HEAD)), per_clip seed 0, abruptness v3", "purpose": "report: final main-table interruption column; registered before the run", "host": "zehan-server + xr-xx"}
EOF
cd $CODE && tar cf - scmg scripts > /tmp/final_code_$TAG.tar && /bin/cat /tmp/final_code_$TAG.tar | ssh zehan-server "mkdir -p $I/code-final-$TAG && cd $I/code-final-$TAG && tar xf -"
ssh zehan-server "cat > $I/final_$TAG.sh" <<EOF
#!/bin/bash
G=\$1; PART=\$2; SH=\$3
export SCMG_J=39 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\$G PYTHONPATH=$I/code-final-$TAG
PY=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python; cd $I/code-final-$TAG
A=(--ckpt $CK --vae $VAE --stats $C/scmg/STATS_echog_g1_v1.json --view $C/echog-g1-v1/test --text-cache $C/scmg/text-e5-echog-g1-v1-test
   --split test --protocol engine --text-causal $B/models/text-e5-causal-echog-g1-v1-test --seed 0 --seed-mode per_clip --shard \$SH)
if [ \$PART = noint ]; then \$PY scripts/interrupt/infer_retract.py "\${A[@]}" --no-interrupt --stems $I/stems_test.txt --out $I/$O/no_int
else \$PY scripts/interrupt/infer_retract.py "\${A[@]}" --interrupts $I/interrupts_test.json --words $I/utterances.json --audio end --restart chunk \
       --prefix-feats $I/prefix_test --out $I/$O/B3c; fi
echo "\$(date -Is) FINAL_DONE \$PART \$SH" >> $I/logs/final_$TAG.log
EOF
n=${#GPUS[@]}
for k in $(seq 0 $((n - 1))); do
  ssh zehan-server "setsid nohup bash -c 'bash $I/final_$TAG.sh ${GPUS[$k]} noint $k/$n; bash $I/final_$TAG.sh ${GPUS[$k]} int $k/$n' > /dev/null 2>&1 < /dev/null &"
done
until ssh zehan-server "[ \$(grep -c 'FINAL_DONE int' $I/logs/final_$TAG.log 2>/dev/null || echo 0) -ge $n ]"; do sleep 60; done
cd $W && mkdir -p $O && ssh zehan-server "cd $I/$O && tar cf - no_int B3c" | tar xf - -C $O
PYH=/home/xr-25b7/bulk/echog-official-ro/official/src; PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python; INT=interrupts_test.json; HJ=human_ref_v2_train_drop.json
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/B3c --prefix-from $O/no_int --out-dir $O/R0causal --interrupts $INT --human $HJ --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/no_int --full-clip --out-dir $O/B2 --interrupts $INT --human $HJ --tr 1.8 --tb 0.5 --w0 1.0
for d in R0causal B2 B3c; do for sh in 0 1 2 3; do CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYH ECHOG_OFFICIAL_SRC=$PYH OMP_NUM_THREADS=2 $PY $S/safety_project.py \
  --in-dir $O/$d --out-dir $O/${d}_p300 --interrupts $INT --smooth 300 --shard $sh/4 > /dev/null 2>&1 & done; wait; done
U=$O/no_int; E=preds/echog; EG=/mnt/bulk/zehan/echog-dataset
bash ev.sh $INT metrics/final_$TAG.json 1 echog_B1=B1:$E echog_B2=B2:$E echog_B2fast=B2fast:$E ours_B2_p=B3:$O/B2_p300:$U ours_B3c=B3:$O/B3c:$U ours_B3c_p=B3:$O/B3c_p300:$U \
  R0causal_p=B3:$O/R0causal_p300:$U GT_B2=B2:$EG/ds/motion_39d_30fps > logs/final_$TAG.txt 2>&1
PYTHONPATH=$PYH $PY $S/rkd_diff_boot.py metrics/final_$TAG.jsonl R0causal_p metrics/final_$TAG.jsonl ours_B2_p > metrics/final_${TAG}_rkddiff.txt 2>&1
PYTHONPATH=$PYH $PY $S/rkd_diff_boot.py metrics/final_$TAG.jsonl R0causal_p metrics/final_$TAG.jsonl echog_B2 >> metrics/final_${TAG}_rkddiff.txt 2>&1
PYTHONPATH=$PYH $PY $S/boot.py --human ${HJ%.json} --jsonl metrics/final_$TAG.jsonl --n 500 > metrics/final_${TAG}_boot.txt 2>&1
echo FINAL_TEST_DONE $TAG
