#!/bin/bash
# 11-07: the frozen R0causal interruption pipeline of final_test_r0causal.sh with the continuation in the CAUSAL psync protocol
# (infer_retract.py --protocol psync, = the main-table psync setting: grades CAM++, incremental feat89, voiced text,
# --seed-mode per_clip --seed 0; scripts/tests/test_retract_psync.py ALL_OK). Everything after the inference is unchanged:
#   R0causal : blend_retract.py --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1 --prefix-from <uninterrupted>
#   B2       : blend_retract.py --full-clip --tr 1.8 --tb 0.5 --w0 1.0
#   projection safety_project.py --smooth 300 on R0causal, B2, B3c; scoring eval_interrupt.py (v3 main) + rkd_diff_boot + boot
#   psync_r0.sh dev|test <ckpt> <tag> [gpus, default "1 1 1 1"]     (all on this host; test registers in TEST_USAGE first)
# Differences to final_test_r0causal.sh: protocol psync instead of engine + prefix_test, inference on this host (local paths).
set -uo pipefail
SIDE=$1; CK=$2; TAG=$3; GPUS=(${4:-1 1 1 1})
W=$HOME/bulk/interrupt-1105; I=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/interrupt-baseline-1105
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; M=$HOME/bulk/cpfs-mirror/CKPT/zehan/gesture-scmg
CODE=$(cd "$(dirname "$0")/../.." && pwd); S=$CODE/scripts/interrupt; EG=/mnt/bulk/zehan/echog-dataset; CZ=$HOME/bulk/causal-train-1105
VAE=${VAE:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
export SCMG_J=39 SCMG_NO_GLITCH=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 SCMG_LOCAL_MIRROR=$HOME/bulk/cpfs-mirror
unset SCMG_TEXT_CAUSAL
PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python
if [ $SIDE = dev ]; then
  INT=interrupts_dev.json; HJ=human_ref_v2_train_drop.json; O=dev/psync_$TAG
  A=(--view $B/views/echog-g1-v1-hos --split val --text-cache $C/scmg/text-e5-echog-g1-v1 --text-causal $B/models/text-e5-causal-echog-g1-v1
     --wav-dir $M/echog-g1-hos-assets/audio --words-dir $CZ/words-dev)
else
  INT=interrupts_test.json; HJ=human_ref_v2_train_drop.json; O=final_psync_$TAG
  A=(--view $C/echog-g1-v1/test --split test --text-cache $C/scmg/text-e5-echog-g1-v1-test --text-causal $B/models/text-e5-causal-echog-g1-v1-test
     --wav-dir $EG/ds/audio --words-dir $CZ/words-vc)
  ssh zehan-server "cat >> $I/TEST_USAGE.jsonl" <<EOF
{"time": "$(date -Is)", "split": "val_common subset 600 clips x 3 interrupts (interrupts_test.json)", "model": "$CK (sha256 $(sha256sum $CK | cut -c1-16))", "setting": "FINAL SYSTEM interruption column, CAUSAL psync continuation: R0causal + B2_p + B3c (frozen pipeline of final_test_r0causal.sh, continuation infer_retract --protocol psync grades/inc, psync_r0.sh, code $(cd $CODE && git rev-parse --short HEAD)), per_clip seed 0, abruptness v3", "purpose": "report: final main-table interruption column; registered before the run", "host": "xr-xx"}
EOF
fi
A+=(--ckpt $CK --vae $VAE --stats $C/scmg/STATS_echog_g1_v1.json --protocol psync --pstats $CODE/scripts/echog_g1/PROSODY_STATS_echog_g1_hos_train.json
    --campplus $HOME/bulk/models-ro/campplus --spk-refresh grades --causal-feat inc --seed 0 --seed-mode per_clip --steps 10 --cfg-w 2.5)
cd $W && mkdir -p $O logs metrics
$PY -c "import json; print('\n'.join(sorted({r['stem'] for r in json.load(open('$INT'))['interrupts']})))" > $O/stems.txt
n=${#GPUS[@]}; echo "$(date -Is) start $SIDE $TAG $CK $(wc -l < $O/stems.txt) stems, $n processes" >> logs/psync_$TAG.log
for k in $(seq 0 $((n - 1))); do
  ( cd $CODE && CUDA_VISIBLE_DEVICES=${GPUS[$k]} PYTHONPATH=$CODE:$HOME/bulk/pydeps/ort-cpu $PY scripts/interrupt/infer_retract.py "${A[@]}" --shard $k/$n \
      --no-interrupt --stems $W/$O/stems.txt --out $W/$O/no_int > $W/logs/psync_${TAG}_noint_$k.log 2>&1
    CUDA_VISIBLE_DEVICES=${GPUS[$k]} PYTHONPATH=$CODE:$HOME/bulk/pydeps/ort-cpu $PY scripts/interrupt/infer_retract.py "${A[@]}" --shard $k/$n \
      --interrupts $W/$INT --audio end --restart chunk --out $W/$O/B3c > $W/logs/psync_${TAG}_int_$k.log 2>&1
    echo "$(date -Is) PSYNC_DONE $k" >> $W/logs/psync_$TAG.log ) &
done; wait
PYH=/home/xr-25b7/bulk/echog-official-ro/official/src
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/B3c --prefix-from $O/no_int --out-dir $O/R0causal --interrupts $INT --human $HJ --tr 1.4 --tb 0.5 --w0 0.6 --tex 1.0 --tex-smooth 0.1
PYTHONPATH=$PYH $PY $S/blend_retract.py --in-dir $O/no_int --full-clip --out-dir $O/B2 --interrupts $INT --human $HJ --tr 1.8 --tb 0.5 --w0 1.0
for d in R0causal B2 B3c; do for sh in 0 1 2 3; do CUDA_VISIBLE_DEVICES= PYTHONPATH=$PYH ECHOG_OFFICIAL_SRC=$PYH OMP_NUM_THREADS=2 $PY $S/safety_project.py \
  --in-dir $O/$d --out-dir $O/${d}_p300 --interrupts $INT --smooth 300 --shard $sh/4 > /dev/null 2>&1 & done; wait; done
U=$O/no_int; E=preds/echog
if [ $SIDE = dev ]; then ROWS=(ours_B2_p=B3:$O/B2_p300:$U ours_B3c=B3:$O/B3c:$U ours_B3c_p=B3:$O/B3c_p300:$U R0causal_p=B3:$O/R0causal_p300:$U)
else ROWS=(echog_B1=B1:$E echog_B2=B2:$E echog_B2fast=B2fast:$E ours_B2_p=B3:$O/B2_p300:$U ours_B3c=B3:$O/B3c:$U ours_B3c_p=B3:$O/B3c_p300:$U
           R0causal_p=B3:$O/R0causal_p300:$U GT_B2=B2:$EG/ds/motion_39d_30fps); fi
bash ev.sh $INT metrics/psync_$TAG.json 1 "${ROWS[@]}" > logs/psync_${TAG}_eval.txt 2>&1
PYTHONPATH=$PYH $PY $S/rkd_diff_boot.py metrics/psync_$TAG.jsonl R0causal_p metrics/psync_$TAG.jsonl ours_B2_p > metrics/psync_${TAG}_rkddiff.txt 2>&1
[ $SIDE = test ] && PYTHONPATH=$PYH $PY $S/rkd_diff_boot.py metrics/psync_$TAG.jsonl R0causal_p metrics/psync_$TAG.jsonl echog_B2 >> metrics/psync_${TAG}_rkddiff.txt 2>&1
PYTHONPATH=$PYH $PY $S/boot.py --human ${HJ%.json} --jsonl metrics/psync_$TAG.jsonl --n 500 > metrics/psync_${TAG}_boot.txt 2>&1
echo "$(date -Is) PSYNC_R0_DONE $TAG" >> logs/psync_$TAG.log
