#!/bin/bash
# 11-05 input-modality ablation of the selected G1 model (stream warm start + self_force_p 0 + contact-aware decoder):
# speech only / text only / speech + text at INFERENCE (no training), official ECHO-G benchmark offline + engine, semantic
# recognition (pair_win / det_auc, 3 recogniser seeds) and text sensitivity (same audio, swapped text vs seed floor).
#   modality_ablation.sh dev  <out root>    held-out-speaker dev (400 clips, hos view; selection / analysis side)
#   modality_ablation.sh test <out root>    val_common, ONCE per setting after the dev protocol is fixed (usage logged)
#   modality_ablation.sh check <out dir>    infer_test.py default == HEAD and == --modality both (2 dev clips, bitwise)
# env GPUS (bench shards, e.g. "0 0 1"), SETTINGS (labels of X; test default: both audio text), PROTOS (offline engine),
# DEV_CKPT / TEST_CKPT / DEC, PY. Single-modality CFG: cond = the kept modality, uncond = both dropped (the unchanged null
# branch), CFG 2.5, 10 steps; text = null audio in every branch, the audio still gives length / clock / word times.
set -uo pipefail
side=$1; O=$2; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
PY=${PY:-/home/xr-25b7/envs/part2-exec-20261029/bin/python}; export PY
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
LA=${LOCAL_ASSETS:-$HOME/bulk/modality-ablation-1105}               # local copies (CPFS is sshfs on the 5090 host)
A=$B/echog-g1-hos-assets; [ -d $LA/echog-g1-hos-assets/motion_39d_30fps ] && A=$LA/echog-g1-hos-assets
SB=$B/semantic-baseline-1105/g1; [ -f $LA/sem/data-sem-hos/MOTION.npz ] && SB=$LA/sem
# local byte-identical copies of the sf0 checkpoints (md5 checked against CPFS) when present
DCK=$B/echog-g1-sf-1105/dev/sf0/final.pt; [ -f $LA/ckpt/sf0-dev-final.pt ] && DCK=$LA/ckpt/sf0-dev-final.pt
TCK=$B/echog-g1-sf-1105/test/sf0/final.pt; [ -f $LA/ckpt/sf0-test-final.pt ] && TCK=$LA/ckpt/sf0-test-final.pt
declare -A X=( [both]="" [audio]="--modality audio" [text]="--modality text" [textnospk]="--modality text --no-spk"
               [bothnospk]="--no-spk" [bothswap]="--text-swap" [textswap]="--modality text --text-swap" )
log() { echo "$(date -Is) [$side] $*" >> $O/modality.log; }
case $side in
  dev)
    CK=${DEV_CKPT:-$DCK}
    export REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt VIEW_TEST=$B/views/echog-g1-v1-hos VSPLIT=val \
      TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1
    DEF="both audio text textnospk bothnospk bothswap textswap both:1 text:1";;
  test)
    CK=${TEST_CKPT:-$TCK}; DEF="both audio text";;
  check)
    P=$O; mkdir -p $P; head -2 $A/dev_stems.txt > $P/stems2.txt
    rm -rf $P/head && git worktree add --detach $P/head HEAD > /dev/null 2>&1 || { echo "worktree failed"; exit 1; }
    AR=(--ckpt ${DEV_CKPT:-$DCK} --vae $DEC --stats $C/scmg/STATS_echog_g1_v1.json --view $B/views/echog-g1-v1-hos
        --text-cache $C/scmg/text-e5-echog-g1-v1 --stems $P/stems2.txt --split val --cfg-w 2.5 --steps 10 --seed 0)
    export SCMG_J=39 SCMG_NO_GLITCH=1 SCMG_LOCAL_MIRROR=${SCMG_LOCAL_MIRROR:-$HOME/bulk/cpfs-mirror} CUDA_VISIBLE_DEVICES=${GPUS:-0}
    for pr in offline engine; do
      PE=(--protocol $pr); [ $pr = engine ] && PE+=(--text-causal $B/models/text-e5-causal-echog-g1-v1)
      (cd $P/head && $PY scripts/echog_g1/infer_test.py "${AR[@]}" "${PE[@]}" --out $P/$pr-head) > $P/$pr-head.log 2>&1
      $PY scripts/echog_g1/infer_test.py "${AR[@]}" "${PE[@]}" --out $P/$pr-new > $P/$pr-new.log 2>&1
      $PY scripts/echog_g1/infer_test.py "${AR[@]}" "${PE[@]}" --modality both --out $P/$pr-both > $P/$pr-both.log 2>&1
    done
    $PY - $P <<'EOF'
import sys, torch
from pathlib import Path
P = Path(sys.argv[1]); ok = True
for pr in ('offline', 'engine'):
    for b in ('new', 'both'):
        for f in sorted((P / f'{pr}-head').glob('*.pt')):
            x = torch.load(f, weights_only=False)['robot_repr']; y = torch.load(P / f'{pr}-{b}' / f.name, weights_only=False)['robot_repr']
            d = float((x - y).abs().max()); ok &= d == 0. and x.shape == y.shape
            print(f'{pr:8s} HEAD vs {b:5s} {f.stem}: shape {tuple(x.shape)} max_abs_diff {d}')
print('PARITY_OK' if ok else 'PARITY_FAIL')
EOF
    git worktree remove --force $P/head; exit 0;;
  *) echo "usage: modality_ablation.sh dev|test|check <out>"; exit 2;;
esac
R=$O/$side; mkdir -p $R
for spec in ${SETTINGS:-$DEF}; do
  lab=${spec%%:*}; sd=0; [ "$spec" != "$lab" ] && sd=${spec#*:}
  [ -n "${X[$lab]+x}" ] || { log "unknown setting $lab"; continue; }
  for p in ${PROTOS:-offline engine}; do
    tag=${lab}_${p}_seed_$(printf %03d $sd)
    [ -f $R/benchmark_$tag.json ] && continue
    log "START $tag ckpt $CK extra '${X[$lab]}'"
    if [ $side = test ]; then        # test-set usage record (one evaluation per setting; purpose = paper ablation table)
      echo "{\"time\": \"$(date -Is)\", \"split\": \"val_common\", \"ckpt\": \"$CK\", \"decoder\": \"$DEC\", \"setting\": \"$lab\", \"infer_extra\": \"${X[$lab]}\", \"protocol\": \"$p\", \"seed\": $sd, \"purpose\": \"report: input-modality ablation table (protocol fixed on dev before this run)\", \"host\": \"$(hostname)\"}" >> $O/TEST_USAGE.jsonl
    fi
    CFGW=2.5 BENCH_STEPS=10 INFER_EXTRA="${X[$lab]}" GPUS="${GPUS:-0}" bash scripts/echog_g1/bench.sh scmg $CK $DEC $R/$tag $p $sd >> $O/run-$side.log 2>&1 \
      && log "DONE $tag" || log "FAILED $tag"
  done
done
# semantic recognition (seed 0 runs; dev: hos recognisers on the dev events; test: v1 recognisers)
if [ $side = dev ]; then DATA=$SB/data-sem-hos; RECS="$SB/rec-sem-hos-seed*"; SPL=dev
else DATA=$B/semantic-baseline-1105/g1/data-sem; RECS="$B/semantic-baseline-1105/g1/rec-sem-v1*"; SPL=test; fi
specs=(gt=-)
for f in $R/benchmark_*_seed_000.json; do t=$(basename $f .json); t=${t#benchmark_}; specs+=(${t%_seed_000}=$R/$t); done
PAIRS=${PAIRS:-} bash scripts/semantic/sem_score_dirs.sh ${GPUS%% *} $DATA "$RECS" $O/sem-$side $SPL "${specs[@]}" >> $O/sem-$side.log 2>&1 && log "SEM_DONE" || log "SEM_FAILED"
if [ $side = dev ]; then                                            # text sensitivity: swap vs seed floor
  ts=()
  for m in both text; do for p in ${PROTOS:-offline engine}; do
    [ -f $R/benchmark_${m}swap_${p}_seed_000.json ] && [ -f $R/benchmark_${m}_${p}_seed_001.json ] && \
      ts+=(${m}_$p=$R/${m}_${p}_seed_000,$R/${m}swap_${p}_seed_000,$R/${m}_${p}_seed_001)
  done; done
  [ ${#ts[@]} -gt 0 ] && $PY scripts/echog_g1/text_sensitivity.py --out $O/TEXT_SENS-$side.json "${ts[@]}" >> $O/modality.log 2>&1 && log "TEXT_SENS_DONE"
fi
log "ALL_DONE"
