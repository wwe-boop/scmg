#!/bin/bash
# ECHO-G trained streaming variant on EX002 (34D, 50 FPS) in ONE DLC pod (11-05), the EX002 counterpart of g1_confirm.sh:
#   1) train_stream.py --profile ex002 from the EX002 offline ECHO-G (official SGDiT trained by dlc-package-r2, INIT) on
#      full-data-r1 (DDP on all GPUS, STEPS / LR as G1)
#   2) dev inference (data-dev: the 43 dev clips as the 183 official <= 12 s chunks), seeds SEEDS (0 1), rows:
#        official_offline   INIT, whole chunk (stream_infer --offline == echo_g.inference / export_predictions of INIT)
#        slide_none         not trained: echog_slide.py --audio prefix --text voiced --prefix none (INIT)
#        slide_inpaint      (optional, not in the default ROWS) same + --prefix inpaint
#        stream_prefix      trained: stream_infer.py --audio prefix (strictly causal: Wav2Vec2 re-run on the arrived waveform,
#                           EX002 pipeline = official 30 FPS extraction + the recorded 30 -> 50 FPS resampling)
#        stream_frozen      trained: stream_infer.py --audio frozen (training-time full-chunk features; leak control)
#        stream_offline     trained: stream_infer.py --offline (offline ability kept?)
#      + intermediate EMA snapshots (ema_step*.pt): stream_prefix seed 0 (training curve)
#      every chunk is generated independently (the offline cuts of the official EX002 export; a streaming row restarts cold at
#      each chunk); every (row, seed) is split into NSLOT = |GPUS| x SPG shards
#   3) eval-dev export (ex002_export.py == export_dev.py --retrained, + its audit; full runs also the frozen audit_exports.py),
#      EX002 FGD v2 score_dirs.py (2 evaluator seeds averaged; generation seed 0 per row, --seed1 = the same row's generation
#      seed 1; the seed-1 export is also scored as <row>_s1), block_stats.py --ex002-dev (joint space, seed 0)
#   4) SUMMARY.md / summary.json
# Env (defaults = DLC / CPFS):  GPUS="0 1 2 3"  SPG=3  STEPS=20000 LR=1e-4 BATCH=<per GPU, default 64/|GPUS|>  SEEDS="0 1"
#   STREAM_CFG (default {"cut_boost": 4.0}, as G1)  CLIP_BATCH=8  N=<first N dev clips (ci), smoke>  ROWS="..."  TAG (OUT=.../ex002-$TAG)
#   OUT INIT CONFIG DATA DEV W2V FGD PY TRAIN_ARGS (extra train_stream.py args)
#   TAG=confirm bash scripts/echog_stream/ex002_stream.sh
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
E=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
PKG=${PKG:-$E/reference/dlc-package-r2}                 # frozen EX002 adaptation package (official src + export/audit code)
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
TAG=${TAG:-confirm}
OUT=${OUT:-/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-1105/ex002-$TAG}
DATA=${DATA:-/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/echog-official/full-data-r1}
INIT=${INIT:-/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/echog-official/runs/dlc5ji8ln8tjz64w/final.pt}
DEV=${DEV:-$E/data-dev}; W2V=${W2V:-$E/models/wav2vec2-large-xlsr-53-english}
FGD=${FGD:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/fgd-ex002-v2-20261102/code}
GL=(${GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))}); NG=${#GL[@]}; SPG=${SPG:-3}; NSLOT=$((NG * SPG))   # SPG 3: a prefix-audio process held ~19 GB in the smoke
STEPS=${STEPS:-20000}; LR=${LR:-1e-4}; BATCH=${BATCH:-$((64 / NG))}; SEEDS=${SEEDS:-0 1}
ROWS=${ROWS:-official_offline slide_none stream_prefix stream_frozen stream_offline}
STREAM_CFG=${STREAM_CFG:-'{"cut_boost": 4.0}'}
CFG=(); [ -n "${CONFIG:-}" ] && CFG=(--config $CONFIG)    # default: the config embedded in INIT (train_ex002.py embeds it)
export PYTHONPATH=$PKG/official/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # GPUS = nvidia-smi indices
mkdir -p $OUT; T=$OUT/train; P=$OUT/preds; X=$OUT/eval-dev; B=$OUT/bench; M=$OUT/dev_meta; mkdir -p $P $X $B $M
log() { echo "$(date -Is) $*" | tee -a $OUT/confirm.log; }
[ -f "$INIT" ] || { log "INIT missing: $INIT"; exit 1; }

# ---- 0 dev stems (chunks of the first N clips) + lengths.csv (= the official ConditionDataset rule: frames = feature rows)
$PY - $DEV $M "${N:-0}" <<'EOF' || { log DEV_META_FAILED; exit 1; }
import json, sys, torch
from pathlib import Path
dev, m, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
ch = json.loads((dev / 'MANIFEST.json').read_text()); cis = sorted({c['ci'] for c in ch})[: n or None]
sel = [c for c in ch if c['ci'] in cis]; rows = []
for c in sel:
    x = torch.load(dev / 'condition_50fps' / f"{c['stem']}.pt", map_location='cpu', weights_only=True)
    F = len(x['audio_features']); assert F == c['frames'], (c['stem'], F, c['frames']); rows.append(f"{c['stem']},{F},")
(m / 'lengths.csv').write_text('stem,aligned_frames,error\n' + '\n'.join(rows) + '\n')
(m / 'stems.txt').write_text(''.join(c['stem'] + '\n' for c in sel)); (m / 'ci.txt').write_text(' '.join(map(str, cis)))
print(len(cis), 'clips', len(sel), 'chunks')
EOF
STEMS=$M/stems.txt; CIS=$(cat $M/ci.txt); CIARG=(); [ -n "${N:-}" ] && CIARG=(--ci $CIS)
log "stream $STREAM_CFG"; log "code $(git rev-parse HEAD 2>/dev/null || cat GIT_COMMIT 2>/dev/null) GPUS=${GL[*]} SPG=$SPG STEPS=$STEPS LR=$LR BATCH=$BATCH INIT=$INIT dev clips $(wc -w < $M/ci.txt) chunks $(wc -l < $STEMS)"

# ---- 1 training
if [ ! -f $T/final.pt ]; then
  log TRAIN_START
  CUDA_VISIBLE_DEVICES=$(echo ${GL[*]} | tr ' ' ,) $PY -m torch.distributed.run --nnodes=1 --nproc_per_node=$NG --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:${RDZV_PORT:-29638} scripts/echog_stream/train_stream.py --init $INIT "${CFG[@]}" --data-root $DATA --out $T \
    --profile ex002 --steps $STEPS --lr $LR --batch $BATCH --stream "$STREAM_CFG" ${TRAIN_ARGS:-} >> $OUT/train.log 2>&1
  [ -f $T/final.pt ] || { log TRAIN_FAILED; exit 1; }
fi
log TRAIN_DONE

# ---- 2 inference jobs "<name> <seed>", each sharded over all slots
SL=(--wav-dir $DEV/audio --audio-model $W2V)
COM=(--cond-dir $DEV/condition_50fps --meta $M --clip-batch ${CLIP_BATCH:-8})
job() {  # name seed
  local n=$1 sd=$2 o=$P/$1/seed_00$2
  case $n in
    official_offline) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $INIT ${CFG[*]} ${COM[*]} --stems $STEMS --out $o --seed $sd --offline";;
    slide_none|slide_inpaint) echo "$o scripts/echog_g1/echog_slide.py --checkpoint $INIT ${CFG[*]} --cond-dir $DEV/condition_50fps --meta $M --stems $STEMS --out $o --seed $sd --audio prefix --text voiced --prefix ${n#slide_} ${SL[*]}";;
    stream_prefix) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/final.pt ${COM[*]} --stems $STEMS --out $o --seed $sd --audio prefix ${SL[*]}";;
    stream_frozen) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/final.pt ${COM[*]} --stems $STEMS --out $o --seed $sd --audio frozen";;
    stream_offline) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/final.pt ${COM[*]} --stems $STEMS --out $o --seed $sd --offline";;
    snap_*) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/ema_${n#snap_}.pt ${COM[*]} --stems $STEMS --out $o --seed $sd --audio prefix ${SL[*]}";;
  esac
}
JOBS=()
for r in $ROWS; do for sd in $SEEDS; do JOBS+=("$r $sd"); done; done
SNAPS=$(ls $T 2>/dev/null | grep -o 'step[0-9]*' | sort -u)
for s in $SNAPS; do JOBS+=("snap_$s 0"); done
log "INFER_START ${#JOBS[@]} jobs x $NSLOT shards"
for i in $(seq 0 $((NSLOT - 1))); do
  ( g=${GL[$((i / SPG))]}
    for j in "${JOBS[@]}"; do set -- $j; spec=$(job $1 $2); o=${spec%% *}; cmd=${spec#* }; mkdir -p $o
      CUDA_VISIBLE_DEVICES=$g $PY $cmd --shard $i/$NSLOT >> $o.shard$i.log 2>&1 || echo "$(date -Is) SHARD_FAILED $1 $2 $i" >> $OUT/confirm.log
    done ) &
done; wait
log INFER_DONE

# ---- 3 export + evaluation
nst=$(wc -l < $STEMS); LAB=(); S1=(); PAN=(); done_rows=()
for r in $ROWS $(for s in $SNAPS; do echo snap_$s; done); do
  ok=1; for sd in $SEEDS; do [ "${r#snap_}" != "$r" ] && [ $sd != 0 ] && continue
    n=$(ls $P/$r/seed_00$sd 2>/dev/null | grep -c '\.pt$'); [ "$n" = "$nst" ] || { log "INCOMPLETE $r seed $sd: $n / $nst"; ok=0; }; done
  [ $ok = 1 ] || continue
  $PY scripts/echog_stream/ex002_export.py --native $P/$r --prefix $r --output $X "${CIARG[@]}" >> $B/export.log 2>&1 || { log "EXPORT_FAILED $r"; continue; }
  if [ -z "${N:-}" ] && [ -d $X/${r}_seed1 ]; then                     # all 43 clips: also the frozen package audit
    $PY $PKG/code/audit_exports.py --root $X --variants $r ${r}_seed1 >> $B/export.log 2>&1 && mv $X/EXPORT_AUDIT.json $X/EXPORT_AUDIT_frozen_$r.json || log "FROZEN_AUDIT_FAILED $r"
  fi
  LAB+=("$r=$X/$r"); done_rows+=($r)
  if [ -d $X/${r}_seed1 ]; then S1+=(--seed1 "$r=$X/${r}_seed1"); LAB+=("${r}_s1=$X/${r}_seed1"); fi
  [ "${r#snap_}" = "$r" ] && PAN+=("$r=$X/$r")
done
log "EXPORT_DONE ${done_rows[*]}"
( cd $FGD && CUDA_VISIBLE_DEVICES=${GL[0]} $PY score_dirs.py $B/fgd_v2.json "${LAB[@]}" "${S1[@]}" ) > $B/fgd_v2.log 2>&1 || log FGD_FAILED
$PY scripts/echog_stream/block_stats.py --ex002-dev --out $B/block_stats_seed0.json --workers ${STAT_WORKERS:-8} "${PAN[@]}" > $B/block_stats_seed0.txt 2>&1 || log BLOCK_STATS_FAILED
log EVAL_DONE

# ---- 4 summary
$PY - $OUT "${done_rows[*]}" <<'EOF' | tee $OUT/SUMMARY.md
import json, sys, glob
out, rows = sys.argv[1], sys.argv[2].split()
try: fg = json.load(open(f'{out}/bench/fgd_v2.json'))['methods']
except Exception: fg = {}
try: bs = json.load(open(f'{out}/bench/block_stats_seed0.json'))['labels']
except Exception: bs = {}
def lim(d):
    v = [json.load(open(p))['retarget_quality']['raw_limit_frame_fraction'] for p in glob.glob(f'{out}/eval-dev/{d}/meta_*.json')]
    return sum(v) / len(v) if v else float('nan')
f = lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else '-'
log = [l.split(' ', 1)[1] for l in open(f'{out}/confirm.log') if l.split(' ')[1:2] and l.split(' ')[1] in ('stream', 'code')]
print('# ECHO-G trained streaming on EX002 (dev 43 clips / official 12 s chunks)\n'); [print('-', l.strip()) for l in log[-2:]]
print('\nFGD = EX002 FGD v2 (mean of the 2 evaluator seeds [seed0, seed1]); BC / Diversity robot-space (FGD v2); Div s1 = same-clip '
      'diversity vs the generation seed 1; raw limit = share of frames outside the EX002 limits before clipping; accel / jerk @boundary '
      '= block-boundary ratio / GT ratio (joint space, 18 upper joints); beat phase max/min over 10 bins of the 320 ms period (1 = uniform)\n')
print('| row | gen seed | FGD | FGD evaluator seeds | BC (GT) | Div L1 | Div s1 | raw limit | accel@bnd vs GT | jerk@bnd vs GT | beat phase max/min |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
summ = {}
for r in rows:
    for lab, sd, d in ((r, 0, r), (r + '_s1', 1, r + '_seed1')):
        m = fg.get(lab, {}).get('all')
        if m is None:
            if lab == r: print(f'| {r} | {sd} | (missing) |')
            continue
        b = bs.get(r, {}) if sd == 0 else {}
        summ[lab] = dict(fgd=m, block=b, raw_limit=lim(d))
        print(f"| {r} | {sd} | {f(m['FGD_mean'])} | {' / '.join(f(x) for x in m['FGD_seeds'])} | {f(m['BC'])} ({f(m['BC_gt'])}) | "
              f"{f(m['Diversity_L1_rad'])} | {f(m.get('Diversity_seed1_L1_rad'))} | {f(lim(d))} | {f(b.get('accel_ratio_vs_gt'))} | "
              f"{f(b.get('jerk_ratio_vs_gt'))} | {f(b.get('beat_phase_maxmin'))} |")
if 'GT' in bs: print(f"\nGT: accel ratio {bs['GT']['accel_ratio']:.3f}, jerk ratio {bs['GT']['jerk_ratio']:.3f}, beat phase max/min {bs['GT']['beat_phase_maxmin']:.2f}")
n = next(iter(fg.values()), {}).get('all', {}).get('n')
print(f'\nclips scored: {n}')
json.dump(summ, open(f'{out}/summary.json', 'w'), indent=1)
EOF
log CONFIRM_DONE
