#!/bin/bash
# ECHO-G trained streaming variant, G1 confirmation in ONE DLC pod (11-05):
#   1) train_stream.py from the official best.pt (DDP on all GPUS, STEPS / LR)
#   2) val_common inference, inference seeds SEEDS (0 1), rows:
#        official_offline   official best.pt, whole clip (stream_infer --offline == echo_g.inference, test_stream 2a/2d)
#        slide_none         not trained: echog_slide.py --audio prefix --text voiced --prefix none   (official model)
#        slide_inpaint      not trained: same + --prefix inpaint (RePaint-style committed frames)
#        official_stream    not trained: stream_infer.py protocol (ctx 8 s, tail) with the official best.pt (zero known_projection)
#        stream_prefix      trained: stream_infer.py --audio prefix (strictly causal, default)
#        stream_frozen      trained: stream_infer.py --audio frozen (training-time features; leak control)
#        stream_offline     trained: stream_infer.py --offline (offline ability kept?)
#      every (row, seed) is split into NSLOT = |GPUS| x SPG shards, slot s runs shard s of every job (balanced, GPUs full)
#   3) official evaluator per (row, seed); block_stats.py (accel / jerk at block boundaries vs GT, beat phase) and
#      scripts/echog_g1/beat_phase.py per row (seed 0); intermediate EMA snapshots (ema_step*.pt): stream_prefix seed 0 on the
#      first SNAP_N val stems + evaluator (training curve)
#   4) SUMMARY.md / summary.json
# Env (defaults = DLC / CPFS):  GPUS="0 1 2 3"  SPG=3  STEPS=20000 LR=1e-4 BATCH=<per GPU, default 64/|GPUS|>  SEEDS="0 1"
#   CLIP_BATCH=8 (stream_infer lockstep clips)  N=<first N val stems, smoke>  ROWS="..."  OUT  DATA  OFF  INIT  W2V  FGD_CKPT  PY  TRAIN_ARGS (extra train_stream.py args)
#   bash scripts/echog_stream/g1_confirm.sh
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
E=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
PY=${PY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}
OUT=${OUT:-/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-1105/g1-confirm}
DATA=${DATA:-/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/echog-stream-1105/data_root}
OFF=${OFF:-$E/official}; INIT=${INIT:-$E/weights/best.pt}; W2V=${W2V:-$E/models/wav2vec2-large-xlsr-53-english}
FGD_CKPT=${FGD_CKPT:-$E/weights/evaluation/g1_fgd_encoder.pt}; MMAE=${MMAE:-$DATA/eval_assets/mmae/g1_mmae_30body_30fps.npy}
GL=(${GPUS:-$(seq 0 $(($(nvidia-smi -L | wc -l) - 1)))}); NG=${#GL[@]}; SPG=${SPG:-3}; NSLOT=$((NG * SPG))
STEPS=${STEPS:-20000}; LR=${LR:-1e-4}; BATCH=${BATCH:-$((64 / NG))}; SEEDS=${SEEDS:-0 1}
ROWS=${ROWS:-official_offline slide_none slide_inpaint official_stream stream_prefix stream_frozen stream_offline}
SNAP_N=${SNAP_N:-300}; EVAL_PAR=${EVAL_PAR:-6}
export PYTHONPATH=$OFF/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
mkdir -p $OUT; T=$OUT/train; P=$OUT/preds; B=$OUT/bench; mkdir -p $P $B
log() { echo "$(date -Is) $*" | tee -a $OUT/confirm.log; }
STEMS=$OUT/stems.txt
if [ -n "${N:-}" ]; then head -n $N $DATA/splits/val_common.txt > $STEMS; else cp $DATA/splits/val_common.txt $STEMS; fi
head -n $SNAP_N $STEMS > $OUT/stems_snap.txt
log "code $(git rev-parse HEAD 2>/dev/null || cat GIT_COMMIT 2>/dev/null) GPUS=${GL[*]} SPG=$SPG STEPS=$STEPS LR=$LR BATCH=$BATCH stems $(wc -l < $STEMS)"

# ---- 1 training
if [ ! -f $T/final.pt ]; then
  log TRAIN_START
  CUDA_VISIBLE_DEVICES=$(echo ${GL[*]} | tr ' ' ,) $PY -m torch.distributed.run --nnodes=1 --nproc_per_node=$NG --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:${RDZV_PORT:-29637} scripts/echog_stream/train_stream.py --init $INIT --config ${CONFIG:-$OFF/configs/sgdit_audio_text.yaml} --data-root $DATA --out $T \
    --profile g1 --steps $STEPS --lr $LR --batch $BATCH ${TRAIN_ARGS:-} >> $OUT/train.log 2>&1
  [ -f $T/final.pt ] || { log TRAIN_FAILED; exit 1; }
fi
log TRAIN_DONE

# ---- 2 inference jobs: "<name> <seed> <stems> <cmd...>" ; each sharded over all slots
SL=(--wav-dir $DATA/audio --audio-model $W2V)
COM=(--cond-dir $DATA/condition_30fps --meta $DATA --clip-batch ${CLIP_BATCH:-8})
job() {  # name seed stems
  local n=$1 sd=$2 st=$3 o=$P/$1/seed_00$2
  case $n in
    official_offline) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $INIT --config $OFF/configs/sgdit_audio_text.yaml ${COM[*]} --stems $st --out $o --seed $sd --offline";;
    slide_none|slide_inpaint) echo "$o scripts/echog_g1/echog_slide.py --checkpoint $INIT --config $OFF/configs/sgdit_audio_text.yaml --cond-dir $DATA/condition_30fps --meta $DATA --stems $st --out $o --seed $sd --audio prefix --text voiced --prefix ${n#slide_} ${SL[*]}";;
    official_stream) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $INIT --config $OFF/configs/sgdit_audio_text.yaml ${COM[*]} --stems $st --out $o --seed $sd --audio prefix ${SL[*]}";;
    stream_prefix) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/final.pt ${COM[*]} --stems $st --out $o --seed $sd --audio prefix ${SL[*]}";;
    stream_frozen) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/final.pt ${COM[*]} --stems $st --out $o --seed $sd --audio frozen";;
    stream_offline) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/final.pt ${COM[*]} --stems $st --out $o --seed $sd --offline";;
    snap_*) echo "$o scripts/echog_stream/stream_infer.py --checkpoint $T/ema_${n#snap_}.pt ${COM[*]} --stems $st --out $o --seed $sd --audio prefix ${SL[*]}";;
  esac
}
JOBS=()
for r in $ROWS; do for sd in $SEEDS; do JOBS+=("$r $sd $STEMS"); done; done
for s in $(ls $T 2>/dev/null | grep -o 'step[0-9]*' | sort -u); do JOBS+=("snap_$s 0 $OUT/stems_snap.txt"); done
log "INFER_START ${#JOBS[@]} jobs x $NSLOT shards"
for i in $(seq 0 $((NSLOT - 1))); do
  ( g=${GL[$((i / SPG))]}
    for j in "${JOBS[@]}"; do set -- $j; spec=$(job $1 $2 $3); o=${spec%% *}; cmd=${spec#* }; mkdir -p $o
      CUDA_VISIBLE_DEVICES=$g $PY $cmd --shard $i/$NSLOT >> $o.shard$i.log 2>&1 || echo "$(date -Is) SHARD_FAILED $1 $2 $i" >> $OUT/confirm.log
    done ) &
done; wait
log INFER_DONE

# ---- 3 evaluation
nst=$(wc -l < $STEMS); EV=()
for j in "${JOBS[@]}"; do set -- $j; o=$P/$1/seed_00$2; n=$(ls $o 2>/dev/null | grep -c '\.pt$'); m=$(wc -l < $3)
  [ "$n" = "$m" ] || { log "INCOMPLETE $1 seed $2: $n / $m"; continue; }
  EV+=("$o $3 $1_seed$2 $B/$1_seed_00$2.json")
done
i=0
for e in "${EV[@]}"; do set -- $e; [ -f $4 ] && continue
  ( PY=$PY REF_DIR=$DATA/motion_39d_30fps WAV_DIR=$DATA/audio MMAE=$MMAE FGD_CKPT=$FGD_CKPT OFF=$OFF CUDA_VISIBLE_DEVICES=${GL[$((i % NG))]} \
      bash scripts/echog_g1/bench.sh eval $1 $2 $3 $4 > $B/$3.log 2>&1 || echo "$(date -Is) EVAL_FAILED $3" >> $OUT/confirm.log ) &
  i=$((i + 1)); [ $((i % EVAL_PAR)) = 0 ] && wait
done; wait
PAN=(); for r in $ROWS; do [ -d $P/$r/seed_000 ] && PAN+=("$r=$P/$r/seed_000"); done
$PY scripts/echog_stream/block_stats.py --ref-dir $DATA/motion_39d_30fps --stems $STEMS --mmae-file $MMAE --out $B/block_stats_seed0.json \
  --workers ${STAT_WORKERS:-16} "${PAN[@]}" > $B/block_stats_seed0.txt 2>&1 || log BLOCK_STATS_FAILED
$PY scripts/echog_g1/beat_phase.py --mmae-file $MMAE --ref-dir $DATA/motion_39d_30fps --stems $STEMS "${PAN[@]}" > $B/beat_phase_seed0.txt 2>&1 || log BEAT_PHASE_FAILED
log EVAL_DONE

# ---- 4 summary
$PY - $OUT "$ROWS" "$SEEDS" <<'EOF' | tee $OUT/SUMMARY.md
import json, sys, glob, os
out, rows, seeds = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
def met(p):
    try: m = json.load(open(p))['metrics']
    except Exception: return None
    return m
bs = {}
try: bs = json.load(open(f'{out}/bench/block_stats_seed0.json'))['labels']
except Exception: pass
summ = {}
print('| row | seed | FGD | BA (GT) | Div (GT) | Jerk (GT) | accel@boundary vs GT | jerk@boundary vs GT | beat phase max/min |')
print('|---|---|---|---|---|---|---|---|---|')
def pick(m, *names):
    for n in names:
        v = m.get(n)
        if isinstance(v, dict): v = v.get('mean')
        if v is not None: return v
    return float('nan')
for r in rows + sorted({os.path.basename(p).rsplit('_seed', 1)[0] for p in glob.glob(f'{out}/bench/snap_*.json')}):
    for s in (seeds if not r.startswith('snap_') else ['0']):
        m = met(f'{out}/bench/{r}_seed_00{s}.json')
        if m is None: print(f'| {r} | {s} | (missing) |'); continue
        b = bs.get(r, {}) if s == seeds[0] else {}
        summ[f'{r}_seed{s}'] = dict(metrics=m, block=b)
        f = lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else '-'
        print(f"| {r} | {s} | {f(pick(m, 'FGD'))} | {f(pick(m, 'BA'))} ({f(pick(m, 'BA_GT'))}) | {f(pick(m, 'Div'))} ({f(pick(m, 'Div_GT'))}) | "
              f"{f(pick(m, 'Jerk'))} ({f(pick(m, 'Jerk_GT'))}) | {f(b.get('accel_ratio_vs_gt'))} | {f(b.get('jerk_ratio_vs_gt'))} | {f(b.get('beat_phase_maxmin'))} |")
if 'GT' in bs: print(f"\nGT: accel ratio {bs['GT']['accel_ratio']:.3f}, jerk ratio {bs['GT']['jerk_ratio']:.3f}, beat phase max/min {bs['GT']['beat_phase_maxmin']:.2f}")
json.dump(summ, open(f'{out}/summary.json', 'w'), indent=1)
EOF
log CONFIRM_DONE
