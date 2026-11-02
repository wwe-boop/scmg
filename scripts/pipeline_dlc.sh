#!/bin/bash
# SCMG full training + ablations on ONE DLC pod with NGPU (default 8) A800s.
#   pipeline_dlc.sh <code dir> <out root> <vae.pt>
# env: NGPU (8), SCALE (1.0: multiply every step count; selftest uses a tiny value), NEVAL (380 dev clips),
#      BATCH (128 per GPU), EXTRA (extra train args, e.g. --mmap for selftests)
# Every train/eval is skipped when its DONE.json / RESULTS.json exists (resubmission continues where it stopped).
#
#   base (all GPUs, DDP) --+--> g0 main      peft -> causal -> state
#                          +--> g1 nomtc     peft(mtc=false) -> causal          [Multi-Token Conditioning]
#                          +--> g2 nostyle   peft(style=false) -> causal        [Style Distribution Head]
#                          +--> g3 seed1     peft -> causal -> state (seed 1)   [seed variance]
#                          +--> g4 eval base, then nocat-causal (p_stream=0)    [Chunk-Aware Causal Training]
#                          +--> g5 noop-causal (no out-painting / continuity)   [Masked Out-painting]
#                          +--> g6 fullft-causal (--unfreeze-flow)              [PEFT vs limited unfreezing]
#                          +--> g7 measured-state (history = measured), then unfreeze-state   [Robot State Adapter]
set -uo pipefail
CODE=${1:?code dir}; R=${2:?out root}; VAE=${3:?vae}
NGPU=${NGPU:-8}; SCALE=${SCALE:-1.0}; NEVAL=${NEVAL:-120}; NFULL=${NFULL:-380}; NMID=${NMID:-40}; BATCH=${BATCH:-128}; EXTRA=${EXTRA:-}
PY=${PY:-python}
D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
TOKC=${TOKC:-}; BASECFG=${BASECFG:-{\}}          # VoxCPM-style planner: TOKC=<token cache>, BASECFG={"planner":"voxcpm"}
C="--stats $D/STATS_v36_1.json --text-cache $D/text-e5-v36_1 --vae $VAE --workers ${WORKERS:-6} $EXTRA ${TOKC:+--tok-cache $TOKC}"
cd $CODE && export PYTHONPATH=$CODE
mkdir -p $R
rm -f $R/*/FAILED                          # a resubmission retries lanes that failed in an earlier job
steps() { python3 -c "print(max(4, int($1 * $SCALE)))"; }
# lane -> GPU: index into the devices the job was GIVEN (CUDA_VISIBLE_DEVICES), never a bare 0..N-1 -- on a shared DSW a
# selftest launched with CUDA_VISIBLE_DEVICES=1 was remapped to physical GPU 0 (forbidden there) by the old version (11-02)
VISIBLE_DEVS=${CUDA_VISIBLE_DEVICES-}
DEVS=(${VISIBLE_DEVS//,/ }); [ ${#DEVS[@]} -eq 0 ] && DEVS=($(seq 0 $((NGPU - 1))))
g() { echo ${DEVS[$(( $1 % ${#DEVS[@]} ))]}; }

train() {  # gpu out args...
  local gpu=$1 o=$2; shift 2
  [ -f $R/$o/DONE.json ] && { echo "skip $o"; return 0; }
  [ -f $R/$o/FAILED ] && return 1
  mkdir -p $R/$o; echo "$(date -Is) start $o on GPU $gpu"
  if CUDA_VISIBLE_DEVICES=$gpu $PY -m scmg.train --out $R/$o $C --batch $BATCH "$@" >> $R/$o/log.txt 2>&1; then
    echo "$(date -Is) done $o"
  else
    echo "$(date -Is) FAILED $o"; touch $R/$o/FAILED; return 1
  fi
}
evaluate() {  # gpu dir n ckpt-name extra...
  local gpu=$1 o=$2 n=$3 ck=$4; shift 4
  local out=$R/$o/eval-${ck%.pt}${EVTAG:-}
  [ -f $out/RESULTS.json ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY -m scmg.evaluate --ckpt $R/$o/$ck --vae $VAE --stats $D/STATS_v36_1.json \
    --text-cache $D/text-e5-v36_1 ${TOKC:+--tok-cache $TOKC} --cfg-w ${CFGW:-2.5} --out $out --n $n "$@" $( [[ "$EXTRA" == *--mmap* ]] && echo --mmap ) >> $R/$o/eval.log 2>&1 \
    || echo "$(date -Is) eval FAILED $o/$ck"
}
wait_for() {  # dir : returns 1 if upstream failed
  until [ -f $R/$1/DONE.json ]; do [ -f $R/$1/FAILED ] && return 1; sleep 30; done
}
mid_eval() {  # gpu dir until-file : evaluate every new snapshot (mid-training evaluation, ~every 5000 steps)
  local gpu=$1 o=$2
  while true; do
    for s in $R/$o/snap_*.pt; do [ -f "$s" ] && evaluate $gpu $o $NMID $(basename $s) --horizons inf,0; done
    [ -f $R/$o/DONE.json ] || [ -f $R/$o/FAILED ] && break
    sleep 120
  done
}

# ---------------- runtime check (CUDA really computes on every GPU)
$PY - <<'EOF' || exit 2
import torch, sys
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    x = torch.randn(1024, 1024, device=f'cuda:{i}'); y = (x @ x).sum().item()
    assert y == y, 'nan'
print('CUDA_COMPUTE_OK', flush=True)
EOF

# ---------------- VoxCPM module ablation (VOXABL=1): base stage only, one GPU per lane, identical budget; which parts of
# the VoxCPM-style planner make it work (pretrained text LM / FSQ skeleton / residual LM / LocEnc / text in the sequence),
# against the chunk planner at the same budget and a second seed of the full design (noise floor).
# ---------------- Qwen text-LM ablation (VOXQ=1): same budget as VOXABL; each lane with its own LM and token cache.
# 11-02 VOXABL finding: Qwen2.5 RANDOM init (FKD 1.25 / speed 0.93) and NO text (1.75 / 0.68) beat the pretrained LM
# (4.62 / 0.42) -> lanes test whether a newer pretrained LM changes that, and whether the cause is pretraining or text.
# ---------------- VoxCPM component ablation around the BEST base (VOXABL2=1, 11-03): VOXABL removed components from the
# pretrained-Qwen base, which is itself poor (4.62); here every lane starts from the random-init LM (vox-randlm, 1.25 / 0.93)
# so each component's effect is measured on a working model. Seed noise: vox-randlm (seed 0, VOXABL), qwen25-random-seed1
# (VOXQ), r-seed2 here; chunk-planner-seed1 gives the chunk planner's seed noise.
if [ "${VOXABL2:-0}" = 1 ]; then
  VB="--stage base --steps $(steps 8000) --lr 3e-4 --warmup $(steps 1000) --eval-every $(steps 500) --snap-every 0"
  V='"planner": "voxcpm", "vox_lm_init": "random"'
  vlane() {  # gpu dir cfg-json extra...
    local gpu=$1 o=$2 cf=$3; shift 3
    train $gpu $o $VB --cfg "$cf" "$@" && CFGW=2.0 evaluate $gpu $o $NEVAL best.pt --horizons inf,0
  }
  L4=()
  ( vlane $(g 0) r-lmlr1 "{$V, \"lm_lr_scale\": 1.0}" ) & L4+=($!)
  ( vlane $(g 1) r-nofsq "{$V, \"vox_fsq\": false}" ) & L4+=($!)
  ( vlane $(g 2) r-noralm "{$V, \"vox_ralm\": false}" ) & L4+=($!)
  ( vlane $(g 3) r-nolocenc "{$V, \"vox_locenc\": false}" ) & L4+=($!)
  ( vlane $(g 4) r-notext "{$V, \"vox_text\": false}" ) & L4+=($!)
  ( vlane $(g 5) r-fsqtanh "{$V, \"vox_fsq\": \"tanh\"}" ) & L4+=($!)
  ( vlane $(g 6) r-seed2 "{$V}" --seed 2 ) & L4+=($!)
  ( vlane $(g 7) chunk-planner-seed1 "{}" --seed 1 ) & L4+=($!)
  wait "${L4[@]}"
  $PY -m scmg.summarize $R > $R/SUMMARY.md 2>&1 || true
  echo "$(date -Is) VOXABL2_DONE"; touch $R/VOXABL2_DONE; exit 0
fi
# ---------------- Text-use ablation (TEXTUSE=1, 11-03): user "让文本被利用". Code review: pretrained LM saturates the FSQ
# input (57% at init, 38% trained) and evaluation fed misaligned LM text (fixed: --window-lm-text). Literature: SANA (RMSNorm +
# scale on LLM features), ConvoFusion (modality dropout / per-modality guidance). Lanes isolate interface fixes, LM
# freezing and audio dropout; every lane is evaluated with windowed LM text, plus a separate text-guidance (cfg_t 3) pass.
if [ "${TEXTUSE:-0}" = 1 ]; then
  VB="--stage base --steps $(steps 8000) --lr 3e-4 --warmup $(steps 1000) --eval-every $(steps 500) --snap-every 0"
  V='"planner": "voxcpm"'; NM='"vox_prefsq_norm": true, "vox_in_match": true'
  tlane() {  # gpu dir cfg-json extra...
    local gpu=$1 o=$2 cf=$3; shift 3
    train $gpu $o $VB --cfg "$cf" "$@" && SCMG_SAVE_N=43 CFGW=2.0 evaluate $gpu $o $NEVAL best.pt --horizons inf,0 --window-lm-text \
      && SCMG_SAVE_N=43 CFGW=2.0 EVTAG=-cfgt3 evaluate $gpu $o $NEVAL best.pt --horizons inf --window-lm-text --cfg-t 3.0
  }
  L5=()
  ( tlane $(g 0) t-pre-norm "{$V, \"vox_prefsq_norm\": true}" ) & L5+=($!)
  ( tlane $(g 1) t-pre-normmatch "{$V, $NM}" ) & L5+=($!)
  ( tlane $(g 2) t-pre-normmatch-frozen "{$V, $NM, \"lm_lr_scale\": 0.0}" ) & L5+=($!)
  ( tlane $(g 3) t-pre-normmatch-da3 "{$V, $NM, \"drop_audio\": 0.3}" ) & L5+=($!)
  ( tlane $(g 4) t-rand-normmatch "{$V, $NM, \"vox_lm_init\": \"random\"}" ) & L5+=($!)
  ( tlane $(g 5) t-rand-da3 "{$V, \"vox_lm_init\": \"random\", \"drop_audio\": 0.3}" ) & L5+=($!)
  ( tlane $(g 6) t-rand-normmatch-da3 "{$V, $NM, \"vox_lm_init\": \"random\", \"drop_audio\": 0.3}" ) & L5+=($!)
  ( tlane $(g 7) t-pre-normmatch-seed1 "{$V, $NM}" --seed 1 ) & L5+=($!)
  wait "${L5[@]}"
  $PY -m scmg.summarize $R > $R/SUMMARY.md 2>&1 || true
  echo "$(date -Is) TEXTUSE_DONE"; touch $R/TEXTUSE_DONE; exit 0
fi
if [ "${VOXQ:-0}" = 1 ]; then
  M=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/models
  VB="--stage base --steps $(steps 8000) --lr 3e-4 --warmup $(steps 1000) --eval-every $(steps 500) --snap-every 0"
  qlane() {  # gpu dir tokcache cfg-json extra...
    local gpu=$1 o=$2 tc=$3 cf=$4; shift 4
    train $gpu $o $VB --tok-cache $tc --cfg "$cf" "$@" && TOKC=$tc CFGW=2.0 evaluate $gpu $o $NEVAL best.pt --horizons inf,0
  }
  Q2=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/tok-qwen25-v36_1; Q3=$M/tok-qwen3-0.6b-v36_1
  L3=()
  ( qlane $(g 0) qwen3-pretrained $Q3 "{\"planner\": \"voxcpm\", \"vox_lm\": \"$M/Qwen3-0.6B-Base\"}" ) & L3+=($!)
  ( qlane $(g 1) qwen3-random $Q3 "{\"planner\": \"voxcpm\", \"vox_lm\": \"$M/Qwen3-0.6B-Base\", \"vox_lm_init\": \"random\"}" ) & L3+=($!)
  ( qlane $(g 2) qwen25-pretrained-lmlr1 $Q2 "{\"planner\": \"voxcpm\", \"lm_lr_scale\": 1.0}" ) & L3+=($!)
  ( qlane $(g 3) qwen25-random-seed1 $Q2 "{\"planner\": \"voxcpm\", \"vox_lm_init\": \"random\"}" --seed 1 ) & L3+=($!)
  wait "${L3[@]}"
  $PY -m scmg.summarize $R > $R/SUMMARY.md 2>&1 || true
  echo "$(date -Is) VOXQ_DONE"; touch $R/VOXQ_DONE; exit 0
fi

if [ "${VOXABL:-0}" = 1 ]; then
  VB="--stage base --steps $(steps 8000) --lr 3e-4 --warmup $(steps 1000) --eval-every $(steps 500) --snap-every 0"
  V='"planner": "voxcpm"'
  vlane() {  # gpu dir cfg-json extra...
    local gpu=$1 o=$2 cf=$3; shift 3
    train $gpu $o $VB --cfg "$cf" "$@" && CFGW=2.0 evaluate $gpu $o $NEVAL best.pt --horizons inf,0
  }
  L2=()
  ( vlane $(g 0) vox-full "{$V}" ) & L2+=($!)
  ( vlane $(g 1) vox-randlm "{$V, \"vox_lm_init\": \"random\"}" ) & L2+=($!)
  ( vlane $(g 2) vox-nofsq "{$V, \"vox_fsq\": false}" ) & L2+=($!)
  ( vlane $(g 3) vox-noralm "{$V, \"vox_ralm\": false}" ) & L2+=($!)
  ( vlane $(g 4) vox-nolocenc "{$V, \"vox_locenc\": false}" ) & L2+=($!)
  ( vlane $(g 5) vox-notext "{$V, \"vox_text\": false}" ) & L2+=($!)
  ( vlane $(g 6) chunk-planner "{}" ) & L2+=($!)
  ( vlane $(g 7) vox-full-seed1 "{$V}" --seed 1
    vlane $(g 7) vox-fsqtanh "{$V, \"vox_fsq\": \"tanh\"}" ) & L2+=($!)
  wait "${L2[@]}"
  $PY -m scmg.summarize $R > $R/SUMMARY.md 2>&1 || true
  echo "$(date -Is) VOXABL_DONE"; touch $R/VOXABL_DONE; exit 0
fi

# ---------------- stage 0: motion VAE (only when the given VAE does not exist yet, e.g. the 25 Hz ablation)
if [ ! -f $VAE ]; then
  vd=$(basename $(dirname $VAE))
  train $(g 0) $vd --stage vae --steps $(steps 30000) --batch 128 --lr 1e-3 --warmup $(steps 500) \
    --deriv-ramp $(steps 3000) --snap-every 0 || exit 1
  CUDA_VISIBLE_DEVICES=$(g 0) $PY scripts/check_vae.py $VAE > $R/$vd/CHECK_VAE.json 2>> $R/$vd/log.txt || exit 1
  cat $R/$vd/CHECK_VAE.json
fi

# ---------------- base: all GPUs, DDP
if [ "${ONLY_ADSC:-0}" = 1 ]; then
  wait_for base || { echo 'base FAILED upstream'; exit 1; }
elif [ ! -f $R/base/DONE.json ]; then
  mkdir -p $R/base; echo "$(date -Is) start base on $NGPU GPUs"
  ( mid_eval $(g 7) base ) & BASEWATCH=$!          # keeps evaluating snapshots in the background; never blocks lanes
  if [ $NGPU -gt 1 ]; then
    $PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU -m scmg.train --stage base --out $R/base $C --cfg "$BASECFG" \
      --batch $BATCH --steps $(steps 8000) --lr 3e-4 --warmup $(steps 1000) --eval-every $(steps 500) --snap-every $(steps ${SNAP:-2000}) >> $R/base/log.txt 2>&1
  else
    CUDA_VISIBLE_DEVICES=$(g 0) $PY -m scmg.train --stage base --out $R/base $C --cfg "$BASECFG" \
      --batch $BATCH --steps $(steps 8000) --lr 3e-4 --warmup $(steps 1000) --eval-every $(steps 500) --snap-every $(steps ${SNAP:-2000}) >> $R/base/log.txt 2>&1
  fi
  [ -f $R/base/DONE.json ] || { echo "base FAILED"; touch $R/base/FAILED; kill $BASEWATCH 2>/dev/null; exit 1; }
fi

PE="--steps $(steps 6000) --lr 1e-4 --warmup $(steps 300) --eval-every $(steps 500) --snap-every $(steps ${SNAP:-2000})"
CA="--steps $(steps 8000) --lr 1e-4 --warmup $(steps 300) --eval-every $(steps 500) --snap-every $(steps ${SNAP:-2000})"
ST="--steps $(steps 6000) --lr 1e-4 --warmup $(steps 300) --eval-every $(steps 500) --snap-every $(steps ${SNAP:-2000})"
B=$R/base/best.pt
LANES=()
# ---------------- ADSC (Asynchronous Dual-Stream Conditioning) ablation. Modes: offline (all audio + text + timing),
# online text-final, online text-streaming (X2Streaming-TTS token commitment, or whole TTS segments); online samples
# see token timing only once voiced. adsc = masks + stream-state token, adscmask = masks only, control = main-causal.
# Online evals: audio horizon 2 (160 ms; robot TTS audio arrives 0.04-0.15 s before playback) and 0.
lead_evals() {  # gpu dir : online modes, token times only once voiced (deployment information)
  EVTAG=-onfinal evaluate $1 $2 $NEVAL best.pt --horizons 2,0 --deploy-timing
  EVTAG=-x2 evaluate $1 $2 $NEVAL best.pt --horizons 2,0 --deploy-timing --text-mode x2 --x2-rate 15 --x2-lag 0.15
  EVTAG=-seg evaluate $1 $2 $NEVAL best.pt --horizons 2,0 --deploy-timing --text-mode segment --seg-lead 0.3
}
ADSC_SC='{"p_async": 0.5}'
adsc_lanes() {  # gA gB gC
  ( wait_for main-peft && train $1 adsc-causal --stage causal --init $R/main-peft/best.pt --cfg '{"adsc": true}' --sc "$ADSC_SC" $CA \
      && evaluate $1 adsc-causal $NEVAL best.pt --horizons inf,4,0 && lead_evals $1 adsc-causal ) & LANES+=($!)
  ( wait_for main-peft && train $2 adscmask-causal --stage causal --init $R/main-peft/best.pt --sc "$ADSC_SC" $CA \
      && evaluate $2 adscmask-causal $NEVAL best.pt --horizons inf,4,0 && lead_evals $2 adscmask-causal ) & LANES+=($!)
  ( wait_for main-causal && lead_evals $3 main-causal ) & LANES+=($!)
}
if [ "${ONLY_ADSC:-0}" = 1 ]; then         # add-on job for a pipeline that is already running elsewhere (same R)
  adsc_lanes $(g 0) $(g 1) $(g 2)
  wait "${LANES[@]}"; echo "$(date -Is) ADSC_DONE"; touch $R/ADSC_DONE; exit 0
fi
( train $(g 0) main-peft --stage peft --init $B $PE && evaluate $(g 0) main-peft $NEVAL best.pt --horizons inf,4,0 \
  && train $(g 0) main-causal --stage causal --init $R/main-peft/best.pt $CA && evaluate $(g 0) main-causal $NFULL best.pt --horizons inf,8,4,2,0 \
  && train $(g 0) main-state --stage state --init $R/main-causal/best.pt $ST && evaluate $(g 0) main-state $NFULL best.pt --horizons 4,0 --closed-loop ) & LANES+=($!)
( train $(g 1) nomtc-peft --stage peft --init $B --cfg '{"mtc": false}' $PE \
  && train $(g 1) nomtc-causal --stage causal --init $R/nomtc-peft/best.pt --cfg '{"mtc": false}' $CA && evaluate $(g 1) nomtc-causal $NEVAL best.pt --horizons inf,4,0 ) & LANES+=($!)
( train $(g 2) nostyle-peft --stage peft --init $B --cfg '{"style": false}' $PE \
  && train $(g 2) nostyle-causal --stage causal --init $R/nostyle-peft/best.pt --cfg '{"style": false}' $CA && evaluate $(g 2) nostyle-causal $NEVAL best.pt --horizons inf,4,0 ) & LANES+=($!)
( train $(g 3) seed1-peft --stage peft --init $B --seed 1 $PE \
  && train $(g 3) seed1-causal --stage causal --init $R/seed1-peft/best.pt --seed 1 $CA && evaluate $(g 3) seed1-causal $NEVAL best.pt --horizons inf,4,0 \
  && train $(g 3) seed1-state --stage state --init $R/seed1-causal/best.pt --seed 1 $ST && evaluate $(g 3) seed1-state $NEVAL best.pt --horizons 4 --closed-loop ) & LANES+=($!)
( evaluate $(g 4) base $NEVAL best.pt --horizons inf,8,4,0; wait_for main-peft \
  && train $(g 4) nocat-causal --stage causal --init $R/main-peft/best.pt --sc '{"p_stream": 0.0}' $CA && evaluate $(g 4) nocat-causal $NEVAL best.pt --horizons inf,8,4,2,0 ) & LANES+=($!)
( wait_for main-peft \
  && train $(g 5) noop-causal --stage causal --init $R/main-peft/best.pt --sc '{"outpaint": false, "w_cont": 0.0}' $CA && evaluate $(g 5) noop-causal $NEVAL best.pt --horizons inf,4,0 ) & LANES+=($!)
( wait_for noop-causal; wait_for main-peft && train $(g 5) adsc-causal --stage causal --init $R/main-peft/best.pt --cfg '{"adsc": true}' --sc "$ADSC_SC" $CA \
  && evaluate $(g 5) adsc-causal $NEVAL best.pt --horizons inf,4,0 && lead_evals $(g 5) adsc-causal ) & LANES+=($!)
( wait_for main-peft \
  && train $(g 6) fullft-causal --stage causal --init $R/main-peft/best.pt --unfreeze-flow $CA && evaluate $(g 6) fullft-causal $NEVAL best.pt --horizons inf,4,0 ) & LANES+=($!)
( wait_for fullft-causal; wait_for main-peft && train $(g 6) adscmask-causal --stage causal --init $R/main-peft/best.pt --sc "$ADSC_SC" $CA \
  && evaluate $(g 6) adscmask-causal $NEVAL best.pt --horizons inf,4,0 && lead_evals $(g 6) adscmask-causal ) & LANES+=($!)
( wait_for nocat-causal; wait_for main-causal && lead_evals $(g 4) main-causal ) & LANES+=($!)
( wait_for main-causal \
  && train $(g 7) measured-state --stage state --init $R/main-causal/best.pt --sc '{"state_hist_measured": true}' $ST && evaluate $(g 7) measured-state $NEVAL best.pt --horizons 4 --closed-loop
  wait_for main-causal && train $(g 7) unfreeze-state --stage state --init $R/main-causal/best.pt --unfreeze-flow $ST && evaluate $(g 7) unfreeze-state $NEVAL best.pt --horizons 4,0 --closed-loop ) & LANES+=($!)
WATCH=()
for o in main-peft main-causal main-state; do ( mid_eval $(g 0) $o ) & WATCH+=($!); done
wait "${LANES[@]}"
kill "${WATCH[@]}" ${BASEWATCH:-} 2>/dev/null
$PY -m scmg.summarize $R > $R/SUMMARY.md 2>&1 || true
echo "$(date -Is) PIPELINE_DONE"; touch $R/PIPELINE_DONE
