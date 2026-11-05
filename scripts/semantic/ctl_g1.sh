#!/bin/bash
# 11-06 G1 control / intent tokens + text-modality modules (dev side only: held-out-speaker view, 400-clip unseen-speaker dev).
# Every lane: hos-base warm start, the sf0 recipe (stream training, p_async 1, p_async_seg 0, self_force_p 0), STEPS (2000), then
# the official benchmark offline + engine WITHOUT tokens (r4_lane.sh, outputs saved at 50 Hz too). Afterwards per lane:
#   ctl lanes (cfg ctl): official benchmark offline with tokens: gt / ann / lex / auto (+ engine gt / auto); control-only runs
#                        on the token clips (shuffle, amp0, amp2, shift, point, gt_novals, gt + cfg_ctl CFGCTL); control metrics
#                        (scripts/semantic/g1_ctl_eval.py: STG, amplitude hit, +-0.3 s action rate, |dq| vs the no-token run)
#   CFGT_LANES : cfg_t sweep (separate text guidance, inference only) offline: CFGT values
#   TS_LANES   : text sensitivity (scripts/echog_g1/text_sensitivity.py: text-swap and no-text vs inference seed 1) and the semantic-window
#                text-vs-audio contribution (scripts/semantic/sem_window_sens.py; no-text AND no-audio runs, 11-06 criterion 1);
#                with CFGT_LANES also per cfg_t value (+ its seed-1 run)
#   cfg: CT / CP / SH (sem_head + sem_cond) / FT (flow_text_xattn) / CTFT / empty (adsc)
#   every official-benchmark output: semantic recognition pair_win (3 recognisers rec-sem-hos-seed*, trained on the train split)
# 'ctrl' = the existing sf0 dev model (no training, INIT_CTRL). Lanes (LANES, space-separated label|cfg|sc-fragment|data, cfg:
#   CT = adsc + ctl + ctl_flow, CP = adsc + ctl (planner text-stream route only), empty = adsc; sc fragment = comma-joined JSON members):
#   data flags: c = control tokens (G1_CTL), g = BEAT2 g_k track (G1_GEST, for p_sel_audio), f = ctl_focus CTL_FOCUS
#   ctl_g1.sh <out root>     env GPUS (one per lane), PY, BENCH_PY, EG/OFF/... as bench.sh, STEPS, LANES, CFGT_LANES, TS_LANES
set -uo pipefail
O=$1; CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
DEC=${DEC:-$B/echog-g1-decft-1105/decft-w3d0m05/final.pt}
A=$B/echog-g1-hos-assets; HV=$B/views/echog-g1-v1-hos; SB=${SEMREC:-$B/semantic-baseline-1105/g1}
CTL=${CTL:-$B/control-token-1105/ctl-tokens-hos}; GEST=${GEST:-$B/models/gest-beat2sem-echog-g1-v1-hos}
INIT=${INIT:-$B/echog-g1-arch-1104/hos-base/best.pt}; INIT_CTRL=${INIT_CTRL:-$B/echog-g1-sf-1105/dev/sf0/final.pt}
VENV=(G1_VIEW=$HV G1_BENCH_STEMS=$A/dev_stems.txt REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt
      VIEW_TEST=$HV VSPLIT=val TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1)
S='"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "self_force_p": 0'
GL=(${GPUS:-0 1 2 3 4 5 6}); R=$O/dev
export PROFILE=g1 G1_BENCH=1 G1_BENCH_SEEDS=0 SNAP=1000000 TRAIN_SEED=0
PYX=${PY:-python}
log() { echo "$(date -Is) $*" >> $O/ctl.log; }
bench() {   # gpu ckpt outdir proto seed extra...   (official benchmark; existing benchmark json = skip)
  local g=$1 ck=$2 P=$3 proto=$4 sd=$5; shift 5
  [ -f $(dirname $P)/benchmark_$(basename $P).json ] && return 0
  env "${VENV[@]}" PY=${BENCH_PY:-$PYX} GPUS=$g CFGW=${CFGW:-2.5} INFER_EXTRA="$*" bash scripts/echog_g1/bench.sh scmg $ck $DEC $P $proto $sd $A/dev_stems.txt \
    >> $O/bench-$(basename $(dirname $(dirname $P)))-$(basename $(dirname $P)).log 2>&1 || log "BENCH_FAILED $P"
}
infer() {   # gpu ckpt outdir stems proto extra...   (control-only runs, no official benchmark)
  local g=$1 ck=$2 P=$3 st=$4 proto=$5; shift 5
  local PA=(--protocol offline); [ $proto = engine ] && PA=(--protocol engine --text-causal $B/models/text-e5-causal-echog-g1-v1)
  SCMG_J=39 SCMG_NO_GLITCH=1 CUDA_VISIBLE_DEVICES=$g ${BENCH_PY:-$PYX} scripts/echog_g1/infer_test.py --ckpt $ck --vae $DEC \
    --stats $C/scmg/STATS_echog_g1_v1.json --view $HV --text-cache $C/scmg/text-e5-echog-g1-v1 --out $P --stems $st --seed 0 \
    --steps 10 --cfg-w ${CFGW:-2.5} --split val --save50 "${PA[@]}" "$@" >> $O/infer.log 2>&1 || log "INFER_FAILED $P"
}
score() {   # gpu label=dir ...  -> $O/sem/<rec>--<label>--dev.json
  local g=$1; shift
  PY=${BENCH_PY:-$PYX} PAIRS= bash scripts/semantic/sem_score_dirs.sh $g $SB/data-sem-hos "$SB/rec-sem-hos-seed*" $O/sem dev "$@" >> $O/sem.log 2>&1
}
post() {    # gpu label ckpt ctl(0/1)
  local g=$1 lab=$2 ck=$3 c=$4 L=$R/$2; local sp=()
  for P in offline engine; do [ -d $L/bench/$P/seed_000 ] && sp+=($lab-$P=$L/bench/$P/seed_000); done
  if [ $c = 1 ]; then
    for m in gt ann lex auto; do bench $g $ck $L/ctl/offline-$m/seed_000 offline 0 --ctl-tokens $CTL --ctl-mode $m; sp+=($lab-off-$m=$L/ctl/offline-$m/seed_000); done
    for m in gt auto; do bench $g $ck $L/ctl/engine-$m/seed_000 engine 0 --ctl-tokens $CTL --ctl-mode $m; sp+=($lab-eng-$m=$L/ctl/engine-$m/seed_000); done
    for m in shuffle amp0 amp2 shift point gt_novals; do infer $g $ck $L/ctl/x-$m $CTL/dev_ann_stems.txt offline --ctl-tokens $CTL --ctl-mode $m; done
    infer $g $ck $L/ctl/x-gt-cfgctl $CTL/dev_ann_stems.txt offline --ctl-tokens $CTL --ctl-mode gt --cfg-ctl ${CFGCTL:-3.0}
    local ev=(); for d in $L/ctl/offline-* $L/ctl/engine-*; do ev+=($(basename $d)=$d/seed_000); done
    for d in $L/ctl/x-*; do ev+=($(basename $d)=$d); done
    SCMG_J=39 ${BENCH_PY:-$PYX} scripts/semantic/g1_ctl_eval.py --ctl $CTL --view $HV --out $L/ctl/CTL_EVAL.json --pair-none $L/bench/offline/seed_000 \
      none-offline=$L/bench/offline/seed_000 none-engine=$L/bench/engine/seed_000 "${ev[@]}" > $L/ctl/ctl_eval.log 2>&1 || log "CTL_EVAL_FAILED $lab"
  fi
  case " ${TS_LANES:-} " in *" $lab "*)
    bench $g $ck $L/ts/seed1/seed_001 offline 1; bench $g $ck $L/ts/swap/seed_000 offline 0 --text-swap
    bench $g $ck $L/ts/notext/seed_000 offline 0 --modality audio; bench $g $ck $L/ts/noaudio/seed_000 offline 0 --modality text
    sp+=($lab-notext=$L/ts/notext/seed_000 $lab-swap=$L/ts/swap/seed_000 $lab-noaudio=$L/ts/noaudio/seed_000)
    ${BENCH_PY:-$PYX} scripts/echog_g1/text_sensitivity.py --out $L/ts/TEXT_SENS.json swap=$L/bench/offline/seed_000,$L/ts/swap/seed_000,$L/ts/seed1/seed_001 \
      notext=$L/bench/offline/seed_000,$L/ts/notext/seed_000,$L/ts/seed1/seed_001 > $L/ts/ts.log 2>&1 || log "TS_FAILED $lab"
    local ws=(base=$L/bench/offline/seed_000,$L/ts/seed1/seed_001,$L/ts/notext/seed_000,$L/ts/noaudio/seed_000);;
  *) local ws=();; esac
  case " ${CFGT_LANES:-} " in *" $lab "*)    # separate text guidance (inference only); no-text / no-audio runs are shared (no text = no cfg_t)
    for ctv in ${CFGT:-2.5 4.0 6.0}; do bench $g $ck $L/cfgt/ct$ctv/seed_000 offline 0 --cfg-t $ctv; sp+=($lab-ct$ctv=$L/cfgt/ct$ctv/seed_000)
      if [ ${#ws[@]} -gt 0 ]; then bench $g $ck $L/cfgt/ct$ctv-s1/seed_001 offline 1 --cfg-t $ctv
        ws+=(ct$ctv=$L/cfgt/ct$ctv/seed_000,$L/cfgt/ct$ctv-s1/seed_001,$L/ts/notext/seed_000,$L/ts/noaudio/seed_000); fi; done;; esac
  [ ${#ws[@]} -gt 0 ] && { ${BENCH_PY:-$PYX} scripts/semantic/sem_window_sens.py --ctl $CTL --out $L/ts/WINDOW_SENS.json "${ws[@]}" > $L/ts/ws.log 2>&1 || log "WS_FAILED $lab"; }
  score $g "${sp[@]}" && log "SEM_DONE $lab" || log "SEM_FAILED $lab"
  log "POST_DONE $lab"
}
lane() {    # gpu label cfg sc data
  local g=$1 lab=$2 cf=$3 sc=$4 dt=$5
  ${PYX} -c "import json,sys; json.loads(sys.argv[1]); json.loads(sys.argv[2])" "$cf" "$sc" || { log "BAD_JSON $lab"; return 2; }
  local de=(); case $dt in *c*) de+=(G1_CTL=$CTL);; esac; case $dt in *g*) de+=(G1_GEST=$GEST);; esac
  case $dt in *f*) de+=(G1_CTL=$CTL CTL_FOCUS=${CTL_FOCUS:-0.3});; esac
  log "lane $lab gpu $g cfg $cf sc $sc data $dt"
  if [ $lab = ctrl ]; then
    mkdir -p $R/ctrl; ln -sfn $INIT_CTRL $R/ctrl/final.pt
    for P in offline engine; do bench $g $R/ctrl/final.pt $R/ctrl/bench/$P/seed_000 $P 0 --save50; done
  else
    env "${de[@]}" INIT=$INIT VAE=$DEC "${VENV[@]}" INFER_EXTRA=--save50 bash scripts/r4_lane.sh $g $R $lab "$cf" "$sc" ${STEPS:-2000} > $O/lane-$lab.log 2>&1
  fi
  log "LANE_END $lab"
  [ -f $R/$lab/final.pt ] || { log "NO_FINAL $lab"; return 1; }
  local c=0; case $cf in *'"ctl": true'*) c=1;; esac
  post $g $lab $R/$lab/final.pt $c
}
[ -s $CTL/dev_ann_stems.txt ] || ${PYX} -c "
import json,sys; r=json.load(open(sys.argv[1]+'/CTL.json'))
print('\n'.join(sorted(s for s,ts in r.items() if ts and ts[0]['split']=='dev' and any(t['type']<=4 for t in ts))))" $CTL > $CTL/dev_ann_stems.txt
i=0
CT='{"adsc": true, "ctl": true, "ctl_flow": true}'
for spec in ${LANES:-'ctrl||' 'ctl|CT|"ctl_w":1.0,"ctl_field_drop":0.2|c'}; do   # label|cfg (CT / CP / empty)|sc fragment|data flags
  IFS='|' read -r lab cfr scr dt <<< "$spec"
  case $cfr in CT) cf=$CT;; CP) cf='{"adsc": true, "ctl": true}';; SH) cf='{"adsc": true, "sem_head": true, "sem_cond": true}';;
    FT) cf='{"adsc": true, "flow_text_xattn": true}';; CTFT) cf='{"adsc": true, "ctl": true, "ctl_flow": true, "flow_text_xattn": true}';; *) cf='{"adsc": true}';; esac
  sc="{$S}"; [ -n "$scr" ] && sc="{$S, ${scr//,/, }}"
  g=${GL[$((i % ${#GL[@]}))]}; i=$((i + 1))
  lane $g $lab "$cf" "$sc" "${dt:-}" &
  sleep 20
done
wait
# ground truth + floors on the same events, then the table
PAIRS= score ${GL[0]} gt=- gt-mismatch=$SB/outputs-dev/gt-mismatch gt-shift2s=$SB/outputs-dev/gt-shift2s
srcs=$(for f in $O/sem/*--dev.json; do b=$(basename $f --dev.json); echo ${b#*--}; done | sort -u | grep -v '^gt')
pairs=$( (echo "$srcs" | grep -v '^ctrl-offline$' | sed 's/$/:ctrl-offline/'
          for x in $srcs; do case $x in *-notext) b=${x%-notext}; echo "$srcs" | grep -qx "$b-offline" && echo "$b-offline:$x";; esac; done) | paste -sd,)
${BENCH_PY:-$PYX} scripts/semantic/sem_report.py --scores $O/sem --rec-prefix rec-sem-hos- --split dev --pairs "$pairs" --cluster recording --out $O/sem/REPORT.json >> $O/sem.log 2>&1
${PYX} scripts/semantic/ctl_summary.py $O > $O/SUMMARY.txt 2>&1
log "CTL_G1_DONE"
