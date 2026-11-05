#!/bin/bash
# 11-04 G1 inference-only sweep on the held-out DEV split (306 clips, echog-g1-dev-assets; never val_common): which
# inference setting closes the gap to ECHO-G on dDiv / dJerk / foot / slide without hurting FGD.  Pre-registered rule
# (ops/SCMG_IMPROVE_DESIGN_1104.md 7.9): among settings whose dev FGD <= 1.05 x the reference row, pick the lowest mean
# rank over (dDiv, dBA, dJerkW, foot error, contact slide); ties -> lower FGD.  The winner then runs ONCE on val_common.
#   infer_sweep.sh <ckpt> <vae> <out root>      env GPUS (shards), SETTINGS (names below), PROTOS (default offline)
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; A=${DEV_ASSETS:-$B/echog-g1-dev-assets}; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture   # DEV_ASSETS / DEV_VIEW: held-out-speaker dev (11-05)
D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
export OFF=$W/official REF_DIR=$A/motion_39d_30fps WAV_DIR=$A/audio SPLIT=$A/dev_stems.txt MMAE=$D/eval_assets/mmae/g1_mmae_30body_30fps.npy \
  FGD_CKPT=$W/weights/evaluation/g1_fgd_encoder.pt VIEW_TEST=${DEV_VIEW:-$C/echog-g1-v1} TEXT_TEST=$C/scmg/text-e5-echog-g1-v1 \
  TEXT_CAUSAL_TEST=$B/models/text-e5-causal-echog-g1-v1 VSPLIT=val
ck=$1; vae=$2; O=$3; mkdir -p $O
# our signed jerk gap is NEGATIVE (smoother than GT, VAE jerk ratio 0.75) -> whole-body smoothing is the wrong direction;
# only the legs are smoothed (contact slide).  Div is too HIGH (+0.53) -> null speaker / smaller initial noise.
declare -A X=( [ref]="" [nospk]="--no-spk" [ns09]="--noise-scale 0.9" [ns08]="--noise-scale 0.8" [legs31]="--smooth-legs 31,2"
               [legs15]="--smooth-legs 15,2" [steps20]="STEPS=20" [sm9]="--smooth 9,2"
               [cfg15]="CFGW=1.5" [cfg20]="CFGW=2.0" [cfg30]="CFGW=3.0" [cfg20nospk]="CFGW=2.0 --no-spk" )
for s in ${SETTINGS:-ref nospk ns09 ns08 legs31 legs15 steps20}; do for p in ${PROTOS:-offline}; do
  [ -f $O/benchmark_${s}_${p}_seed_000.json ] && continue
  e=${X[$s]}; st=10; cw=2.5
  case "$e" in STEPS=*) st=${e#STEPS=}; e="";; esac
  case "$e" in CFGW=*) cw=${e%% *}; cw=${cw#CFGW=}; case "$e" in *" "*) e=${e#* };; *) e="";; esac;; esac
  CFGW=$cw STEPS=$st INFER_EXTRA="$e" bash scripts/echog_g1/bench.sh scmg $ck $vae $O/${s}_${p}_seed_000 $p 0 >> $O/run.log 2>&1 \
    && echo "$(date -Is) SWEEP_DONE $s $p" >> $O/sweep.log || echo "$(date -Is) SWEEP_FAILED $s $p" >> $O/sweep.log
done; done
echo "$(date -Is) INFER_SWEEP_ALL_DONE" >> $O/sweep.log
