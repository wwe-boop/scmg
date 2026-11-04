#!/bin/bash
# 11-05 MM20 (official protocol, docs/BENCHMARK.md: 20 independent samples per clip, run_000..run_019, --require-all-multimodality)
# on a STEM SUBSET for both ECHO-G (official sampler, 8 steps, guidance 1, as published) and an SCMG checkpoint, so the
# multimodality metric can be compared before the full-val_common MM20 of the final model.
#   mm20_subset.sh <scmg ckpt> <scmg vae> <out root> <stem list>      env GPUS (SCMG shards), EGPU (ECHO-G GPU), SCMG_EXTRA
set -uo pipefail
CODE=$(cd "$(dirname "$0")/../.." && pwd); cd $CODE
ck=$1; vae=$2; O=$3; ST=$4; mkdir -p $O
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=$B/baseline-repro-1104/echog/data_root; W=/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/echog-official
C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; EPY=${EPY:-/mnt/cpfs/zbl-cpfs-new/USERS/zehan/envs/echog/bin/python}; PY=${PY:-python}
log() { echo "$(date -Is) $*" >> $O/mm.log; }
# --- ECHO-G: official condition-only sampler (echo-g-infer) on the stem list, published frozen-condition manifest + SHA256
# (checked 11-05: identical to the full-val_common seed-0 export for the same stems)
A=$D/audit/condition
(
  if [ ! -f $O/echog/.done ]; then
    CUDA_VISIBLE_DEVICES=${EGPU:-0} PYTHONPATH=$W/official/src $EPY -c "import sys; from echo_g.inference import infer_main; sys.exit(infer_main())" \
      --config $W/weights/configs/sgdit_audio_text.yaml --checkpoint $W/weights/best.pt --condition-dir $D/condition_30fps \
      --condition-manifest $A/frozen_conditions.json --condition-manifest-sha256 81e0b1197821f9014d147c8d17970ac9143a7d7f72c529d74e891a944d692b01 \
      --lengths-csv $A/per_clip_condition_motion_audit.csv --stem-list $ST --output-dir $O/echog \
      --seeds $(seq -s ' ' 0 19) --steps 8 --guidance-scale 1 --batch-size 3 --device cuda >> $O/echog.log 2>&1 \
      && touch $O/echog/.done || log "ECHOG_SAMPLE_FAILED"
  fi
  for i in $(seq -f %03g 0 19); do [ -e $O/echog/run_$i ] || ln -s seed_$i $O/echog/run_$i; done
  log "ECHOG_SAMPLED"
) &
# --- SCMG: 20 offline samples (inference seeds 0..19), same settings as bench.sh
GL=(${GPUS:-1}); n=${#GL[@]}
for s in $(seq 0 19); do
  P=$O/scmg/run_$(printf %03d $s); [ -f $P/.done ] && continue; mkdir -p $P
  for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=${GL[$i]} SCMG_J=39 SCMG_NO_GLITCH=1 $PY scripts/echog_g1/infer_test.py --ckpt $ck --vae $vae --stats $C/scmg/STATS_echog_g1_v1.json \
      --view $C/echog-g1-v1/test --text-cache $C/scmg/text-e5-echog-g1-v1-test --out $P --stems $ST --seed $s --steps ${STEPS:-10} \
      --cfg-w ${CFGW:-2.5} --protocol offline --shard $i/$n ${SCMG_EXTRA:-} >> $O/scmg.log 2>&1 &
  done; wait
  [ $(ls $P/*.pt | wc -l) -eq $(grep -c . $ST) ] && touch $P/.done || log "SCMG_RUN_INCOMPLETE $s"
done
log "SCMG_SAMPLED"; wait
ev() {  # <name> <root>
  OMP_NUM_THREADS=${OMP:-8} PYTHONPATH=$W/official/src $EPY $W/official/scripts/eval_g1_motion_cls.py --pred-dir $2/run_000 --ref-dir $D/motion_39d_30fps \
    --val-split $ST --wav-dir $D/audio --mmae-file $D/eval_assets/mmae/g1_mmae_30body_30fps.npy --g1-ae-ckpt $W/weights/evaluation/g1_fgd_encoder.pt \
    --fps 30 --ba-direction audio_to_motion --enable-foot-metrics --require-all-stems --multimodality-root $2 --multimodality-runs 20 \
    --require-all-multimodality --tag mm20_$1 --out $O/benchmark_mm20_$1.json >> $O/eval_$1.log 2>&1 && log "EVAL_DONE $1" || log "EVAL_FAILED $1"
}
ev echog $O/echog; ev scmg $O/scmg
log "MM20_SUBSET_DONE"
