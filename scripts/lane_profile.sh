#!/bin/bash
# Data / model profile of the r4 lane scripts (sourced by r4_lane.sh and bagger_r4.sh; 11-04).
#   PROFILE=ex002 (default): EXACTLY the assets the 11-04 EX002 scripts hard-coded (r4 base, vae-r3, v36_1 view / stats /
#                            e5 / causal e5 / Qwen2.5 tokens / intent tracks / semantic holdout); nothing changes for EX002.
#   PROFILE=g1             : ECHO-G Unitree G1 (SCMG_J=39, view echog-g1-v1). INIT (G1 base checkpoint) and VAE (its G1 VAE)
#                            are REQUIRED -- there is no G1 default and an EX002 path is refused. No token / intent / holdout
#                            / glitch assets exist for G1: they are off (and scmg.data refuses EX002 ones on a G1 view).
# Exports: SCMG_TEXT_CAUSAL (+ SCMG_J, SCMG_NO_GLITCH for g1); sets INIT VAE VIEW STATS TEXT, arrays DATA_ARGS (scmg.train
# view/asset flags) and COLLECT_ARGS (scripts/bagger_collect.py asset flags), LANES_OK (lanes this profile supports).
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; D=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg
case "${PROFILE:-ex002}" in
  ex002)
    export SCMG_TEXT_CAUSAL=$B/models/text-e5-causal-v36_1
    INIT=${INIT:-$B/dlc-scmg-train-r4-1102/base/best.pt}; VAE=${VAE:-$B/trained/vae-r3/final.pt}; export INIT VAE
    STATS=$D/STATS_v36_1.json; TEXT=$D/text-e5-v36_1; SEMMAN=$B/semantic-recognizer-v1/manifest-n200
    DATA_ARGS=(--stats $STATS --text-cache $TEXT --tok-cache $D/tok-qwen25-v36_1 --tok-intent $B/models/intent-qwen25-v36_1
               --intent-frames $B/models/intent-frames-v36_1 --holdout $SEMMAN/MANIFEST.json)
    COLLECT_ARGS=()                                   # bagger_collect.py EX002 defaults
    LANES_OK="a b c";;
  g1)
    export SCMG_J=39 SCMG_NO_GLITCH=1
    INIT=${INIT:?PROFILE=g1 needs INIT=<G1 base checkpoint>}; VAE=${VAE:?PROFILE=g1 needs VAE=<the G1 VAE of that base>}; export INIT VAE
    case "$INIT $VAE" in *v36_1*|*dlc-scmg-train-r4-1102*|*trained/vae-r3*) echo "PROFILE=g1: EX002 checkpoint in INIT/VAE" >&2; exit 2;; esac
    C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture
    VIEW=${G1_VIEW:-$C/echog-g1-v1}; STATS=${G1_STATS:-$D/STATS_echog_g1_v1.json}; TEXT=${G1_TEXT:-$D/text-e5-echog-g1-v1}
    export SCMG_TEXT_CAUSAL=${G1_TEXT_CAUSAL:-$B/models/text-e5-causal-echog-g1-v1}   # prefix-causal e5 of the SAME rows (View checks)
    DATA_ARGS=(--view $VIEW --stats $STATS --text-cache $TEXT)
    [ -n "${G1_GEST:-}" ] && DATA_ARGS+=(--gest-frames $G1_GEST)   # 11-05: BEAT2-annotation g_k frame track (semantic head lanes)
    COLLECT_ARGS=(--vae $VAE --view $VIEW --stats $STATS --text-cache $TEXT --tok-cache none --holdout none)
    LANES_OK="b c";;                                   # lane a needs intent tracks / semantic data (EX002 only)
  *) echo "unknown PROFILE=${PROFILE}" >&2; exit 2;;
esac
