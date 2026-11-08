# D5 quick lane (module 6 revived; pre-registered b2a7ea8 + amendments 4cd9c9a / 1a37e7b): local GPU0. Trains the gated control modules
# (both routes / ctl_flow only) on enc-s0 for 2000 steps (lane-4 recipe, true freeze), generates offline seed 0 on the shard 0/4 dev clips
# (dense plan 1112): given for gate / gateflow / ungated 4000-step baseline (b4k), none for the base -- all on this GPU.
set -u
CODE=$1; O=$2; cd $CODE
B=/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg; C=/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture; X=$B/control-token-1105/ex002; T=$X/tokens
PY=/home/xr-25b7/envs/part2-exec-20261029/bin/python; export SCMG_J=34 SCMG_NO_GLITCH=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
VIEW=$C/train-v1-20261101/views/data-v36_1; ST=$C/scmg/STATS_v36_1.json; TX=$C/scmg/text-e5-v36_1; TOK=$C/scmg/tok-qwen25-v36_1; VAE=$B/trained/vae-r3/final.pt
ENC=$B/ex002-sfscope-1108/enc-s0/train/final.pt; FD=$X/PLAN_dev_m9b_dense.json; B4K=$X/m10-enc/b4k/final.pt
train() {  # out cfgjson
  [ -f $1/DONE.json ] && return 0
  $PY -m scmg.train --stage base --view $VIEW --stats $ST --text-cache $TX --vae $VAE --init $ENC --out $1 --steps 2000 --batch 64 --lr 1e-4 --warmup 200 --workers 8 \
    --seed 0 --eval-every 1000 --snap-every 1000000 --semsynth-share 0.6 --tok-cache $TOK --tok-intent $B/models/intent-qwen25-v36_1 --intent-frames $B/models/intent-frames-v36_1 \
    --holdout $B/semantic-recognizer-v1/manifest-n200/MANIFEST.json --ctl-tokens $T --ctl-focus 0.5 --train-only ctl_ --cfg "$2" \
    --sc '{"p_stream": 0.67, "p_async": 1.0, "p_async_seg": 0.0, "w_int": 0.3, "sem_w": 2.0, "self_force_p": 0, "self_force_scope": "encoders", "ctl_w": 1.0, "w_ctl_aux": 3.0, "ctl_field_drop": 0.2, "ctl_cf_p": 0.5, "ctl_lr_scale": 1.0, "w_ctl_finger": 1.0, "ctl_cf_finger_p": 0.5}' \
    > $1.log 2>&1
}
echo "$(date -Is) START" >> $O/lane.log
train $O/gate '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true, "ctl_gate": true}' && echo "$(date -Is) TRAINED gate" >> $O/lane.log || echo "$(date -Is) TRAIN_FAILED gate" >> $O/lane.log
train $O/gateflow '{"adsc": true, "intent_aux": true, "ctl": true, "ctl_flow": true, "ctl_gate": true, "ctl_route": "flow"}' && echo "$(date -Is) TRAINED gateflow" >> $O/lane.log || echo "$(date -Is) TRAIN_FAILED gateflow" >> $O/lane.log
gen() { $PY scripts/ex002/bench_ex002.py infer --ckpt $1 --protocol offline --seed 0 --out $2 --shard 0/4 $3 >> $O/infer.log 2>&1; }
gen $ENC $O/gen/none_base ""
gen $O/gate/final.pt $O/gen/none_gate ""
gen $O/gate/final.pt $O/gen/gate "--plan $FD --mode given"
gen $O/gateflow/final.pt $O/gen/gateflow "--plan $FD --mode given"
gen $B4K $O/gen/b4k "--plan $FD --mode given"
echo "$(date -Is) GENERATED" >> $O/lane.log
