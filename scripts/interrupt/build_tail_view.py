"""Post-speech tail views for the floor-state training (11-06).

A packaged ECHO-G clip ends exactly at its last word. For the training utterances followed by a VAD-confirmed silence
(human_gaps.py, BEAT2 take audio, >= 1.2 s), the GMR-retargeted motion of that silence (same pipeline as the reproduced
baselines, calibrated joint MAE 0.013 rad vs ECHO-G GT) is appended to the item:
  motion  : the item's 50 Hz rows unchanged, then the tail (30 -> 50 Hz with scmg/g1.py to50 from 3 frames before the speech
            offset; the offset jump to the last GT frame is removed with a smooth weight over 0.3 s on all 39 channels)
  audio   : NONE after the offset -- tail feat rows hold the normalisation mean (= 0 after normalisation, the end-of-audio
            convention of the streaming engine); the loader marks the audio end (aend) at the first tail frame, so the
            stream state says "audio final" there, exactly as after an interruption with the audio truncated
  text    : unchanged (every word was spoken before the offset)
  FLOOR.npy (int8 per row): 0 = speaking (the item's own rows), tail rows 1 = holding the floor / 2 = yielding
Tail labels (rest height of human_ref R_sil + 5 cm, icommon.yield_time over the whole tail):
  retract = hands above rest height at the offset and lowered before the tail ends; low = hands already at rest height;
  hold = hands up and staying up.
Variants:  labelled  every tail, floor 2 (retract | low) / 1 (hold)        yield  only retract | low tails, floor 2
Excluded speakers (never get tails): the dev held-out speakers (HOS.json: 2_scott, 6_carla) and the val_common speakers.
Writes <out>/ (SPLIT of echog-g1-v1) and <out>-hos/ (SPLIT / HOS of views/echog-g1-v1-hos, arrays symlinked).
    python scripts/interrupt/build_tail_view.py --base <echog-g1-v1> --hos <views/echog-g1-v1-hos> --tails <gaps_train_tail>
        --human human_ref_v2_train_drop.json --stats STATS --variant labelled --out <dataset dir>/echog-g1-v1-tail-lab
"""
import argparse, glob, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1])); sys.path.insert(0, str(HERE))
from scmg import g1

ap = argparse.ArgumentParser()
ap.add_argument('--base', required=True); ap.add_argument('--hos', required=True); ap.add_argument('--tails', required=True)
ap.add_argument('--human', required=True); ap.add_argument('--stats', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--variant', required=True, choices=['labelled', 'yield']); ap.add_argument('--fade', type=float, default=0.3)
a = ap.parse_args()
base, out = Path(a.base), Path(a.out)
items = json.loads((base / 'train.json').read_text())
q = np.load(base / 'train_q.npy', mmap_mode='r'); f = np.load(base / 'train_feat.npy', mmap_mode='r')
fm = np.asarray(json.loads(Path(a.stats).read_text())['f_mean'], np.float32)
rest_z = np.array(json.loads(Path(a.human).read_text())['rest']['R_sil']['hand_z'])
hos = json.loads((Path(a.hos) / 'HOS.json').read_text())
EXCL = {s.split('_', 1)[1] for s in hos['speakers']} | {'1_wayne', '3_solomon', '21_ayana'}   # dev held-out + val_common

# ---- tails: label + 50 Hz rows starting right after the offset
import icommon as I
idx = [json.loads(l) for fn in sorted(glob.glob(f'{a.tails}/index.*.jsonl')) for l in open(fn)]
by_stem = {}
lab = Counter()
for c in idx:
    if c['split'] != 'train_drop' or c['speaker'] in EXCL: lab['excluded'] += 1; continue
    d = np.load(f"{a.tails}/seg/{c['id']}.npz"); rr = d['robot_repr'].astype(np.float64); off = int(d['off'])
    t, need = I.yield_time(I.fk(rr), off, rest_z, horizon=len(rr) - off)
    kind = 'low' if not need else ('retract' if t is not None else 'hold')
    lab[kind] += 1
    if a.variant == 'yield' and kind == 'hold': continue
    by_stem[c['stem']] = dict(rr=rr, off=off, kind=kind, floor=1 if kind == 'hold' else 2, id=c['id'])
print('tails', dict(lab), 'used', len(by_stem), flush=True)

# ---- assemble
rows_q, rows_f, rows_fl, new_items, off = [], [], [], [], 0
n_ext = Counter(); skipped = Counter()
for it in items:
    o, n = it['offset'], it['frames']
    qi = np.asarray(q[o:o + n], np.float32); fi = np.asarray(f[o:o + n]); fl = np.zeros(n, np.int8)
    t = by_stem.get(it['id'])
    e = it.get('echog', {})
    if t is not None:
        dur = float(e['end_sec']) - float(e['start_sec'])
        if abs(n / 50. - dur) > 0.04: skipped['item_not_ending_at_speech_end'] += 1; t = None   # aligned prefix shorter than the clip
    if t is not None:
        r30 = t['rr'][t['off'] - 3:]
        r50 = g1.to50(r30).astype(np.float64)[5:]                          # 3 frames @30 = 5 frames @50 before the offset
        r50 = r50[1:]                                                      # first tail row = 20 ms after the offset
        k = np.arange(len(r50)) / 50.; u = np.clip(k / a.fade, 0, 1)[:, None]; w = 1 - u * u * (3 - 2 * u)
        ch = np.r_[0:39]                                                   # rot6d, yaw rate, base velocity, joints
        jump = qi[-1, ch] - g1.to50(t['rr'][t['off'] - 3:t['off'] + 3])[5, ch] if len(qi) else 0.
        r50[:, ch] += w * jump
        qi = np.concatenate([qi, r50.astype(np.float32)]); fi = np.concatenate([fi, np.repeat(fm[None].astype(fi.dtype), len(r50), 0)])
        fl = np.concatenate([fl, np.full(len(r50), t['floor'], np.int8)])
        n_ext[t['kind']] += 1
        it = dict(it, frames=len(qi), tail=dict(id=t['id'], kind=t['kind'], floor=t['floor'], frames=len(r50), speech_frames=n))
    rows_q.append(qi); rows_f.append(fi); rows_fl.append(fl)
    new_items.append(dict(it, offset=off)); off += len(qi)
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'train_q.npy', np.concatenate(rows_q)); np.save(out / 'train_feat.npy', np.concatenate(rows_f))
np.save(out / 'FLOOR.npy', np.concatenate(rows_fl))
(out / 'train.json').write_text(json.dumps(new_items))
for name in ('cond', 'alignments', 'test', 'FROZEN.json', 'labels.jsonl', 'META.json', 'SPLIT.json'):
    p = out / name
    if not p.exists(): os.symlink(base / name, p)
info = dict(variant=a.variant, tails=dict(lab), extended=dict(n_ext), skipped=dict(skipped), rows=off, excluded_speakers=sorted(EXCL),
            base=str(base), tails_dir=str(a.tails), fade_s=a.fade, audio_after_offset='none (feat = f_mean, aend marks the offset)')
(out / 'TAIL.json').write_text(json.dumps(info, indent=1))
hv = Path(str(out) + '-hos'); hv.mkdir(exist_ok=True)
for name in ('train_q.npy', 'train_feat.npy', 'FLOOR.npy', 'train.json', 'cond', 'alignments', 'test', 'FROZEN.json', 'labels.jsonl', 'META.json', 'TAIL.json'):
    p = hv / name
    if not p.exists(): os.symlink(out / name, p)
for name in ('SPLIT.json', 'HOS.json'):
    p = hv / name
    if not p.exists(): os.symlink(Path(a.hos) / name, p)
print(json.dumps(info))
