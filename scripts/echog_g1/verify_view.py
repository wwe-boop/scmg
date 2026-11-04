"""Self-checks of an assembled echog-g1 view (+ test/):  channel order vs the official G1_JOINT_NAMES, splits disjoint
(stems and takes), row bookkeeping, frame counts vs the official audit, 50 -> 30 FPS round trip of stored rows vs the
official motion files, statistics computed on train rows only, text cache coverage (optional).
    verify_view.py <view> <official motion dir> <meta dir> [text cache] [test text cache]"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scmg import g1
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES

V, mdir, meta = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ok = {}
M = json.loads((V / 'META.json').read_text())
ok['channel_order'] = M['joints'][10:] == [n + '_joint' for n in G1_JOINT_NAMES] and len(M['joints']) == 39
sp = json.loads((V / 'SPLIT.json').read_text()); spt = json.loads((V / 'test' / 'SPLIT.json').read_text())
tr, dv, te = set(sp['train_ids']), set(sp['val_ids']), set(spt['test_ids'])
take = lambda s: s.rsplit('_utt_', 1)[0]
ok['stems_disjoint'] = not (tr & dv or tr & te or dv & te) and 'test_ids' not in sp and not spt['train_ids'] and not spt['val_ids']
ok['takes_disjoint'] = not ({take(s) for s in tr} & {take(s) for s in dv} or {take(s) for s in tr | dv} & {take(s) for s in te})
off_tr = (meta / 'splits/train_drop.txt').read_text().split(); off_te = (meta / 'splits/val_common.txt').read_text().split()
ok['train_dev_eq_train_drop'] = tr | dv == set(off_tr) and te == set(off_te)
al = {r['stem']: int(r['aligned_frames']) for r in csv.DictReader(open(meta / 'audit/condition/per_clip_condition_motion_audit.csv'))}
rng = np.random.default_rng(0); errs = []
for root in (V, V / 'test'):
    items = json.loads((root / 'train.json').read_text()); q = np.load(root / 'train_q.npy', mmap_mode='r'); f = np.load(root / 'train_feat.npy', mmap_mode='r')
    off = 0; good = True
    for it in items:
        good &= it['offset'] == off and it['frames'] == int(round(al[it['id']] * 5 / 3)) and it['echog']['frames30'] == al[it['id']]
        off += it['frames']
    ok[f'rows_{root.name}'] = good and off == len(q) == len(f) and q.shape[1] == 39 and f.shape[1] == 89
    ok[f'finite_{root.name}'] = bool(np.isfinite(np.asarray(q[::97])).all() and np.isfinite(np.asarray(f[::97], np.float32)).all())
    for it in [items[i] for i in rng.choice(len(items), 40, replace=False)]:
        r = torch.load(mdir / f"{it['id']}.pt", map_location='cpu', weights_only=True)['robot_repr'].double().numpy()[:al[it['id']]]
        back = g1.to30(np.asarray(q[it['offset']:it['offset'] + it['frames']], np.float64), len(r))
        errs.append(np.abs(back - r).max(0))
e = np.max(errs, 0)
ok['roundtrip_rows_vs_official'] = dict(max_rot6d=float(e[:6].max()), max_yaw=float(e[6]), max_vel=float(e[7:10].max()), max_joint=float(e[10:].max()))
for k, tc in (('text', sys.argv[4] if len(sys.argv) > 4 else None), ('text_test', sys.argv[5] if len(sys.argv) > 5 else None)):
    if not tc: continue
    idx = json.loads((Path(tc) / 'TEXT_INDEX.json').read_text()); root = V if k == 'text' else V / 'test'
    items = json.loads((root / 'train.json').read_text())
    n = np.array([idx[it['id']][1] for it in items if it['id'] in idx])
    ok[k] = dict(items=len(items), indexed=len(n), tokens_max=int(n.max()), at_160_cap=int((n >= 158).sum()),
                 prep=json.loads((Path(tc) / 'PREP.json').read_text()))
print(json.dumps(ok, indent=1))
