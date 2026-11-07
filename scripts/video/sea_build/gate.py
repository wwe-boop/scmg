"""G6 data build step 4: subwindow re-gate of the ready utterances (newdata-1103 seamless_finalize.py logic, with the (c)1 R4 rule
in place of the R0 repair fraction): on the utterance's parent frame range, far-frame repair fraction (repair > 5 cm on a hand that is
not near-body, near = build/near/<parent>.npy) <= 0.20, continuity (max |dq_full| <= 0.06 rad) and edit fraction <= 0.20 as before;
q34 must equal the parent q_ctrl rows. Writes seamless/GATED.json, SUBWINDOW_EXCLUDED_R4.json and GATE_SUMMARY.json.
    python gate.py <build dir> [<near dir>]"""
import json, sys
from pathlib import Path
import numpy as np
B = Path(sys.argv[1]); N = Path(sys.argv[2]) if len(sys.argv) > 2 else B / 'near'; D = B / 'seamless'
snap = {}
for f in ('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/scmg/newdata-1103/SEAMLESS_SNAPSHOT.jsonl', str(B / 'SELECT.jsonl')):
    if Path(f).exists():
        for l in open(f): r = json.loads(l); snap[r['id']] = r
items = json.loads((D / 'ITEMS_READY.json').read_text()); out, exc = [], []
for it in items:
    m = json.loads((Path(it['dir']) / 'META.json').read_text()); p = snap[m['parent_id']]; root = Path(p['result']).parent; a, b = m['parent_frame_range']
    rep = np.load(root / 'repair_cm.npy')[a:b] > 5; near = np.load(N / f"{m['parent_id']}.npy")[a:b]; n = min(len(rep), len(near))
    far = float((rep[:n] & ~near[:n]).any(1).mean()); allrep = float(rep.any(1).mean())
    full = np.load(root / 'q_full.npy')[a:b]; base = np.load(root / 'release06_candidate.npy')[a:b]
    edit = float((np.abs(full - base).max(1) > 1e-4).mean()); cont = bool(np.abs(np.diff(full.astype(float), axis=0)).max() <= .06)
    assert np.array_equal(np.load(Path(it['dir']) / 'q34.npy'), np.load(root / 'q_ctrl.npy')[a:b])
    it['subwindow_gate_r4'] = dict(far_repair_fraction=far, repair_gt5cm_fraction=allrep, near_fraction=float(near[:n].any(1).mean()),
                                   changed_fraction_new=edit, continuity_pass=cont, rule='(c)1 R4 on the utterance subwindow')
    (out if (far <= .2 and edit <= .2 and cont) else exc).append(it)
(D / 'GATED.json').write_text(json.dumps(out, ensure_ascii=False)); (D / 'SUBWINDOW_EXCLUDED_R4.json').write_text(json.dumps(exc))
s = dict(ready=len(items), kept=len(out), excluded=len(exc), hours=sum(r['frames'] for r in out) / 180000, speakers=len({r['speaker'] for r in out}),
         parents=len({r['clip_id'] for r in out}), near_fraction_mean=float(np.mean([r['subwindow_gate_r4']['near_fraction'] for r in out])) if out else None)
(D / 'GATE_SUMMARY.json').write_text(json.dumps(s, indent=1)); print('GATE_DONE', json.dumps(s))
