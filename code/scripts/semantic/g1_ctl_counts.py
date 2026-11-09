"""Count labels for the control tokens (11-07 §7.55 item 3, "tap / raise N times"). Adds 'count' (0..4, 4 = four or more) to every
non-POINT annotation token (iconic / metaphoric / beat) whose span lasts COUNT_SPAN (0.1-4 s), measured on the ground-truth G1 motion
with scmg/ctl.py count_np (strokes of the span's dominant hand: smoothed-wrist-speed peaks above the APEX.json action speed, +-0.1 s
neighbourhood). No other field changes; POINT tokens are untouched. Writes a NEW token dir (CTL.json + the source's PREP / APEX /
dev stems) and COUNT.json (rule, threshold, count distribution and chance agreement sum p_k^2 per split).
    python scripts/semantic/g1_ctl_counts.py --ctl <ctl-tokens-hos> --view <hos view> --out <ctl-tokens-hos-count>
"""
import argparse, collections, json, shutil, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from scmg import ctl
from g1_ctl_apex import gt_geo


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ctl', required=True); ap.add_argument('--view', required=True)
    ap.add_argument('--out', required=True); a = ap.parse_args()
    src, out = Path(a.ctl), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((src / 'CTL.json').read_text()); thr = json.loads((src / 'APEX.json').read_text())['action_speed']
    stems = [s for s, ts in raw.items() if any(ctl.group_of(t['type']) == 0 and t['type'] != ctl.POINT for t in ts)]
    dist = collections.defaultdict(collections.Counter); n_tok = 0
    for i in range(0, len(stems), 500):
        G = gt_geo(a.view, stems[i:i + 500])
        for s, g in G.items():
            sp = ctl.count_speed(g['wrist'])
            for t in raw[s]:
                if ctl.group_of(t['type']) != 0 or t['type'] == ctl.POINT: continue
                if not ctl.COUNT_SPAN[0] <= t['b'] - t['a'] <= ctl.COUNT_SPAN[1]: continue
                c = ctl.count_np(g['wrist'], int(round(t['a'] * 50)), int(round(t['b'] * 50)), thr, sp)
                if c is None: continue
                t['count'] = c; dist[t['split']][c] += 1; n_tok += 1
        print(f'{min(i + 500, len(stems))}/{len(stems)}', flush=True)
    (out / 'CTL.json').write_text(json.dumps(raw))
    for f in ('PREP.json', 'APEX.json', 'dev_ann_stems.txt'):
        if (src / f).exists(): shutil.copy(src / f, out / f)
    meta = dict(source=str(src), action_speed=thr, rule='scmg/ctl.py count_np', count_max=ctl.COUNT_MAX, smooth=ctl.COUNT_SMOOTH,
                half=ctl.COUNT_HALF, span=ctl.COUNT_SPAN, n_tokens=n_tok, splits={})
    for sp_, c in dist.items():
        n = sum(c.values()); p = np.array([c[k] / n for k in range(ctl.COUNT_MAX + 1)])
        meta['splits'][sp_] = dict(n=n, dist=p.round(4).tolist(), chance=float((p ** 2).sum()), majority=float(p.max()))
    (out / 'COUNT.json').write_text(json.dumps(meta, indent=1)); print(json.dumps(meta['splits']))


if __name__ == '__main__':
    main()
