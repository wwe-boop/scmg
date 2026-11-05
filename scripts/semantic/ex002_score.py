"""EX002 semantic-recognition baseline on manifest-n200 (semantic-baseline-1105).

Runs scripts/semantic_eval.py score --per-event (recogniser semantic-recognizer-v1, centre window, +-0.5 s search) on
<root>/<lane>/<protocol>/semantic-s<seed>/<label>/{own,notext,swap} and on the ground truth, then adds the paired text
controls that semantic_eval.py does not report:
  own_minus_swap    top-1(own text) - top-1(another clip's text, same audio / seed)   <- text-caused recognition
  own_minus_notext  top-1(own) - top-1(text masked)
  pair_win_swap     P(p_intended(own) > p_intended(swap)) over events (0.5 = text has no effect on the intended class)
    python scripts/semantic/ex002_score.py --root ~/bulk/semantic-baseline-1105/ex002 --out <json>
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MAN = Path('~/bulk/semrec-cache/manifest-n200/MANIFEST.json').expanduser()   # local copy of semantic-recognizer-v1/manifest-n200


def run(gen_dir, out, prefix='gen_', py=sys.executable):
    if not Path(out).exists():
        subprocess.run([py, str(HERE.parent / 'semantic_eval.py'), 'score', '--gen-dir', str(gen_dir), '--manifest', str(MAN),
                        '--out', str(out), '--per-event', '--prefix', prefix], check=True, cwd=HERE.parent.parent)
    return json.loads(Path(out).read_text())


def summ(r):
    c = r['centre']
    return dict(top1=c['top1'], bal_top1=c['balanced_top1'], shuffled_top1=c['shuffled_top1'], text_following=c['text_following'],
                search_top1=r['search']['top1'], det_rate=c['detection_rate'], fpr=r['false_positive_rate'], n_events=r['n_events'])


def paired(a, b):
    pa = np.array([e['p_intended'] for e in a['events']]); pb = np.array([e['p_intended'] for e in b['events']])
    ta = np.array([e['pred'] == e['key'] for e in a['events']]); tb = np.array([e['pred'] == e['key'] for e in b['events']])
    assert [e['file'] for e in a['events']] == [e['file'] for e in b['events']]
    rng = np.random.default_rng(0); d = ta.astype(float) - tb
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return dict(diff_top1=float(d.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                pair_win=float(np.mean((pa > pb) + 0.5 * (pa == pb))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); root = Path(a.root).expanduser(); sc = root / 'scores'; sc.mkdir(exist_ok=True)
    res = dict(manifest=str(MAN), recogniser='semantic-recognizer-v1 (~/bulk/semrec-cache/final/model.pt)', rows={})
    g = run(MAN.parent, sc / 'gt.json', prefix='gt_'); res['gt'] = summ(g)
    for lane in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != 'scores'):
        for proto in ('offline', 'engine'):
            for sd in sorted((root / lane / proto).glob('semantic-s*')):
                lab = next(p for p in sd.iterdir() if p.is_dir())
                v = {k: run(lab / k, sc / f'{lane}-{proto}-{sd.name}-{k}.json') for k in ('own', 'notext', 'swap')}
                row = {k: summ(x) for k, x in v.items()}
                row['own_vs_swap'] = paired(v['own'], v['swap']); row['own_vs_notext'] = paired(v['own'], v['notext'])
                row['gt_vs_own'] = paired(g, v['own'])
                res['rows'][f'{lane}/{proto}/{sd.name}'] = row
    Path(a.out).expanduser().write_text(json.dumps(res, indent=1))
    print(f"GT top1 {res['gt']['top1']:.3f}")
    for k, r in res['rows'].items():
        print(f"{k:40s} own {r['own']['top1']:.3f} (shuf {r['own']['shuffled_top1']:.3f}, det {r['own']['det_rate']:.2f}) "
              f"notext {r['notext']['top1']:.3f} swap {r['swap']['top1']:.3f} | own-swap {r['own_vs_swap']['diff_top1']:+.3f} "
              f"{r['own_vs_swap']['ci95']} win {r['own_vs_swap']['pair_win']:.3f}")


if __name__ == '__main__':
    main()
