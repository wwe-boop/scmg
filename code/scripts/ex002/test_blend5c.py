"""Module 5c acceptance tests (11-08; criteria docs/criteria/EX002_CTL.md "Module 5c"): smooth composition of the command-local windows.
    python scripts/ex002/test_blend5c.py
 A default path: weight(..., compose='max') == the 5b weight() at commit 366aed0, bitwise, on every dev plan
 B no overlap -> 5c == 5b bitwise: every sparse plan (all clips) and the clips of the dense plans whose windows do not overlap
 C C1: on the dense plans, max |second difference of w| * FPS^2 <= 1.5 pi^2 / r^2 (bound for 1 - (1 - a)(1 - b) of two raised cosines),
   plus a 5 % discretisation margin; the 5b max-composition violates it (the bug this module fixes)
Prints ALL_OK / SOME_FAIL."""
import json, subprocess, sys, types
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex002_local_blend import weight, FPS
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002'
PLANS = ['PLAN_dev', 'PLAN_dev_sparse', 'PLAN_dev_fresh_dense', 'PLAN_dev_fresh_sparse', 'PLAN_dev_port_dense', 'PLAN_dev_port_sparse',
         'PLAN_dev_m78b_dense', 'PLAN_dev_m78b_sparse', 'PLAN_dev_m9_dense', 'PLAN_dev_m9_sparse', 'PLAN_dev_m9b_dense', 'PLAN_dev_m9b_sparse']
PRE, POST, INNER, OUTER = 0.3, 0.9, 0.6, 0.9
W5B_366AED0 = 'def weight(T, ts, inner, outer, pre=None, post=None):\n    # symmetric (module 5): 1 for |t - t_cmd| <= inner, cosine to 0 at outer; asymmetric (5b): 1 on [t - pre, t + post], same ramp width\n    fr = np.arange(T) / FPS; w = np.zeros(T); r = outer - inner\n    for t in ts:\n        if pre is None: d = np.abs(fr - t) - inner\n        else: d = np.where(fr < t - pre, (t - pre) - fr, np.where(fr > t + post, fr - (t + post), 0.))\n        x = np.clip(1 - np.maximum(d, 0) / r, 0, 1) if pre is not None else np.clip((outer - np.abs(fr - t)) / r, 0, 1)\n        w = np.maximum(w, 0.5 - 0.5 * np.cos(np.pi * x))\n    return w\n'   # verbatim weight() of 366aed0 (release dirs have no git)


def old_weight():
    m = types.ModuleType('old_blend'); m.__dict__.update(np=np, FPS=FPS); exec(compile(W5B_366AED0, 'old', 'exec'), m.__dict__)
    try:                                                            # in a git checkout: the embedded copy must equal 366aed0
        src = subprocess.run(['git', 'show', '366aed0:scripts/ex002/ex002_local_blend.py'], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parents[2]).stdout
        assert src[src.index('def weight('):src.index('def main():')].rstrip() + '\n' == W5B_366AED0, 'embedded 5b weight differs from 366aed0'
    except (subprocess.CalledProcessError, FileNotFoundError): pass
    return m.weight


def main():
    ow = old_weight(); ok = dict(A=True, B=True, C=True); r = OUTER - INNER; bound = 1.05 * 1.5 * np.pi ** 2 / r ** 2
    nB = nC = 0; worst_max = worst_s = 0.
    for pn in PLANS:
        f = Path(f'{X}/{pn}.json')
        if not f.exists(): print('skip', pn); continue
        by = {}
        for e in json.loads(f.read_text())['events']: by.setdefault(e['ci'], []).append(e['t'])
        for ci, ts in by.items():
            T = int((max(ts) + 3) * FPS)
            for pre, post in ((PRE, POST), (None, None)):
                if not np.array_equal(weight(T, ts, INNER, OUTER, pre, post), ow(T, ts, INNER, OUTER, pre, post)): ok['A'] = False
            w5b = weight(T, ts, INNER, OUTER, PRE, POST); w5c = weight(T, ts, INNER, OUTER, PRE, POST, compose='smooth')
            gaps = np.diff(sorted(ts))
            if not (gaps < PRE + POST + 2 * r).any():
                nB += 1
                if not np.array_equal(w5b, w5c): ok['B'] = False
            else:
                nC += 1; s5c = np.abs(np.diff(w5c, 2)).max() * FPS ** 2; s5b = np.abs(np.diff(w5b, 2)).max() * FPS ** 2
                worst_s = max(worst_s, s5c); worst_max = max(worst_max, s5b)
                if s5c > bound: ok['C'] = False
    if worst_max <= bound: ok['C'] = False; print('C: the 5b max-composition should violate the bound on the dense plans')
    print(f'A default == 5b (366aed0) bitwise: {ok["A"]}')
    print(f'B no-overlap clips 5c == 5b bitwise: {ok["B"]} ({nB} clip-plans)')
    print(f'C C1 on overlapping clips: {ok["C"]} ({nC} clip-plans; max |w\'\'| 5c {worst_s:.1f} vs bound {bound:.1f}; 5b {worst_max:.1f})')
    print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()
