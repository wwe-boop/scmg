"""Test-set usage log (design v4 §2.3, appendix S6; 11-05): every official-evaluator benchmark_*.json produced on the ECHO-G
test set val_common, with its directory, configuration, mtime and the official 7-metric columns, plus the FGD distribution
of all configurations that have looked at the test set.

Test usage is decided per json from its own record (params.stem_selection.val_split = the stem list it was evaluated on):
  1 the list file is readable -> test usage iff it shares stems with val_common (full = all 3242 stems; else subset k/3242);
  2 otherwise (file gone)     -> basename val_common* or n_clips == 3242 -> test (marked 'by-name'); else 'unknown'.
Directory walk: breadth-first os.scandir inside the given roots only (no find/grep), depth <= --max-depth, never descending
into prediction / data folders (seed_NNN, run_NNN, offline_seed_NNN, bagger-data, ... see SKIP) nor into any folder with
more than --max-entries entries. Pure standard library (runs with the system python3 on the DSW host).

    # on speech_server (CPFS is local there); stdout = scan json
    ssh speech_server python3 - scan --base $B --out - < scripts/echog_g1/test_usage_log.py > usage.json
    python3 scripts/echog_g1/test_usage_log.py report --in usage.json --md usage.md
"""
import argparse, json, os, re, statistics, sys, time
from collections import deque

B_DEFAULT = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'
ROOTS = ['echog-g1-round-1104', 'echog-g1-fixround-1105', 'echog-g1-arch-1104', 'echog-g1-decft-1105', 'echog-g1-stage2-1105',
         'echog-g1-mm20-1105', 'echog-g1-sf-1105', 'semantic-baseline-1105', 'baselines-g1-1105', 'baseline-repro-1104',
         'interrupt-baseline-1105']
VAL_COMMON = B_DEFAULT + '/baseline-repro-1104/echog/data_root/splits/val_common.txt'
SKIP = re.compile(r'^((offline_|engine_|mm_)?seed_\d+|run_\d+|bagger-data|r4-stream-bag|meshes|GMR|weights|hf|dl|pydeps|'
                  r'motion_39d_30fps|audio|data(-.*)?|preds|human(_v2)?|code|g1_(raw|filtered)_seed\d+(_.*)?)$')
BASELINE_DIRS = ('baseline-repro-1104', 'baselines-g1-1105', 'baselines')
BASELINE_TAG = re.compile(r'echog|emage|lsm|robogesture|semtalk|audio_text', re.I)
OTHER_TEST = re.compile(r'(^|[_\-.])test([_\-.]|$)|val_common', re.I)


def walk(root, max_depth, max_entries, log):
    q = deque([(root, 0)])
    while q:
        d, k = q.popleft()
        try: ents = list(os.scandir(d))
        except OSError as e: log.append(f'unreadable {d}: {e}'); continue
        if len(ents) > max_entries and k > 0: log.append(f'not descended (>{max_entries} entries): {d}'); continue
        for e in ents:
            if e.is_dir(follow_symlinks=False):
                if k < max_depth and not SKIP.match(e.name): q.append((e.path, k + 1))
            elif e.is_file(follow_symlinks=False): yield e


def read_list(path, cache):
    if path not in cache:
        try: cache[path] = [s.strip() for s in open(path) if s.strip()]
        except OSError: cache[path] = None
    return cache[path]


def g(m, *ks):
    for k in ks:
        if not isinstance(m, dict) or k not in m: return None
        m = m[k]
    return m


def row_of(path, base, vc, cache):
    try: d = json.load(open(path))
    except Exception as e: return dict(path=os.path.relpath(path, base), error=repr(e))
    m = d.get('metrics', {}); ss = g(d, 'params', 'stem_selection') or {}
    vs = ss.get('val_split'); n = d.get('n_clips')
    lst = read_list(vs, cache) if vs else None
    if lst is not None:
        k = len(set(lst) & vc)
        test = k > 0; how = f'list {k}/{len(vc)}' + (' (full)' if k == len(vc) else '')
    else:
        name = os.path.basename(vs or '')
        test = name.startswith('val_common') or n == len(vc); how = 'by-name' if test else 'unknown'
    st = os.stat(path)
    return dict(path=os.path.relpath(path, base), dir=os.path.relpath(os.path.dirname(path), base), file=os.path.basename(path),
                tag=d.get('tag'), n_clips=n, val_split=vs, test=test, how=how,
                mtime=time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime)), mtime_s=st.st_mtime,
                FGD=m.get('FGD'), dDiv=m.get('Div_gap'), dBA=m.get('BA_gap'), dJerkW=m.get('Jerk_length_weighted_gap'),
                dJerk_clip=m.get('Jerk_gap'), MM=m.get('Multimodality'), foot=g(m, 'foot_ground_error', 'mean'),
                slide=g(m, 'contact_sliding_speed', 'mean'), fgd_feature=g(d, 'params', 'fgd_feature'))


def scan(a):
    vc = set(read_list(a.val_common, {}) or [])
    if len(vc) != 3242: raise SystemExit(f'val_common list {a.val_common}: {len(vc)} stems (expected 3242)')
    rows, other, log, cache = [], [], [], {}
    for r in a.roots:
        root = os.path.join(a.base, r)
        if not os.path.isdir(root): log.append(f'missing root {root}'); continue
        for e in walk(root, a.max_depth, a.max_entries, log):
            if e.name.startswith('benchmark') and e.name.endswith('.json'): rows.append(row_of(e.path, a.base, vc, cache))
            elif OTHER_TEST.search(e.name) and e.name.endswith(('.json', '.jsonl', '.txt')):
                other.append(dict(path=os.path.relpath(e.path, a.base), mtime=time.strftime('%Y-%m-%d %H:%M', time.localtime(e.stat().st_mtime))))
    out = dict(base=a.base, roots=a.roots, val_common=a.val_common, scanned_at=time.strftime('%Y-%m-%d %H:%M'),
               rows=rows, other_test_files=other, log=log)
    s = json.dumps(out, indent=1)
    if a.out == '-': sys.stdout.write(s + '\n')
    else: open(a.out, 'w').write(s)


def fmt(v, p=3):
    return '' if v is None else (f'{v:.{p}f}' if isinstance(v, float) else str(v))


def dist(xs):
    xs = sorted(xs)
    if not xs: return {}
    q = lambda p: xs[min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))]
    return dict(n=len(xs), min=xs[0], p25=q(.25), median=statistics.median(xs), p75=q(.75), max=xs[-1],
                mean=statistics.fmean(xs), std=statistics.pstdev(xs))


def report(a):
    d = json.load(open(a.inp)); rows = [r for r in d['rows'] if r.get('test')]
    rows.sort(key=lambda r: (r['dir'], r['file']))
    seen = {}
    for i, r in enumerate(rows, 1):   # identical metric record = the same predictions evaluated twice (copied / re-run)
        key = (r['n_clips'], r['FGD'], r['dDiv'], r['dBA'], r['dJerkW'])
        r['dup_of'] = seen.get(key); seen.setdefault(key, i)
        r['family'] = 'baseline' if r['dir'].startswith(BASELINE_DIRS) or BASELINE_TAG.search(r['tag'] or '') else 'scmg'
    L = [f"# Test-set (val_common) usage log\n\nscanned {d['scanned_at']} under {d['base']}; roots: {', '.join(d['roots'])}\n",
         f"{len(rows)} benchmark json on val_common ({sum(1 for r in d['rows'] if not r.get('test'))} other benchmark json not on the test set)\n",
         '| # | directory | configuration (file / tag) | clips | split check | mtime | FGD | ΔDiv | ΔBA | \\|ΔJerkW\\| | foot | slide | note |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['dir']} | {r['file'].removeprefix('benchmark_').removesuffix('.json')} / {r['tag']} | {r['n_clips']} | {r['how']} | "
                 f"{r['mtime']} | {fmt(r['FGD'])} | {fmt(r['dDiv'])} | {fmt(r['dBA'])} | {fmt(r['dJerkW'], 2)} | {fmt(r['foot'], 4)} | {fmt(r['slide'], 3)} | "
                 f"{r['family']}{'; same numbers as #' + str(r['dup_of']) if r['dup_of'] else ''} |")
    uniq = [r for r in rows if r['FGD'] is not None and not r['dup_of']]
    full = [r for r in uniq if r['n_clips'] == 3242 and r['family'] == 'scmg']
    L.append('\n## FGD distribution over configurations that looked at the test set (duplicates removed)\n')
    for name, rs in (('SCMG runs, full val_common (3242 clips)', full),
                     ('SCMG runs, full val_common, offline protocol only', [r for r in full if 'offline' in (r['tag'] or '')]),
                     ('SCMG runs, full val_common, engine protocol only', [r for r in full if 'engine' in (r['tag'] or '')]),
                     ('baselines, full val_common', [r for r in uniq if r['n_clips'] == 3242 and r['family'] == 'baseline']),
                     ('all runs on val_common subsets', [r for r in uniq if r['n_clips'] != 3242])):
        s = dist([r['FGD'] for r in rs])
        if s: L.append(f"- {name}: n={s['n']}, min {s['min']:.3f}, p25 {s['p25']:.3f}, median {s['median']:.3f}, p75 {s['p75']:.3f}, "
                       f"max {s['max']:.3f}, mean {s['mean']:.3f} ± {s['std']:.3f}")
    if full:
        best = sorted(full, key=lambda r: r['FGD'])[:5]
        L.append('- lowest five SCMG runs on the full set: ' + '; '.join(f"{r['dir']}/{r['file']} {r['FGD']:.3f}" for r in best))
    if d.get('other_test_files'):
        L.append('\n## Other files on the test set (not benchmark json; listed for completeness)\n')
        L += [f"- {o['path']} ({o['mtime']})" for o in d['other_test_files']]
    if d.get('log'): L.append('\n## Walk notes\n'); L += [f'- {x}' for x in d['log']]
    md = '\n'.join(L) + '\n'
    if a.md: open(a.md, 'w').write(md)
    print(md)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); sp = ap.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('scan'); s.add_argument('--base', default=B_DEFAULT); s.add_argument('--roots', nargs='+', default=ROOTS)
    s.add_argument('--val-common', default=VAL_COMMON); s.add_argument('--max-depth', type=int, default=6)
    s.add_argument('--max-entries', type=int, default=400); s.add_argument('--out', default='-')
    r = sp.add_parser('report'); r.add_argument('--in', dest='inp', required=True); r.add_argument('--md', default=None)
    a = ap.parse_args(); scan(a) if a.cmd == 'scan' else report(a)
