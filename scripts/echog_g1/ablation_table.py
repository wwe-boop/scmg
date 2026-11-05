"""11-05 G1 ablation summary table (paper M13): input-modality rows (modality_ablation.sh) + module rows from EXISTING lanes
(nothing is re-run here).  python scripts/echog_g1/ablation_table.py <modality out root> <out.json> [<out.md>]
Metrics (official ECHO-G evaluator): FGD, dDiv (|Div - Div_GT|, signed in brackets), dBA, dJerkW (Jerk_length_weighted_gap),
foot ground error, contact sliding speed (MM needs 20 samples per clip: not part of these single-sample runs); semantic
pair_win / det_auc (mean of 3 recogniser seeds) where scored; text sensitivity for the modality rows."""
import json, sys
from pathlib import Path

B = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg')
SF, HS, CF, BR = B / 'echog-g1-sf-1105', B / 'echog-g1-hos-sweep-1105', B / 'echog-g1-confirm-1105/decft-w3d0m05', B / 'echog-g1-borrow-1105/dev'
AR = B / 'echog-g1-arch-1104'
# (group, label, side, protocol, path) -- dev = held-out-speaker dev (400 clips), test = val_common (3242 clips)
MODULE = [
    ('streaming training', 'off: hos-base (offline training only)', 'dev', 'offline', HS / 'decft-w3d0m05/benchmark_ref_offline_seed_000.json'),
    ('streaming training', 'off: hos-base (offline training only)', 'dev', 'engine', HS / 'decft-w3d0m05/benchmark_ref_engine_seed_000.json'),
    ('streaming training', 'on: sf0 (selected)', 'dev', 'offline', SF / 'dev/sf0/bench/offline/benchmark_seed_000.json'),
    ('streaming training', 'on: sf0 (selected)', 'dev', 'engine', SF / 'dev/sf0/bench/engine/benchmark_seed_000.json'),
    ('streaming training', 'off: fix-base + decft', 'test', 'offline', CF / 'fixbase-s0/benchmark_offline_seed_000.json'),
    ('streaming training', 'off: fix-base + decft', 'test', 'engine', CF / 'fixbase-s0/benchmark_engine_seed_000.json'),
    ('streaming training', 'on: sf0 (selected)', 'test', 'offline', SF / 'test/sf0/bench/offline/benchmark_seed_000.json'),
    ('streaming training', 'on: sf0 (selected)', 'test', 'engine', SF / 'test/sf0/bench/engine/benchmark_seed_000.json'),
    ('self-forcing', 'ctrl: p 0.5, 4 steps, no CFG', 'dev', 'offline', SF / 'dev/ctrl/bench/offline/benchmark_seed_000.json'),
    ('self-forcing', 'ctrl: p 0.5, 4 steps, no CFG', 'dev', 'engine', SF / 'dev/ctrl/bench/engine/benchmark_seed_000.json'),
    ('self-forcing', 'off: sf0 (selected)', 'dev', 'offline', SF / 'dev/sf0/bench/offline/benchmark_seed_000.json'),
    ('self-forcing', 'off: sf0 (selected)', 'dev', 'engine', SF / 'dev/sf0/bench/engine/benchmark_seed_000.json'),
    ('self-forcing', '10 steps', 'dev', 'offline', SF / 'dev/sfs10/bench/offline/benchmark_seed_000.json'),
    ('self-forcing', '10 steps', 'dev', 'engine', SF / 'dev/sfs10/bench/engine/benchmark_seed_000.json'),
    ('self-forcing', '10 steps + CFG 2.5 (= sampler)', 'dev', 'offline', SF / 'dev/sfm/bench/offline/benchmark_seed_000.json'),
    ('self-forcing', '10 steps + CFG 2.5 (= sampler)', 'dev', 'engine', SF / 'dev/sfm/bench/engine/benchmark_seed_000.json'),
    ('self-forcing', 'ctrl', 'test', 'offline', SF / 'test/ctrl/bench/offline/benchmark_seed_000.json'),
    ('self-forcing', 'ctrl', 'test', 'engine', SF / 'test/ctrl/bench/engine/benchmark_seed_000.json'),
    ('self-forcing', 'off: sf0 (selected)', 'test', 'offline', SF / 'test/sf0/bench/offline/benchmark_seed_000.json'),
    ('self-forcing', 'off: sf0 (selected)', 'test', 'engine', SF / 'test/sf0/bench/engine/benchmark_seed_000.json'),
    ('decoder', 'original VAE (hos-base)', 'dev', 'offline', HS / 'vae-orig/benchmark_ref_offline_seed_000.json'),
    ('decoder', 'contact-aware decft-w3d0m05 (hos-base)', 'dev', 'offline', HS / 'decft-w3d0m05/benchmark_ref_offline_seed_000.json'),
    ('decoder', 'original VAE (fix-base)', 'test', 'offline', AR / 'fix-base/benchmark_offline_seed_000.json'),
    ('decoder', 'original VAE (fix-base)', 'test', 'engine', AR / 'fix-base/benchmark_engine_seed_000.json'),
    ('decoder', 'contact-aware decft-w3d0m05 (fix-base)', 'test', 'offline', CF / 'fixbase-s0/benchmark_offline_seed_000.json'),
    ('decoder', 'contact-aware decft-w3d0m05 (fix-base)', 'test', 'engine', CF / 'fixbase-s0/benchmark_engine_seed_000.json'),
    ('text route', 'sf0 (planner text only)', 'dev', 'offline', SF / 'dev/sf0/bench/offline/benchmark_seed_000.json'),
    ('text route', 'sf0 (planner text only)', 'dev', 'engine', SF / 'dev/sf0/bench/engine/benchmark_seed_000.json'),
    ('text route', '+ word-time flow text attention (sf0ftx)', 'dev', 'offline', BR / 'sf0ftx/bench/offline/benchmark_seed_000.json'),
    ('text route', '+ word-time flow text attention (sf0ftx)', 'dev', 'engine', BR / 'sf0ftx/bench/engine/benchmark_seed_000.json'),
    ('text encoder', 'Qwen3-4B features (sf0llm)', 'dev', 'offline', BR / 'sf0llm/bench/offline/benchmark_seed_000.json'),
    ('text encoder', 'Qwen3-4B features (sf0llm)', 'dev', 'engine', BR / 'sf0llm/bench/engine/benchmark_seed_000.json'),
]
for lab, f in (('engine ref (H 160 ms, x2 text)', 'ref'), ('H 320 ms', 'h4'), ('H 640 ms', 'h8'), ('H 320 ms + sentence text', 'h4tf'),
               ('H 640 ms + sentence text', 'h8tf'), ('H inf + sentence text', 'hinftf')):
    MODULE.append(('inference protocol (sf0)', lab, 'dev', 'engine', SF / f'dev/sf0/tts-sweep/benchmark_{f}_engine_seed_000.json'))
for lab, f in (('engine ref', 'ref'), ('engine - H inf', 'hinf'), ('engine - sentence text', 'tfinal'), ('engine - deploy timing', 'nodt')):
    MODULE.append(('inference protocol (hos-base, leave-one-out)', lab, 'dev', 'engine', HS / f'decft-w3d0m05/benchmark_{f}_engine_seed_000.json'))
for lab, f in (('CFG 2.5 (ref)', 'ref'), ('CFG 1.5', 'cfg15'), ('CFG 2.0', 'cfg20'), ('CFG 3.0', 'cfg30'), ('null speaker', 'nospk'), ('noise x0.9', 'ns09')):
    MODULE.append(('inference guidance (hos-base)', lab, 'dev', 'offline', HS / f'decft-w3d0m05/benchmark_{f}_offline_seed_000.json'))


def metrics(f):
    m = json.loads(Path(f).read_text())['metrics']
    return dict(n=json.loads(Path(f).read_text())['n_clips'], FGD=m['FGD'], dDiv=m['Div_gap'], dDiv_signed=m['Div_signed_gap'], dBA=m['BA_gap'],
                dJerkW=m['Jerk_length_weighted_gap'], dJerkW_signed=m['Jerk_length_weighted_signed_gap'],
                foot=m['foot_ground_error']['mean'], slide=m['contact_sliding_speed']['mean'])


def main():
    O = Path(sys.argv[1]); rows = []
    for side in ('dev', 'test'):
        R = O / side
        sem = {}
        rp = O / f'sem-{side}' / 'REPORT.json'
        if rp.exists():
            sem = json.loads(rp.read_text()).get('table', {})
        ts = json.loads((O / f'TEXT_SENS-{side}.json').read_text()) if (O / f'TEXT_SENS-{side}.json').exists() else {}
        for f in sorted(R.glob('benchmark_*.json')) if R.exists() else []:
            tag = f.stem[len('benchmark_'):]; lab, p, _, sd = tag.rsplit('_', 3)
            r = dict(group='input modality', label=lab, side=side, protocol=p, seed=int(sd), source=str(f), **metrics(f))
            k = f'{lab}_{p}'
            if sd == '000' and k in sem:
                r.update({mm: sem[k][mm]['mean'] for mm in ('pair_win', 'det_auc') if mm in sem[k]})
            if f'{lab}_{p}' in ts: r['text_sens_upper'] = ts[f'{lab}_{p}']['upper']['ratio_of_means']; r['d_swap_upper'] = ts[f'{lab}_{p}']['upper']['d_swap']
            rows.append(r)
        if 'gt' in sem: rows.append(dict(group='input modality', label='ground truth', side=side, protocol='-', **{mm: sem['gt'][mm]['mean'] for mm in ('pair_win', 'det_auc')}))
    for g, lab, side, p, f in MODULE:
        if Path(f).exists(): rows.append(dict(group=g, label=lab, side=side, protocol=p, seed=0, source=str(f), **metrics(f)))
        else: rows.append(dict(group=g, label=lab, side=side, protocol=p, missing=str(f)))
    Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
    if len(sys.argv) > 3:
        cols = ('FGD', 'dDiv', 'dBA', 'dJerkW', 'foot', 'slide', 'pair_win', 'det_auc', 'text_sens_upper')
        fmt = lambda v: '' if v is None else (f'{v:.4f}' if abs(v) < 0.1 else f'{v:.3f}')
        L = ['| group | setting | side | protocol | seed | ' + ' | '.join(cols) + ' |', '|' + '---|' * (5 + len(cols))]
        for r in rows:
            L.append(f"| {r['group']} | {r['label']} | {r['side']} | {r['protocol']} | {r.get('seed', '')} | " +
                     ' | '.join(fmt(r.get(c)) for c in cols) + ' |' + (' MISSING' if 'missing' in r else ''))
        Path(sys.argv[3]).write_text('\n'.join(L) + '\n')
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
