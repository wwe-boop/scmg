"""Markdown tables from the lat_*.py JSONs.   lat_table.py <res dir> [<res dir> ...]"""
import json, sys
from pathlib import Path

ORDER = ['scmg_offline_cgoff', 'scmg_offline_cgon', 'scmg_psync_cgoff', 'scmg_psync_cgon', 'scmg_psync_cgoff_grades', 'scmg_psync_cgon_grades', 'scmg_psync_cgoff_grades_inc', 'scmg_psync_cgon_grades_inc', 'scmg_engine_cgon', 'scmg_offline', 'scmg_psync', 'scmg_engine', 'echog', 'echog_stream', 'echog_stream_nocudnn', 'robogesture', 'semtalk', 'emage', 'lsm', 'gmr_semtalk', 'gmr_lsm', 'gmr_emage']
S = {p.stem: json.load(open(p))['summary'] for R in sys.argv[1:] for p in Path(R).glob('*.json')}
f = lambda x, d=2: '-' if x is None else f'{x:.{d}f}'
print('| method | algorithmic s (p50) | e2e p50 / p95 s | first frame s (p50) | RTF agg (clip p50) | GPU torch peak / nvsmi MiB | CPU RSS MiB |')
print('|---|---|---|---|---|---|---|')
for m in [m for m in ORDER if m in S]:
    s = S[m]
    print(f"| {m} | {f(s['algorithmic_latency_s']['p50'])} | {f(s['e2e_latency_s']['p50'])} / {f(s['e2e_latency_s']['p95'])} | "
          f"{f(s['first_frame_latency_s']['p50'])} | {f(s['rtf']['aggregate'], 3)} ({f(s['rtf']['per_clip_p50'], 3)}) | "
          f"{f(s.get('torch_max_allocated_mib'), 0)} / {f(s.get('nvsmi_proc_peak_mib'), 0)} | {f(s['cpu_rss_peak_mib'], 0)} |")
print('\n| method | stage | ms per clip (p50) | RTF |'); print('|---|---|---|---|')
for m in [m for m in ORDER if m in S]:
    for k, v in S[m]['stages'].items():
        print(f"| {m} | {k} | {f(v['ms_per_clip_p50'], 1)} | {f(v['rtf'], 4)} |")
    if 'filter_ms_per_clip_p50' in S[m]:
        print(f"| {m} | collision filter (CPU, deploy only) | {f(S[m]['filter_ms_per_clip_p50'], 1)} | {f(S[m]['filter_rtf'], 4)} |")
for m in [m for m in ORDER if m in S]:
    if 'deploy_whole_utterance' in S[m]:
        d = S[m]['deploy_whole_utterance']
        print(f"\n{m} official deployment (whole-utterance playback, + filter): algorithmic {f(d['algorithmic_latency_s']['p50'])} s, "
              f"e2e p50/p95 {f(d['e2e_latency_s']['p50'])}/{f(d['e2e_latency_s']['p95'])} s, first frame {f(d['first_frame_latency_s']['p50'])} s, RTF {f(d['rtf']['aggregate'], 3)}")
m0 = next(iter(S.values()))
print('\nhost at start of each method:')
for m in [m for m in ORDER if m in S]:
    h = S[m].get('host_start', {}); print(f"- {m}: {h.get('hostname')} load {h.get('loadavg')} | GPU{S[m].get('gpu_index_visible')} baseline {S[m].get('device_baseline_used_mib')} MiB util {S[m].get('device_baseline_util')}% | {S[m].get('gpu_name')} | cpus {h.get('affinity')}")
print('\nper-tick RTF (tick compute / tick length; real time needs p95 and max < 1):')
print('| method | tick s | n | mean | p50 | p95 | max | share > 1 |'); print('|---|---|---|---|---|---|---|---|')
for m in [m for m in ORDER if m in S and 'per_tick_rtf' in S[m]]:
    t = S[m]['per_tick_rtf']
    print(f"| {m} | {t['tick_len_s']:.2f} | {t['n']} | {t['mean']:.3f} | {t['p50']:.3f} | {t['p95']:.3f} | {t['max']:.3f} | {t['frac_over_1']:.3f} |")
for m in [m for m in ORDER if m in S and 'per_tick_by_2s' in S[m]]:
    b = S[m]['per_tick_by_2s']; ks = [k for k in b[0] if k.endswith('_ms_mean')]
    print(f'\n{m}: per-tick cost by tick start time (2 s buckets, mean ms per tick)')
    print('| t0 s | n | ' + ' | '.join(k[:-8] for k in ks) + ' | tick RTF mean | max |'); print('|' + '---|' * (len(ks) + 4))
    for r in b:
        print(f"| {r['t0_s']} | {r['n']} | " + ' | '.join(f"{r[k]:.1f}" for k in ks) + f" | {r['rtf_mean']:.3f} | {r['rtf_max']:.3f} |")
