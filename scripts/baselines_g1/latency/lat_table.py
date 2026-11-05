"""Markdown tables from the lat_*.py JSONs.   lat_table.py <res dir>"""
import json, sys
from pathlib import Path

R = Path(sys.argv[1])
ORDER = ['scmg_offline', 'scmg_engine', 'echog', 'robogesture', 'semtalk', 'emage', 'lsm', 'gmr_semtalk', 'gmr_lsm', 'gmr_emage']
S = {p.stem: json.load(open(p))['summary'] for p in R.glob('*.json')}
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
