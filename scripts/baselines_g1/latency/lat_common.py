"""Shared latency / memory / RTF harness for the G1 main-table methods (batch 1, inference settings = benchmark).

Clips: the 300-clip RoboGesture subset, every 6th stem -> 50 clips; the first 3 are warm-up and excluded from every
statistic.  Each method runs in its own process (one GPU, CUDA_VISIBLE_DEVICES), records per clip
  dur (s), stage seconds (cuda-synchronised wall time), compute = sum of stages, and an emission schedule
  [(data_ready_s, compute_s, frame_t0_s, n_frames, fps)]  -- tick k can start once the audio/text it needs has arrived
  (data_ready_s, real-time arrival from t = 0) and tick k-1 is done; its frames [t0, t0 + n/fps) are output at the end.
Derived (simulate()):
  algorithmic latency = max over frames of (data_ready - frame time) with zero compute (the pure look-ahead wait)
  end-to-end latency  = per frame (output time - frame time), p50 / p95 over all frames of all clips
  first-frame latency = output time of the first motion frame (first audio sample arrives at t = 0)
  RTF                 = compute / audio duration (per clip, and sum / sum)
Memory: torch.cuda.max_memory_allocated / max_memory_reserved; nvidia-smi used_memory of the PIDs that appeared on our
GPU after this process started (container PID namespaces differ from nvidia-smi's, so PIDs are matched by appearance);
device memory.used - baseline as a fallback; CPU ru_maxrss.
"""
from __future__ import annotations

import json, os, resource, subprocess, threading, time
from pathlib import Path

import numpy as np

SUBSET = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/robogesture/subset_stems.txt')
N_WARM = 3


def stems50():
    s = [l.strip() for l in open(SUBSET) if l.strip()]
    return s[::6][:50]


class Sync:
    """cuda-synchronised stopwatch."""
    def __init__(self, cuda=True):
        import torch
        self.t = torch; self.cuda = cuda and torch.cuda.is_available()

    def now(self):
        if self.cuda: self.t.cuda.synchronize()
        return time.perf_counter()


class GPUMon:
    """Background sampler of nvidia-smi for the GPU this process uses (index from CUDA_VISIBLE_DEVICES)."""
    def __init__(self, period=0.2):
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        self.idx = vis
        q = subprocess.run(['nvidia-smi', '-i', vis, '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.strip().split(', ')
        self.uuid = q[0]; self.base_used = float(q[1]); self.base_util = float(q[2])
        self.base_pids = set(self._apps())
        self.peak_proc = 0.0; self.peak_used = self.base_used; self.util = []
        self.period = period; self._stop = False
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start()

    def _apps(self):
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True).stdout
        r = {}
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(',')]
            if len(p) == 3 and p[2] == self.uuid:
                try: r[p[0]] = float(p[1])
                except ValueError: pass
        return r

    def _run(self):
        while not self._stop:
            try:
                a = self._apps(); new = sum(v for k, v in a.items() if k not in self.base_pids)
                self.peak_proc = max(self.peak_proc, new)
                q = subprocess.run(['nvidia-smi', '-i', self.idx, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True).stdout.strip().split(', ')
                self.peak_used = max(self.peak_used, float(q[0])); self.util.append(float(q[1]))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self._stop = True; self.th.join(timeout=5)
        return dict(gpu_index_visible=self.idx, gpu_uuid=self.uuid, nvsmi_proc_peak_mib=self.peak_proc,
                    nvsmi_device_peak_minus_baseline_mib=self.peak_used - self.base_used, device_baseline_used_mib=self.base_used,
                    device_baseline_util=self.base_util, device_util_mean=float(np.mean(self.util)) if self.util else None)


def host_snapshot():
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    return dict(hostname=sh('hostname'), loadavg=sh('cat /proc/loadavg'), nproc=sh('nproc'),
                affinity=sorted(os.sched_getaffinity(0)), gpus=sh('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader'),
                time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def simulate(rec):
    """rec: dict(dur, schedule=[(ready, compute, t0, n, fps)]). Returns per-frame latencies, algorithmic max, first frame."""
    done = 0.0; lat = []; alg = 0.0; first = None
    for ready, comp, t0, n, fps in rec['schedule']:
        start = max(done, ready); done = start + comp
        if n <= 0: continue
        t = t0 + np.arange(n) / fps
        lat.append(done - t); alg = max(alg, float(np.max(ready - t)))
        if first is None: first = done
    lat = np.concatenate(lat) if lat else np.zeros(0)
    return lat, alg, first


def summarize(rows, extra):
    use = rows[N_WARM:]
    lats, algs, firsts = [], [], []
    for r in use:
        l, a, f = simulate(r); lats.append(l); algs.append(a); firsts.append(f)
    L = np.concatenate(lats)
    comp = np.array([r['compute'] for r in use]); dur = np.array([r['dur'] for r in use])
    stages = sorted({k for r in use for k in r['stages']})
    st = {k: dict(ms_per_clip_p50=float(np.median([r['stages'].get(k, 0.) for r in use]) * 1e3),
                  ms_per_clip_mean=float(np.mean([r['stages'].get(k, 0.) for r in use]) * 1e3),
                  rtf=float(sum(r['stages'].get(k, 0.) for r in use) / dur.sum())) for k in stages}
    out = dict(n_clips=len(use), n_warmup=N_WARM, audio_s_total=float(dur.sum()), dur_p50=float(np.median(dur)),
               algorithmic_latency_s=dict(p50=float(np.median(algs)), max=float(np.max(algs))),
               e2e_latency_s=dict(p50=float(np.percentile(L, 50)), p95=float(np.percentile(L, 95)), max=float(L.max())),
               first_frame_latency_s=dict(p50=float(np.median(firsts)), p95=float(np.percentile(firsts, 95))),
               rtf=dict(per_clip_p50=float(np.median(comp / dur)), per_clip_p95=float(np.percentile(comp / dur, 95)),
                        aggregate=float(comp.sum() / dur.sum())),
               compute_s_per_clip_p50=float(np.median(comp)), stages=st,
               cpu_rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    tk = [t for r in use for t in r.get('ticks', [])]
    if tk:
        st_keys = [k for k in tk[0] if k not in ('t', 'len')]
        ratio = np.array([sum(t[k] for k in st_keys) / t['len'] for t in tk])
        out['per_tick_rtf'] = dict(n=int(len(ratio)), mean=float(ratio.mean()), p50=float(np.median(ratio)),
                                   p95=float(np.percentile(ratio, 95)), max=float(ratio.max()), frac_over_1=float((ratio > 1).mean()),
                                   tick_len_s=float(np.median([t['len'] for t in tk])))
        b = {}
        for t in tk: b.setdefault(int(t['t'] // 2) * 2, []).append(t)
        out['per_tick_by_2s'] = [dict(t0_s=k0, n=len(v), **{f'{k}_ms_mean': float(np.mean([x[k] for x in v]) * 1e3) for k in st_keys},
                                      rtf_mean=float(np.mean([sum(x[k] for k in st_keys) / x['len'] for x in v])),
                                      rtf_max=float(np.max([sum(x[k] for k in st_keys) / x['len'] for x in v]))) for k0, v in sorted(b.items())]
    out.update(extra)
    return out


def cuda_mem():
    import torch
    if not torch.cuda.is_available(): return {}
    return dict(torch_max_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                torch_max_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20, gpu_name=torch.cuda.get_device_name(0))


def write(out, method, rows, extra):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    s = summarize(rows, extra)
    (out / f'{method}.json').write_text(json.dumps(dict(method=method, summary=s, rows=rows), indent=1))
    print('SUMMARY', method, json.dumps({k: s[k] for k in ('algorithmic_latency_s', 'e2e_latency_s', 'first_frame_latency_s', 'rtf')}), flush=True)
    return s
