"""P2 sem_cf render check (11-09; main: 4 events, panels none / given (oracle) / swapped / retrieval, front + 3/4 (45 deg) view, the
descriptor printed on each panel). Selection rule, fixed and committed before rendering: the 62 oracle events of the quick lane
(p2-quick-1109/plans/PLAN_oracle.json) ranked by sha1("<clip>:<ev_t:.3f>:p2render"); an event qualifies if its 10 s window
[ev_t - 5, ev_t + 5] lies inside the recording; the first 4 qualifying events from 4 different recordings. Outputs of the quick
lane (offline, seed 0). Overlay per panel: commanded-hand wrist marker inside W = [ev_t - 0.6, ev_t + 0.6] (yellow), the commanded
direction from the shoulder (0.3 m, yellow), and text lines: the commanded descriptor (physical units: amp_h m, vpeak m/s, z_end m,
dir), the realised descriptor of the panel's motion at ev_t, E(realised, commanded) and E(realised, GT).
    python qual_p2.py --out OUT   (then render_qual_isaac.py --jobs OUT/jobs.json --headless --enable_cameras)"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002'; P2 = X + '/p2-quick-1109'; S = X + '/sem-1109'; FPS = 50
h = lambda s: hashlib.sha1(s.encode()).hexdigest()


def main():
    from qual_ex002 import q37, rec_audio
    from scmg import sem_desc as SD, ex002_ctl as E
    import soundfile as sf
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); O = Path(a.out)
    for d in ('traj', 'draw', 'audio', 'video'): (O / d).mkdir(parents=True, exist_ok=True)
    st = json.loads(Path(S + '/SEM_BANK.json').read_text()); mu, sd = np.asarray(st['mu']), np.asarray(st['sd'])
    P = json.loads(Path(P2 + '/plans/PLAN_oracle.json').read_text())['events']
    R = json.loads(Path(P2 + '/plans/PLAN_retrieval.json').read_text())['events']
    L = {}
    def ok(e):
        if e['ci'] not in L: L[e['ci']] = len(np.load(f"{P2}/offline/none_s0/{e['ci']}.npy")) / FPS
        return e['ev_t'] - 5 >= 0 and e['ev_t'] + 5 <= L[e['ci']]
    pick, seen = [], set()
    for e in sorted(P, key=lambda e: h(f"{e['clip']}:{e['ev_t']:.3f}:p2render")):
        if ok(e) and e['ci'] not in seen: pick.append(e); seen.add(e['ci'])
        if len(pick) == 4: break
    std = lambda raw: np.concatenate([(raw[:SD.NS] - mu) / sd, raw[SD.NS:]])
    raw_of = lambda D: np.concatenate([np.asarray(D[:SD.NS]) * sd + mu, D[SD.NS:]])
    fmt = lambda r, hd: f"amp {r[0]:.2f} vpk {r[3]:.2f} zend {r[7]:+.2f} dir ({r[13]:+.1f},{r[14]:+.1f},{r[15]:+.1f}) {'LR'[hd]}"
    jobs, man = [], []
    for n, e in enumerate(pick):
        ci = e['ci']; f0 = int(round((e['ev_t'] - 5) * FPS)); f1 = f0 + 10 * FPS; t = e['ev_t'] - f0 / FPS
        rt = [r for r in R if r['ci'] == ci and abs(r['t'] + 0.2 - e['ev_t']) <= 1.2]
        rr = min(rt, key=lambda r: abs(r['t'] + 0.2 - e['ev_t'])) if rt else None
        rows = [('none (no token)', 'none_s0', None, e['hand']), ('given (oracle)', 'given_s0', e['desc'], e['hand']),
                ('swapped', 'swapped_s0', e['shuffle']['desc'], e['shuffle']['hand']),
                ('retrieval' + (f" '{rr['word']}' ({rr['src']})" if rr else ' (no token within 1.2 s)'), 'retrieval_s0', rr['desc'] if rr else None, rr['hand'] if rr else e['hand'])]
        panels, draws, info = [], [], []
        for lab, d, cmd, hd in rows:
            yfull = np.load(f'{P2}/offline/{d}/{ci}.npy'); y = yfull[f0:f1]
            tp = O / 'traj' / f"ev{n}__p{len(panels)}_{d}.npz"; np.savez(tp, Q37=q37(y), fps=50.0)
            Dr = std(SD.np_desc(SD.window(yfull, e['ev_t']), hd)); rr_ = raw_of(Dr)
            Eg = float(np.sum((Dr - np.asarray(e['desc'])) ** 2) / SD.ND); Ec = float(np.sum((Dr - np.asarray(cmd)) ** 2) / SD.ND) if cmd is not None else None
            g = E.points(y); fr = []
            for k in range(len(y)):
                tk = k / FPS; on = abs(tk - t) <= SD.HALF; dd = dict(seg=[], pts=[], txt=[])
                if on:
                    w = g['wrist'][k, hd]; dd['pts'].append([w.tolist(), [255, 220, 0], 10])
                    if cmd is not None:
                        rc = raw_of(np.asarray(cmd)); sh = g['shoulder'][k, hd]
                        dd['seg'].append([sh.tolist(), (sh + 0.3 * rc[13:16]).tolist(), [255, 220, 0]])
                dd['txt'].append('cmd  ' + (fmt(raw_of(np.asarray(cmd)), hd) if cmd is not None else 'none'))
                dd['txt'].append('real ' + fmt(rr_, hd) + (f'   E(cmd) {Ec:.2f}' if Ec is not None else '') + f'  E(GT) {Eg:.2f}')
                fr.append(dd)
            panels.append(dict(path=str(tp), kind='q37', label=lab)); draws.append(dict(frames=fr, ref_wrist=[g['wrist'][0, 0].tolist(), g['wrist'][0, 1].tolist()]))
            info.append(dict(panel=lab, E_cmd=Ec, E_gt=Eg))
        dp = O / 'draw' / f'ev{n}.json'; dp.write_text(json.dumps(dict(robots=draws)))
        ap_ = O / 'audio' / f'ev{n}.wav'; sf.write(ap_, rec_audio(ci)[f0 * 320:f1 * 320], 16000, subtype='PCM_16')
        gt = raw_of(np.asarray(e['desc']))
        title = f"P2 ev{n} {e['clip']} t={e['ev_t']:.2f}s {e['cls']} kw '{e['kw']}' | GT {fmt(gt, e['hand'])} | yellow = commanded hand / direction in W"
        jobs.append(dict(trajs=panels, audio=str(ap_), out=str(O / 'video' / f'ev{n}.mp4'), title=title, views=['front', 'q34'], draw=str(dp)))
        man.append(dict(name=f'ev{n}', ci=ci, clip=e['clip'], ev_t=e['ev_t'], t_tok=e['t'], cls=e['cls'], kw=e['kw'], window_s=[f0 / FPS, f1 / FPS],
                        retrieval_token=rr, panels=info))
    (O / 'jobs.json').write_text(json.dumps(jobs, indent=1)); (O / 'MANIFEST.json').write_text(json.dumps(man, indent=1)); print('QUAL_P2_PREP', len(jobs))


if __name__ == '__main__':
    main()
