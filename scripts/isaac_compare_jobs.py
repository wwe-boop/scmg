"""Isaac Sim side-by-side jobs (codex/render-isaac/render_ex002_isaac.py --jobs) for SCMG EX002 34-joint outputs.

  ~/envs/rlsim/bin/python scripts/isaac_compare_jobs.py dev <out dir> <horizon> <ci,...> <label=eval_dir> [...]
      columns: 真人原动作 (gt_<ci>.npy of the first eval dir) + gen_H<h>_<ci>.npy of every eval dir; the audio is cut
      from AUDIO16 exactly as scripts/render_compare.py does (same clip start, n = shortest column).
  ~/envs/rlsim/bin/python scripts/isaac_compare_jobs.py tts <out dir> <replay dir> <label> [<label> ...]
      columns: <replay dir>/<label>/<id>.npy (scripts/tts_deploy_replay.py), audio <replay dir>/<id>.wav, title = sentence.
  ~/envs/rlsim/bin/python scripts/isaac_compare_jobs.py fkcheck <out dir>
      (after rendering) MuJoCo model-r1 FK of the RAW 34-joint arrays placed exactly as render_compare.py does (lift .32,
      lift_driven .16) vs the converted q37 npz (traj_io.load, what Isaac replays) on every column, and Isaac fk_probe
      (render log, column 0) vs MuJoCo FK of the npz.  Bodies under the thumb mimic joint x_1_4 are excluded from the
      raw-vs-npz test (render_compare leaves x_1_4 = 0; the npz restores x_1_4 = 1.389831 x_1_3 + 0.072772).
34 -> 37: META.json ex002.restore (lift 0.32 m, mimic x_1_4 = 1.389831 * x_1_3 + 0.072772), columns by joint NAME.
"""
import json, sys, wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RI = Path('/home/xr-25b7/projects/gesture/codex/render-isaac')
sys.path.insert(0, str(RI))
import traj_io  # noqa: E402  (read-only import; JOINT_ORDER / loaders of the Isaac renderer)

V = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views')
NAMES34 = json.loads((V / 'data-v36_1/META.json').read_text())['joints']
MIMIC_A, MIMIC_B, LIFT = 1.389831, 0.072772, 0.32
VIEWS = ['front', 'q34']


def to37(q34):
    q34 = np.asarray(q34, np.float64)
    assert q34.ndim == 2 and q34.shape[1] == 34 and np.isfinite(q34).all(), q34.shape
    q = np.zeros((len(q34), 37))
    for j, n in enumerate(NAMES34):
        q[:, traj_io.JOINT_ORDER.index(n)] = q34[:, j]
    q[:, 0] = LIFT
    for s in ('left', 'right'):
        q[:, traj_io.JOINT_ORDER.index(f'{s}_1_4_joint')] = MIMIC_A * q[:, traj_io.JOINT_ORDER.index(f'{s}_1_3_joint')] + MIMIC_B
    return q


def save_npz(path, q34, src):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, q37=to37(q34), joint_names=np.array(traj_io.JOINT_ORDER), fps=50.0, source=str(src))
    return dict(path=str(path), kind='q37', src=str(src))


def write_wav(path, x16):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(np.asarray(x16, np.int16).tobytes())


def dev(out, hz, cis, cols):
    sys.path.insert(0, str(HERE))
    from render_compare import val_clips      # the same clip order / start offsets as the MuJoCo version
    clips = val_clips()
    audio = np.load(V / 'data-v36_1-audio16/AUDIO16.npy', mmap_mode='r')
    jobs, index = [], []
    for ci in cis:
        srcs = [Path(cols[0][1]) / f'gt_{ci}.npy'] + [Path(p) / f'gen_H{hz}_{ci}.npy' for _, p in cols]
        qs = [np.load(s) for s in srcs]
        n = min(len(x) for x in qs)
        labels = ['真人原动作'] + [l for l, _ in cols]
        c = clips[ci]
        wav = out / 'audio' / f'clip{ci}.wav'; wav.parent.mkdir(parents=True, exist_ok=True)
        write_wav(wav, audio[c['start'] * 320:(c['start'] + n) * 320])
        trajs = []
        for k, (s, q) in enumerate(zip(srcs, qs)):
            t = save_npz(out / 'traj' / f'clip{ci}_col{k}.npz', q[:n], s); t['label'] = labels[k]; trajs.append(t)
        jobs.append(dict(trajs=trajs, audio=str(wav), out=str(out / f'clip{ci}.mp4'), views=VIEWS,
                         title=f'开发集 clip{ci}  {c["clip"]}  流式 H={hz}（160 ms 前瞻）   左：真人原动作 ｜ 中：修复前 sibag-s2 ｜ 右：修复后 sibag-coll-s1'))
        index.append(dict(clip=c['clip'], ci=ci, video=f'clip{ci}.mp4', poster=f'clip{ci}.jpg', seconds=round(n / 50, 1),
                          frames=n, text=c['text'][:200], sources=[str(s) for s in srcs]))
    (out / 'jobs.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=1))
    (out / 'INDEX.json').write_text(json.dumps(dict(renderer='Isaac Sim 5.1 RTX (render_ex002_isaac.py)', horizon=hz,
                                                    columns=['真人原动作'] + [l for l, _ in cols], views=VIEWS, clips=index),
                                               ensure_ascii=False, indent=1))
    print('jobs', len(jobs), out / 'jobs.json')


def tts(out, rd, labels):
    rep = json.loads((rd / 'REPLAY.json').read_text())
    jobs, index = [], []
    for i, s in rep['sentences'].items():
        qs = [np.load(rd / lab / f'{i}.npy') for lab in labels]
        n = min(len(x) for x in qs)
        trajs = []
        for lab, q in zip(labels, qs):
            t = save_npz(out / 'traj' / f'{i}_{lab}.npz', q[:n], rd / lab / f'{i}.npy'); t['label'] = LABEL.get(lab, lab); trajs.append(t)
        jobs.append(dict(trajs=trajs, audio=str(rd / f'{i}.wav'), out=str(out / f'{i}.mp4'), views=VIEWS,
                         title=f'{i}「{s["text"]}」'))
        index.append(dict(id=i, text=s['text'], video=f'{i}.mp4', poster=f'{i}.jpg', seconds=round(n / 50, 2), frames=n,
                          audio_s=s['seconds'], wav=f'{i}.wav', traj={lab: f'{lab}/{i}.npy' for lab in labels}))
    (out / 'jobs.json').write_text(json.dumps(jobs, ensure_ascii=False, indent=1))
    (out / 'INDEX.json').write_text(json.dumps(dict(renderer='Isaac Sim 5.1 RTX (render_ex002_isaac.py)', columns=[LABEL.get(l, l) for l in labels],
                                                    views=VIEWS, models={l: rep['models'][l]['ckpt'] for l in labels},
                                                    infer=rep['models'][labels[0]]['cfg'], clips=index), ensure_ascii=False, indent=1))
    print('jobs', len(jobs), out / 'jobs.json')


LABEL = {'sibag-s2': '修复前 sibag-s2', 'sibag-coll-s1': '修复后 sibag-coll-s1'}


def fkcheck(out):
    import mujoco
    m = mujoco.MjModel.from_xml_path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/ex002-rm003-v4-20261020/model-r1/scene.xml')
    d = mujoco.MjData(m)
    jadr = {m.joint(i).name: m.jnt_qposadr[i] for i in range(m.njnt)}
    adr34 = [jadr[n] for n in NAMES34]
    thumb = set()
    for s in ('left', 'right'):                      # bodies at/below the x_1_4 joint
        b0 = m.jnt_bodyid[m.joint(f'{s}_1_4_joint').id]
        for b in range(m.nbody):
            p = b
            while p > 0 and p != b0: p = m.body_parentid[p]
            if p == b0: thumb.add(b)
    keep = np.array([b for b in range(1, m.nbody) if b not in thumb])

    def fk_raw(q):
        d.qpos[:] = 0; d.qpos[adr34] = q; d.qpos[jadr['lift_joint']] = LIFT; d.qpos[jadr['lift_driven_joint']] = LIFT / 2
        mujoco.mj_kinematics(m, d); return d.xpos.copy()

    def fk37(q):
        d.qpos[:] = 0
        for i, n in enumerate(traj_io.JOINT_ORDER):
            if n in jadr: d.qpos[jadr[n]] = q[i]
        d.qpos[jadr['lift_driven_joint']] = 0.5 * q[0]
        mujoco.mj_kinematics(m, d); return d.xpos.copy()
    jobs = json.loads((out / 'jobs.json').read_text())
    res = dict(raw_vs_npz_mm=0.0, isaac_vs_mujoco_mm=0.0, thumb_bodies_excluded=len(thumb), bodies=int(len(keep)), n=0, per_job=[])
    for j in jobs:
        rj = dict(out=Path(j['out']).name)
        for t in j['trajs']:
            src = Path(t['src']); raw = np.load(src); L = traj_io.load(t['path'], 'auto')
            assert L['meta']['missing'] == [], L['meta']
            T = len(L['q37']); e = 0.0
            for k in range(0, T, 25):
                e = max(e, float(np.linalg.norm(fk_raw(raw[k])[keep] - fk37(L['q37'][k])[keep], axis=1).max()))
            res['raw_vs_npz_mm'] = max(res['raw_vs_npz_mm'], e * 1e3); res['n'] += 1
            rj.setdefault('raw_vs_npz_mm', []).append(round(e * 1e3, 6))
        log = out / 'logs' / (Path(j['out']).stem + '.json')
        if log.exists():
            lg = json.loads(log.read_text()); L = traj_io.load(j['trajs'][0]['path'], 'auto'); e = 0.0
            for row in lg['fk_probe']['frames']:
                xp = fk37(L['q37'][row[0]])
                for b, p in zip(lg['fk_probe']['bodies'], row[1:]):
                    e = max(e, float(np.linalg.norm(xp[m.body(b).id] - np.array(p))))
            assert lg['frames'] == len(L['q37']), (lg['frames'], len(L['q37']))
            rj['isaac_vs_mujoco_mm'] = round(e * 1e3, 4); rj['frames'] = lg['frames']
            res['isaac_vs_mujoco_mm'] = max(res['isaac_vs_mujoco_mm'], e * 1e3)
        res['per_job'].append(rj)
    (out / 'FK_CHECK.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'per_job'}))


if __name__ == '__main__':
    cmd, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    if cmd == 'dev':
        dev(out, sys.argv[3], [int(x) for x in sys.argv[4].split(',')], [a.split('=', 1) for a in sys.argv[5:]])
    elif cmd == 'tts':
        tts(out, Path(sys.argv[3]), sys.argv[4:])
    elif cmd == 'fkcheck':
        fkcheck(out)
