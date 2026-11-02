"""Side-by-side EX002 (RM003 three-finger) render of SCMG evaluation outputs with the original speech.
    render_compare.py <out dir> <horizon> <clip idx,...> <label=eval_dir> [<label=eval_dir> ...]
eval_dir = an evaluate.py output dir (gen_H<h>_<ci>.npy, gt_<ci>.npy; 34 joints, rad, 50 Hz). The first column is the
ground truth (gt_<ci>.npy of the first eval dir). Clip index ci = order of the val clips exactly as data.View builds
them (contiguous utterances of one recording). Runs on speech_server with EGL (physical GPU1)."""
import os
os.environ.update(__EGL_VENDOR_LIBRARY_FILENAMES='/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-r2-20261030/nvidia-egl.json',
                  MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='1')
import json, subprocess, sys, wave
from collections import OrderedDict
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont

V = Path('/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/train-v1-20261101/views')
MODEL = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/ex002-rm003-v4-20261020/model-r1/scene.xml'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 16)
W, H = 300, 340


def val_clips():
    items = json.loads((V / 'data-v36_1/train.json').read_text())
    keep = set(json.loads((V / 'data-v36_1/SPLIT.json').read_text())['val_ids'])
    clips = OrderedDict()
    for it in items:
        if it['id'] in keep: clips.setdefault(it['clip'], []).append(it)
    out = []
    for c, xs in clips.items():
        xs.sort(key=lambda x: x['offset']); out.append(dict(clip=c, start=xs[0]['offset'], text=' '.join(x.get('text', '') for x in xs)))
    return out


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    hz = sys.argv[2]; cis = [int(x) for x in sys.argv[3].split(',')]
    cols = [a.split('=', 1) for a in sys.argv[4:]]
    clips = val_clips()
    audio = np.load(V / 'data-v36_1-audio16/AUDIO16.npy', mmap_mode='r')
    joints = json.load(open(V / 'data-v36_1/META.json'))['joints']
    m = mujoco.MjModel.from_xml_path(MODEL); d = mujoco.MjData(m)
    m.vis.global_.offwidth = W; m.vis.global_.offheight = H; ren = mujoco.Renderer(m, H, W)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in joints]
    lift = m.jnt_qposadr[m.joint('lift_joint').id]; liftd = m.jnt_qposadr[m.joint('lift_driven_joint').id]
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1:3] = 1
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 1.18]; cam.distance = 1.9; cam.azimuth = 180; cam.elevation = -5

    def frame(q):
        d.qpos[:] = 0; d.qpos[adr] = q; d.qpos[lift] = 0.32; d.qpos[liftd] = 0.16
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        ren.update_scene(d, cam, opt); return ren.render()
    index = []
    for ci in cis:
        qs = [np.load(Path(cols[0][1]) / f'gt_{ci}.npy')] + [np.load(Path(p) / f'gen_H{hz}_{ci}.npy') for _, p in cols]
        n = min(len(x) for x in qs); labels = ['真人原动作'] + [l for l, _ in cols]
        c = clips[ci]; a = np.asarray(audio[c['start'] * 320:(c['start'] + n) * 320], np.int16)
        wav = out / f'clip{ci}.wav'
        with wave.open(str(wav), 'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(a.tobytes())
        vid = out / f'clip{ci}.mp4'; TW = W * len(qs)
        enc = subprocess.Popen(['/usr/bin/ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{TW}x{H + 28}',
                                '-r', '25', '-i', 'pipe:0', '-i', str(wav), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                                '-c:a', 'aac', '-shortest', str(vid)], stdin=subprocess.PIPE)
        for t in range(0, n, 2):                                     # 50 Hz motion -> 25 fps video
            canvas = Image.new('RGB', (TW, H + 28), 'white'); dr = ImageDraw.Draw(canvas)
            for k, q in enumerate(qs):
                canvas.paste(Image.fromarray(frame(q[t])), (k * W, 28)); dr.text((k * W + 8, 5), labels[k], fill='black', font=FONT)
            if t == 0: canvas.save(out / f'clip{ci}.jpg')
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close(); enc.wait(); wav.unlink()
        index.append(dict(clip=c['clip'], ci=ci, video=vid.name, poster=f'clip{ci}.jpg', seconds=round(n / 50, 1), text=c['text'][:200]))
        print('rendered', vid, flush=True)
    (out / 'INDEX.json').write_text(json.dumps(dict(horizon=hz, columns=['真人原动作'] + [l for l, _ in cols], clips=index), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
