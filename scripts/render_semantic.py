"""Side-by-side EX002 render of SEMANTIC-eval generations (Chinese semsynth clips of manifest-n200) with the speech and a
Chinese subtitle; the semantic events (lexicon class + time) are shown under the subtitle while they are active (11-04).
    render_semantic.py <out dir> <manifest.json> <entry idx,...> <label=gen_dir> [...]
gen_dir holds gen_<ci>.npy from scripts/gen_semantic.py (34 joints rad 50 Hz). Ground truth = the view's train_q at the clip.
Runs on speech_server with EGL (physical GPU1)."""
import os
os.environ.update(__EGL_VENDOR_LIBRARY_FILENAMES='/mnt/cpfs/zbl-cpfs-new/USERS/zehan/speaker-style-emb-r2-20261030/nvidia-egl.json',
                  MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='1')
import json, subprocess, sys, wave
from collections import defaultdict
from pathlib import Path
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
MODEL = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture/ex002-rm003-v4-20261020/model-r1/scene.xml'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 16)
SUB = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 22)
W, H, TOP, BOT = 300, 340, 28, 70


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    man = json.load(open(sys.argv[2])); ks = [int(x) for x in sys.argv[3].split(',')]
    cols = [a.split('=', 1) for a in sys.argv[4:]]
    V = Path(man['view']); items = json.loads((V / 'train.json').read_text())
    keep = set(json.loads((V / 'SPLIT.json').read_text())['train_ids'])
    clips = defaultdict(list)
    for it in items:
        if it['id'] in keep: clips[it['clip']].append(it)
    start = {c: min(x['offset'] for x in xs) for c, xs in clips.items()}
    clip_of = {it['id']: it['clip'] for it in items}
    q_all = np.load(V / 'train_q.npy', mmap_mode='r')
    audio = np.load(Path(str(V) + '-audio16') / 'AUDIO16.npy', mmap_mode='r')
    joints = json.load(open(V / 'META.json'))['joints']
    m = mujoco.MjModel.from_xml_path(MODEL); d = mujoco.MjData(m)
    m.vis.global_.offwidth = W; m.vis.global_.offheight = H; ren = mujoco.Renderer(m, H, W)
    adr = [m.jnt_qposadr[m.joint(n).id] for n in joints]
    lift = m.jnt_qposadr[m.joint('lift_joint').id]; liftd = m.jnt_qposadr[m.joint('lift_driven_joint').id]
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 0; opt.geomgroup[1:3] = 1
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 1.18]; cam.distance = 1.9; cam.azimuth = 180; cam.elevation = -5

    def frame(q):
        d.qpos[:] = 0; d.qpos[adr] = q; d.qpos[lift] = 0.32; d.qpos[liftd] = 0.16
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); ren.update_scene(d, cam, opt); return ren.render()
    index = []
    for k in ks:
        e = man['entries'][k]; ci = e['ci']; s0 = start[clip_of[e['item_ids'][0]]] + e['start_frame']
        gt = np.asarray(q_all[s0:s0 + e['frames']], np.float32)
        qs = [gt] + [np.load(Path(p) / f'gen_{ci}.npy') for _, p in cols]
        n = min(len(x) for x in qs); labels = ['真值（合成数据）'] + [l for l, _ in cols]
        a = np.asarray(audio[s0 * 320:(s0 + n) * 320], np.int16); wav = out / f'sem{k}.wav'
        with wave.open(str(wav), 'wb') as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(a.tobytes())
        vid = out / f'sem{k}.mp4'; TW = W * len(qs); TH = TOP + H + BOT
        enc = subprocess.Popen(['/usr/bin/ffmpeg', '-v', 'error', '-y', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{TW}x{TH}',
                                '-r', '25', '-i', 'pipe:0', '-i', str(wav), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23',
                                '-c:a', 'aac', '-shortest', str(vid)], stdin=subprocess.PIPE)
        text = e['text']; ev = e.get('events', [])
        for t in range(0, n, 2):
            canvas = Image.new('RGB', (TW, TH), 'white'); dr = ImageDraw.Draw(canvas)
            for j, q in enumerate(qs):
                canvas.paste(Image.fromarray(frame(q[t])), (j * W, TOP)); dr.text((j * W + 8, 5), labels[j], fill='black', font=FONT)
            dr.text((12, TOP + H + 6), text, fill='black', font=SUB)
            now = t / 50; act = [x['cls_name'] for x in ev if x['t_on'] <= now <= x['t_off']]
            if act: dr.text((12, TOP + H + 38), '语义手势（词典）：' + ' / '.join(act), fill=(200, 0, 0), font=FONT)
            if t == 0: canvas.save(out / f'sem{k}.jpg')
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close(); enc.wait(); wav.unlink()
        index.append(dict(k=k, ci=ci, video=vid.name, poster=f'sem{k}.jpg', seconds=round(n / 50, 1), text=text,
                          events=[dict(cls=x['cls_name'], t_on=x['t_on'], t_off=x['t_off']) for x in ev]))
        print('rendered', vid, flush=True)
    (out / 'INDEX.json').write_text(json.dumps(dict(columns=['真值（合成数据）'] + [l for l, _ in cols], clips=index), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
