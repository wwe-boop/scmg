"""Copy of codex/render-isaac/render_ex002_isaac.py (11-08 qualitative set) with ONE addition: job["draw"] = JSON of per-robot,
per-frame primitives in the kinematics frame (segments = rays, points = target markers, txt = finger counts), projected into
every fixed camera view and drawn in the overlay; the kinematics->Isaac offset is measured per robot at frame 0 from the wrist links.

EX002 (PVT1 + RM003 v4 three-finger) Isaac Sim 5.1 RTX renderer for 37-dim 50 Hz joint trajectories.

Kinematic replay only (write_joint_state_to_sim + forward, no physics stepping); every video frame is one
trajectory sample, rendered off-screen with a fixed time step. 50 fps native (or joint-space cubic spline to 60).
Multiple views per robot (front, q34, hands) are stitched horizontally; several trajectories in one job are
placed side by side (one robot per env, envs 30 m apart) for comparisons.

Encoding: H.264 High, CRF 18, yuv420p, GOP = fps, +faststart, original audio (AAC 160k, padded to video
length); plus a low-bitrate preview (360p, CRF 30, same fps).

Single clip:
  CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES ~/envs/isaacsim51/bin/python render_ex002_isaac.py \
      --traj motion.npz --audio a.wav --out out.mp4 [--views front,q34,hands] [--fps 60] --headless --enable_cameras
Batch (one Isaac boot, many clips; all jobs must have the same number of trajectories):
  ... render_ex002_isaac.py --jobs jobs.json --headless --enable_cameras
  jobs.json: [{"trajs":[{"path":..., "kind":"auto", "label":"..."}], "audio":..., "out":..., "copies":[...],
               "preview":..., "title":..., "views":["front","q34"]}, ...]
"""
import argparse
import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, '/home/xr-25b7/projects/gesture/codex/render-isaac')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--jobs')
p.add_argument('--traj', action='append')
p.add_argument('--label', action='append')
p.add_argument('--audio')
p.add_argument('--out')
p.add_argument('--title', default='')
p.add_argument('--views', default='front,q34')
p.add_argument('--fps', type=int, default=50, choices=[50, 60])
p.add_argument('--height', type=int, default=720)
p.add_argument('--view_width', type=int, default=960)
p.add_argument('--aa', default='DLAA', choices=['DLAA', 'TAA', 'FXAA', 'Off'])
p.add_argument('--render_passes', type=int, default=2)
p.add_argument('--no_preview', action='store_true')
p.add_argument('--usd', default='/home/xr-25b7/isaac/ex002_lab/asset-r1/ex002_lab.usd')
p.add_argument('--log', help='per-job json timing log directory (default: next to out)')
p.add_argument('--lagtest', action='store_true', help='also record pass-to-pass image deltas (diagnostic)')
p.add_argument('--max_frames', type=int, default=0)
p.add_argument('--crf', type=int, default=18)
p.add_argument('--x264_params', default='', help='extra x264 params, e.g. ipratio=1.1:pbratio=1.1')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
t_boot = time.perf_counter()
app = AppLauncher(a).app
BOOT_S = time.perf_counter() - t_boot

FF = '/home/xr-25b7/envs/isaacsim51/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FONT = '/home/xr-25b7/isaac/assets/wqy-microhei.ttc'
VIEW_LABEL = {'front': '正面', 'q34': '3/4 侧', 'hands': '手部特写', 'rl': '交互 3/4'}


def gpu_mem_mib(idx=0):
    try:
        s = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                    text=True, timeout=10)
        return {int(l.split(',')[0]): int(l.split(',')[1]) for l in s.strip().splitlines()}.get(idx)
    except Exception:
        return None


def look_from(target, dist, az_deg, el_deg):
    import numpy as np
    az, el = np.radians(az_deg), np.radians(el_deg)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(target) + dist * d


X264 = ''


def encode_cmd(out, w, h, fps, audio, nframes, crf=18, preset='slow'):
    dur = f'{nframes / fps:.6f}'
    cmd = [FF, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    if audio:
        cmd += ['-i', str(audio)]
    cmd += ['-map', '0:v']
    if audio:
        cmd += ['-map', '1:a', '-c:a', 'aac', '-b:a', '160k', '-ar', '48000', '-af', f'apad=whole_dur={dur}', '-t', dur]
    cmd += ['-c:v', 'libx264', '-profile:v', 'high', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-g', str(fps), '-keyint_min', str(fps), '-sc_threshold', '0', '-threads', '12']
    if X264:
        cmd += ['-x264-params', X264]
    cmd += ['-movflags', '+faststart', str(out)]
    return cmd


def make_preview(src, dst, fps):
    cmd = [FF, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src), '-vf', 'scale=-2:360:flags=lanczos',
           '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'medium', '-crf', '30', '-pix_fmt', 'yuv420p',
           '-g', str(fps), '-keyint_min', str(fps), '-sc_threshold', '0', '-threads', '12',
           '-c:a', 'aac', '-b:a', '64k', '-movflags', '+faststart', str(dst)]
    subprocess.run(cmd, check=True)


try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom, Sdf
    import traj_io

    rep.orchestrator.set_capture_on_play(False)
    X264 = a.x264_params

    if a.jobs:
        JOBS = json.loads(pathlib.Path(a.jobs).read_text())
    else:
        JOBS = [dict(trajs=[dict(path=t, label=(a.label or [''] * 9)[i]) for i, t in enumerate(a.traj)],
                     audio=a.audio, out=a.out, title=a.title, views=a.views.split(','))]
    NR = {len(j['trajs']) for j in JOBS}
    assert len(NR) == 1, 'all jobs in one batch must have the same number of trajectories'
    NR = NR.pop()
    ALL_VIEWS = []
    for j in JOBS:
        j.setdefault('views', a.views.split(','))
        for v in j['views']:
            if v not in ALL_VIEWS:
                ALL_VIEWS.append(v)
    H, VW = a.height, a.view_width
    SPEC = json.loads((pathlib.Path(a.usd).parent / 'asset_spec.json').read_text())

    cams = {}
    for v in ALL_VIEWS:
        w = VW if v != 'hands' else int(VW)
        cams['cam_' + v] = CameraCfg(
            prim_path='{ENV_REGEX_NS}/Cam_' + v, update_period=0.0, height=H, width=w, data_types=['rgb'],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0,
                                             horizontal_aperture=20.955, clipping_range=(0.01, 100.0)))

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path='/World/Ground', init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.05)),
                              spawn=sim_utils.CuboidCfg(size=(400., 400., 0.1), visual_material=sim_utils.PreviewSurfaceCfg(
                                  diffuse_color=(0.42, 0.45, 0.50), roughness=0.85)))
        dome = AssetBaseCfg(prim_path='/World/Dome', spawn=sim_utils.DomeLightCfg(intensity=550.0, color=(0.95, 0.97, 1.0)))
        sun = AssetBaseCfg(prim_path='/World/Sun', init_state=AssetBaseCfg.InitialStateCfg(rot=(0.8536, 0.1464, 0.3536, 0.3536)),
                           spawn=sim_utils.DistantLightCfg(intensity=3000.0, angle=2.0))
        robot = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Robot', articulation_root_prim_path=SPEC['articulation_root_relative'],
            spawn=sim_utils.UsdFileCfg(usd_path=a.usd, rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                                       articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                                           enabled_self_collisions=False, fix_root_link=True)),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={n: d['initial'] for n, d in SPEC['joints'].items()}),
            soft_joint_pos_limit_factor=1.0, actuators={})


    dt = 1.0 / a.fps
    render_cfg = sim_utils.RenderCfg(rendering_mode='quality', antialiasing_mode=a.aa, enable_shadows=True,
                                     enable_ambient_occlusion=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device='cuda:0', render_interval=1,
                                                              use_fabric=True, render=render_cfg))
    scene_cfg = SceneCfg(num_envs=NR, env_spacing=30.0, replicate_physics=False)
    for k, v in cams.items():
        setattr(scene_cfg, k, v)
    scene = InteractiveScene(scene_cfg)
    stage = omni.usd.get_context().get_stage()

    # partner hand (RL interaction): plain USD geometry, no physics, posed per frame
    PARTNER = '/World/envs/env_0/Partner'
    xf = UsdGeom.Xform.Define(stage, PARTNER)
    skin = Gf.Vec3f(0.9, 0.7, 0.6)
    PARTNER_GEOMS = {  # MuJoCo human_hand geoms (handshake_r1_render.xml): name -> (type, size, pos, quat_wxyz)
        'human_palm': ('box', (0.012, 0.042, 0.048), (0, 0, 0), (1, 0, 0, 0)),
        'human_wrist': ('capsule', (0.02, 0.03), (0, 0, -0.085), (1, 0, 0, 0)),
        'human_finger0': ('capsule', (0.008, 0.035), (0, -0.03, 0.083), (1, 0, 0, 0)),
        'human_finger1': ('capsule', (0.008, 0.035), (0, -0.01, 0.083), (1, 0, 0, 0)),
        'human_finger2': ('capsule', (0.008, 0.035), (0, 0.01, 0.083), (1, 0, 0, 0)),
        'human_finger3': ('capsule', (0.008, 0.035), (0, 0.03, 0.083), (1, 0, 0, 0)),
        'human_thumb': ('capsule', (0.01, 0.025), (-0.02, -0.05, 0.01), (0.825336, 0, 0.564642, 0)),
        'human_fist': ('box', (0.03, 0.04, 0.045), (0, 0, 0), (1, 0, 0, 0)),
    }
    partner_prims = {}
    for n, (typ, size, pos, quat) in PARTNER_GEOMS.items():
        path = f'{PARTNER}/{n}'
        if typ == 'box':
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
            sc = Gf.Vec3f(*[2 * s for s in size])
        else:
            g = UsdGeom.Capsule.Define(stage, path)
            g.CreateRadiusAttr(size[0]); g.CreateHeightAttr(2 * size[1]); g.CreateAxisAttr('Z')
            sc = Gf.Vec3f(1, 1, 1)
        g.CreateDisplayColorAttr([skin])
        api = UsdGeom.XformCommonAPI(g)
        api.SetTranslate(Gf.Vec3d(*pos))
        w, x, y, z = quat
        api.SetRotate(Gf.Vec3f(*Gf.Rotation(Gf.Quatd(w, x, y, z)).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))),
                      UsdGeom.XformCommonAPI.RotationOrderXYZ)
        api.SetScale(sc)
        partner_prims[n] = g
    skin_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.64, 0.52), roughness=0.6)
    skin_cfg.func('/World/Looks/PartnerSkin', skin_cfg)
    for n in partner_prims:
        sim_utils.bind_visual_material(f'{PARTNER}/{n}', '/World/Looks/PartnerSkin')
    xf_t = xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    xf_o = xf.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    xf_t.Set(Gf.Vec3d(0, 0, -50))
    xf_o.Set(Gf.Quatd(1, 0, 0, 0))

    def set_partner_visible(on, skill='handshake'):
        im = UsdGeom.Imageable(xf.GetPrim())
        (im.MakeVisible if on else im.MakeInvisible)()
        fist = skill == 'fistbump'
        for n, g in partner_prims.items():
            vis = (n == 'human_fist') if fist else (n != 'human_fist')
            (UsdGeom.Imageable(g).MakeVisible if vis else UsdGeom.Imageable(g).MakeInvisible)()

    t0 = time.perf_counter()
    sim.reset()
    RESET_S = time.perf_counter() - t0
    robot = scene['robot']
    names = list(robot.joint_names)
    assert len(names) == 46, len(names)
    LO = np.array([SPEC['joints'][n]['position_bounds'][0] for n in names])
    HI = np.array([SPEC['joints'][n]['position_bounds'][1] for n in names])
    # open position limits for replay (lift/wheels are locked in the asset; we write exact values anyway)
    lim = torch.tensor(np.stack([np.minimum(LO, -10.0), np.maximum(HI, 10.0)], -1), dtype=torch.float32, device=sim.device)
    robot.write_joint_position_limit_to_sim(lim.unsqueeze(0).repeat(NR, 1, 1))
    idx37 = [traj_io.JOINT_ORDER.index(n) if n in traj_io.JOINT_ORDER else -1 for n in names]
    i_lift_drv = names.index('lift_driven_joint')
    origins = scene.env_origins.cpu().numpy()
    hand_body = [robot.body_names.index(b) for b in ('left_hand_base_link', 'right_hand_base_link')]
    palm_r = robot.body_names.index('right_palm_touch_link') if 'right_palm_touch_link' in robot.body_names else hand_body[1]
    zero = torch.zeros((NR, 46), device=sim.device)
    EXT_BODIES = [robot.body_names.index(b) for b in ('left_hand_base_link', 'right_hand_base_link', 'left_2_3_tip_link', 'right_2_3_tip_link', 'left_1_4_tip_link', 'right_1_4_tip_link', 'left_3_2_tip_link', 'right_3_2_tip_link') if b in robot.body_names]
    head_i = robot.body_names.index('head_pitch_link')
    FK_BODIES = [b for b in ('left_hand_base_link', 'right_hand_base_link', 'right_palm_touch_link', 'right_2_3_tip_link', 'left_arm_link4', 'head_pitch_link') if b in robot.body_names]
    font = ImageFont.truetype(FONT, 22)
    font_s = ImageFont.truetype(FONT, 18)

    def to46(q37):
        q = np.zeros((len(q37), 46))
        for j, i in enumerate(idx37):
            if i >= 0:
                q[:, j] = q37[:, i]
        q[:, i_lift_drv] = 0.5 * q37[:, 0]
        return q

    def set_state(Qs, k):
        q = torch.tensor(np.stack([Q[min(k, len(Q) - 1)] for Q in Qs]), dtype=torch.float32, device=sim.device)
        robot.write_joint_state_to_sim(q, zero)
        sim.forward()
        scene.update(dt)

    def render_all(views):
        for _ in range(a.render_passes):
            sim.render()
        out = {}
        for v in views:
            c = scene['cam_' + v]
            c.update(dt, force_recompute=True)
            out[v] = c.data.output['rgb'][..., :3]
        return out

    TAN_H = 20.955 / 2 / 24.0

    def fit_dist(pts, tgt, az, el, base, margin=0.86):
        """Smallest distance >= base so every point (clip-wide FK samples) stays inside the frame."""
        azr, elr = np.radians(az), np.radians(el)
        d = np.array([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)])
        right = np.cross(d, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, d)
        p = np.asarray(pts) - tgt
        th, tv = TAN_H * margin, TAN_H * H / VW * margin
        need = p @ d + np.maximum(np.abs(p @ right) / th, np.abs(p @ up) / tv)
        return float(max(base, need.max()))

    def place_cameras(views, Qs, partner):
        """Fixed camera per robot and clip. Distance is fitted to the clip-wide FK extent of hands/fingertips/head
        (never moves during the clip); hands/rl cameras aim at the clip hand position."""
        hand_pts = [[] for _ in range(NR)]
        ext_pts = [[] for _ in range(NR)]
        T = max(len(Q) for Q in Qs)
        for k in range(0, T, 2):
            set_state(Qs, k)
            bp = robot.data.body_pos_w.cpu().numpy()
            for r in range(NR):
                o = origins[r]
                hand_pts[r].append(bp[r, hand_body].mean(0) - o if not partner else bp[r, palm_r] - o)
                ext_pts[r] += [bp[r, i] - o for i in EXT_BODIES]
                ext_pts[r].append(bp[r, head_i] - o + np.array([0, 0, 0.16]))
        info = {}
        for v in views:
            tgts, specs = [], []
            for r in range(NR):
                if v == 'front':
                    tgt, az, el, base, pts = np.array([0.0, 0.0, 1.08]), 0.0, 5.0, 2.45, ext_pts[r]
                elif v == 'q34':
                    tgt, az, el, base, pts = np.array([0.05, 0.0, 1.08]), -40.0, 8.0, 2.45, ext_pts[r]
                elif v == 'hands':
                    tgt = np.median(np.array(hand_pts[r]), 0)
                    az, el, base = 15.0, 20.0, 0.9
                    pts = [x for i, x in enumerate(ext_pts[r]) if i % (len(EXT_BODIES) + 1) != len(EXT_BODIES)]
                elif v == 'rl':
                    hp = np.array(hand_pts[r])
                    tgt = hp[int(0.3 * len(hp)):int(0.7 * len(hp))].mean(0)
                    az, el, base, pts = 35.0, 15.0, 1.6, []  # = MuJoCo view (1.1, az 215, el -15)
                dist = fit_dist(pts, tgt, az, el, base) if len(pts) else base
                if v == 'hands':
                    dist = min(dist, 1.5)  # keep it a close-up
                tgts.append(tgt); specs.append((az, el, dist))
            if v in ('front', 'q34'):  # same scale for all robots in a comparison
                dmax = max(s_[2] for s_ in specs)
                specs = [(az_, el_, dmax) for az_, el_, _ in specs]
            eyes = [look_from(t, sp[2], sp[0], sp[1]) + origins[r] for r, (t, sp) in enumerate(zip(tgts, specs))]
            tw = [t + origins[r] for r, t in enumerate(tgts)]
            scene['cam_' + v].set_world_poses_from_view(torch.tensor(np.array(eyes), dtype=torch.float32, device=sim.device),
                                                        torch.tensor(np.array(tw), dtype=torch.float32, device=sim.device))
            info[v] = dict(eye=[list(map(float, e - origins[r])) for r, e in enumerate(eyes)],
                           target=[list(map(float, t)) for t in tgts], distance=[sp[2] for sp in specs])
        return info

    def overlay(img, job, k, fps, labels, views, ncol_per_robot):
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, 'RGBA')
        W = im.width
        title = job.get('title') or ''
        if title:
            dr.rectangle([0, 0, W, 34], fill=(16, 20, 28, 170))
            dr.text((12, 5), title[:110], font=font, fill=(255, 255, 255, 255))
        x = 0
        for r in range(NR):
            for v in views:
                w = VW
                lab = VIEW_LABEL.get(v, v) + (f' · {labels[r]}' if labels[r] else '')
                tw = dr.textlength(lab, font=font_s)
                dr.rectangle([x + 8, H - 32, x + 20 + tw, H - 8], fill=(16, 20, 28, 150))
                dr.text((x + 14, H - 31), lab, font=font_s, fill=(255, 255, 255, 255))
                x += w
        ts = f'{k / fps:5.2f}s  #{k:04d}'
        tw = dr.textlength(ts, font=font_s)
        dr.rectangle([W - tw - 22, H - 32, W - 8, H - 8], fill=(16, 20, 28, 150))
        dr.text((W - tw - 14, H - 31), ts, font=font_s, fill=(255, 255, 255, 255))
        D = job.get('_draw')
        if D is not None:
            fd = [rb['frames'][min(k, len(rb['frames']) - 1)] for rb in D['robots']]
            for r in range(NR):
                if r >= len(fd): break
                for vi, v in enumerate(views):
                    ci = job['_cam'][v]; eye = np.asarray(ci['eye'][r]); tg = np.asarray(ci['target'][r])
                    dv = (tg - eye) / np.linalg.norm(tg - eye); rt = np.cross(dv, [0, 0, 1.0]); rt /= np.linalg.norm(rt); up = np.cross(rt, dv)
                    x0 = (r * len(views) + vi) * VW; off = job['_off'][r]
                    def pj(pt):
                        rel = np.asarray(pt) + off - eye; zc = rel @ dv
                        if zc < 0.05: return None
                        return (x0 + VW / 2 + (rel @ rt) / zc / TAN_H * VW / 2, H / 2 - (rel @ up) / zc / TAN_H * VW / 2)
                    def clip(pa, pb, xmin, xmax, ymin=0., ymax=float(H)):      # Liang-Barsky: keep the segment inside this view tile
                        dx, dy = pb[0] - pa[0], pb[1] - pa[1]; t0, t1 = 0., 1.
                        for pp, qq in ((-dx, pa[0] - xmin), (dx, xmax - pa[0]), (-dy, pa[1] - ymin), (dy, ymax - pa[1])):
                            if pp == 0:
                                if qq < 0: return None
                            else:
                                tt = qq / pp
                                if pp < 0: t0 = max(t0, tt)
                                else: t1 = min(t1, tt)
                        if t0 > t1: return None
                        return (pa[0] + t0 * dx, pa[1] + t0 * dy), (pa[0] + t1 * dx, pa[1] + t1 * dy)
                    for a_, b_, col in fd[r]['seg']:
                        pa, pb = pj(a_), pj(b_)
                        cl = clip(pa, pb, x0, x0 + VW - 1) if (pa and pb) else None
                        if cl: dr.line(list(cl), fill=tuple(col) + (230,), width=4)
                    for pt, col, rad in fd[r]['pts']:
                        pc = pj(pt)
                        if pc and x0 <= pc[0] < x0 + VW: dr.ellipse([pc[0] - rad, pc[1] - rad, pc[0] + rad, pc[1] + rad], fill=tuple(col) + (220,), outline=(0, 0, 0, 255), width=2)
                    if vi == 0:
                        for li, tx in enumerate(fd[r]['txt']):
                            yy = 44 + 30 * li; tw_ = dr.textlength(tx, font=font_s)
                            dr.rectangle([x0 + 8, yy, x0 + 20 + tw_, yy + 26], fill=(16, 20, 28, 170)); dr.text((x0 + 14, yy + 2), tx, font=font_s, fill=(255, 235, 120, 255))
        if job.get('_status'):
            s = job['_status'][min(k, len(job['_status']) - 1)]
            dr.rectangle([0, 34 if title else 0, W, (34 if title else 0) + 28], fill=(0, 0, 0, 140))
            dr.text((12, (34 if title else 0) + 3), s, font=font_s, fill=(255, 235, 160, 255))
        return np.asarray(im)

    summary = []
    # renderer warm-up (shader compile, DLAA history)
    tw0 = time.perf_counter()
    for _ in range(30):
        sim.render()
    WARM_S = time.perf_counter() - tw0
    mem_after_warm = gpu_mem_mib(0)
    print(f'BOOT {BOOT_S:.1f}s RESET {RESET_S:.1f}s WARM {WARM_S:.1f}s MEM {mem_after_warm} MiB', flush=True)

    for ji, job in enumerate(JOBS):
        out = pathlib.Path(job['out']).expanduser()
        if out.exists() and not job.get('overwrite'):
            print('SKIP exists', out, flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tj = time.perf_counter()
        views = job['views']
        loaded = [traj_io.load(t['path'], t.get('kind', 'auto')) for t in job['trajs']]
        partner = loaded[0].get('partner')
        Qs = []
        for L in loaded:
            q37 = traj_io.resample(L['q37'], L['fps'], a.fps)
            Qs.append(to46(q37))
        if partner is not None:
            partner = traj_io.resample_partner(partner, loaded[0]['fps'], a.fps, len(Qs[0]))
            meta = loaded[0]['meta']
            tt = np.arange(len(Qs[0])) / a.fps
            src_i = np.minimum(np.round(tt * loaded[0]['fps']).astype(int), len(meta['state']) - 1)
            job['_status'] = [f"{partner['skill']}  机器人={meta['state'][i]}  对方={meta['phase'][i]}  指力={meta['finger_f'][i]:4.1f} N"
                              for i in src_i]
        set_partner_visible(partner is not None, partner['skill'] if partner else '')
        T = max(len(Q) for Q in Qs)
        if a.max_frames:
            T = min(T, a.max_frames)
        cam_info = place_cameras(views, Qs, partner is not None)
        job['_cam'] = cam_info; job['_draw'] = None
        if job.get('draw'):
            job['_draw'] = json.loads(pathlib.Path(job['draw']).read_text())
            set_state(Qs, 0); bp = robot.data.body_pos_w.cpu().numpy()
            wl = [robot.body_names.index(b) for b in ('left_arm_link7', 'right_arm_link7')]
            job['_off'] = []
            for r in range(NR):
                iso = bp[r, wl] - origins[r]; ref = np.asarray(job['_draw']['robots'][min(r, len(job['_draw']['robots']) - 1)]['ref_wrist'])
                job['_off'].append((iso - ref).mean(0)); print('DRAW offset robot', r, np.round(iso - ref, 4).tolist(), flush=True)
        labels = [t.get('label', '') for t in job['trajs']]
        Wtot = VW * len(views) * NR
        tmp = out.with_suffix('.tmp.mp4')
        enc = subprocess.Popen(encode_cmd(tmp, Wtot, H, a.fps, job.get('audio'), T, crf=a.crf), stdin=subprocess.PIPE)
        q_frames = queue.Queue(maxsize=16)
        enc_times = []

        werr = []

        def writer():
            try:
                while True:
                    item = q_frames.get()
                    if item is None:
                        break
                    te = time.perf_counter()
                    img = overlay(item[1], job, item[0], a.fps, labels, views, len(views))
                    enc.stdin.write(img.tobytes())
                    enc_times.append(time.perf_counter() - te)
            except Exception as e:  # surface encoder failure to the render loop
                werr.append(repr(e))
        th = threading.Thread(target=writer, daemon=True)
        th.start()
        # pre-roll at frame 0 so temporal AA history starts on the first pose
        set_state(Qs, 0)
        for _ in range(8):
            sim.render()
        per_ms, mem, fk_probe = [], [], []
        lag = []
        for k in range(T):
            torch.cuda.synchronize()
            t = time.perf_counter()
            set_state(Qs, k)
            if partner is not None:
                pp = partner['pos'][min(k, len(partner['pos']) - 1)] + origins[0]
                pq = partner['quat_wxyz'][min(k, len(partner['pos']) - 1)]
                xf_t.Set(Gf.Vec3d(*map(float, pp)))
                xf_o.Set(Gf.Quatd(*map(float, pq)))
            imgs = render_all(views)
            row = torch.cat([imgs[v][r] for r in range(NR) for v in views], dim=1)
            frame = row.cpu().numpy().astype(np.uint8)
            torch.cuda.synchronize()
            per_ms.append((time.perf_counter() - t) * 1e3)
            if a.lagtest and k % 25 == 0:
                extra = render_all(views)
                row2 = torch.cat([extra[v][r] for r in range(NR) for v in views], dim=1).cpu().numpy().astype(np.int16)
                lag.append(float(np.abs(row2 - frame.astype(np.int16)).mean()))
            while True:
                if werr:
                    enc.kill()
                    raise RuntimeError('encoder failed: ' + werr[0])
                try:
                    q_frames.put((k, frame), timeout=1.0)
                    break
                except queue.Full:
                    pass
            if k % 50 == 0:
                mem.append(gpu_mem_mib(0))
            if k % 25 == 0:
                bp = robot.data.body_pos_w[0].cpu().numpy() - origins[0]
                fk_probe.append([k] + [list(map(float, bp[robot.body_names.index(b)])) for b in FK_BODIES])
        q_frames.put(None)
        th.join()
        enc.stdin.close()
        rc = enc.wait()
        assert rc == 0, f'ffmpeg exit {rc}'
        tmp.rename(out)
        prev = None
        if not a.no_preview:
            prev = pathlib.Path(job.get('preview') or out.parent / 'preview' / out.name)
            prev.parent.mkdir(parents=True, exist_ok=True)
            make_preview(out, prev, a.fps)
        for c in job.get('copies', []):
            c = pathlib.Path(c)
            c.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, c)
            if prev is not None:
                (c.parent / 'preview').mkdir(exist_ok=True)
                shutil.copy2(prev, c.parent / 'preview' / out.name)
        pm = np.array(per_ms)
        rec = dict(out=str(out), preview=str(prev) if prev else None, copies=job.get('copies', []),
                   trajs=[t['path'] for t in job['trajs']], audio=job.get('audio'), frames=T, fps=a.fps,
                   src_frames=[len(L['q37']) for L in loaded], src_fps=[L['fps'] for L in loaded],
                   missing_joints=[L['meta'].get('missing') for L in loaded],
                   size=[Wtot, H], views=views, aa=a.aa, render_passes=a.render_passes,
                   render_ms=dict(mean=float(pm.mean()), p50=float(np.percentile(pm, 50)), p95=float(np.percentile(pm, 95)),
                                  max=float(pm.max())),
                   overlay_encode_write_ms_mean=float(np.mean(enc_times) * 1e3) if enc_times else None,
                   wall_s=time.perf_counter() - tj, gpu0_mem_mib=dict(samples=mem, max=max([m for m in mem if m] or [0])),
                   per_frame_ms=[round(x, 2) for x in per_ms], lagtest_mean_abs=lag or None, cameras=cam_info,
                   partner=partner['skill'] if partner else None,
                   fk_probe=dict(bodies=FK_BODIES, frames=fk_probe))
        logd = pathlib.Path(a.log) if a.log else out.parent / 'logs'
        logd.mkdir(parents=True, exist_ok=True)
        (logd / (out.stem + '.json')).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        summary.append({k: v for k, v in rec.items() if k not in ('per_frame_ms',)})
        print(f'DONE {ji + 1}/{len(JOBS)} {out.name} frames={T} render_ms={pm.mean():.1f}/{np.percentile(pm, 95):.1f} '
              f'wall={rec["wall_s"]:.1f}s mem={rec["gpu0_mem_mib"]["max"]}', flush=True)
    print('ALL_DONE', json.dumps(dict(boot_s=BOOT_S, reset_s=RESET_S, warm_s=WARM_S, jobs=len(summary))), flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    raise
finally:
    if 'sim' in globals():
        sim.clear_all_callbacks()
        sim.clear_instance()
    app.close(wait_for_replicator=False)
