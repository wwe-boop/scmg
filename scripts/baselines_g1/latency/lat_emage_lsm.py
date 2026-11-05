"""EMAGE (haiyangliu0/emage_audio) and GestureLSM (shortcut_reflow.bin, configs_new/shortcut_rvqvae_128.yaml) latency with
the exact benchmark inference code of gmr-repro/scripts/{emage_infer,lsm_infer}.py (same model calls, seeds, padding),
batch 1, cuda-synchronised stage timers inserted.  Both are whole-utterance pipelines (EMAGE encodes the whole clip in
one pass; GestureLSM's windows read pre-computed whole-clip onset/amplitude features), compute starts once the clip has
arrived.  Run with gmr-repro/env/gmr on zehan-server (weights under /workspace/zehan/gmr-local).
  EMAGE stages: audio_feat = wav read / resample; generate = EmageAudioModel.inference (its audio encoder is inside and
                cannot be separated without editing the model); decode = VQ decode + global motion.
  LSM stages:   audio_feat = onset + amplitude; text_enc = TextGrid word ids; generate = shortcut-flow window loop;
                decode = RVQ latent2origin + 6D -> axis-angle."""
import argparse, json, os, sys, zlib
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402

R = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/gmr-repro')
DR = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root')
WORDS = R / 'data/echog/annotations/words'


def run_emage(stems, clk, dev):
    sys.argv = ['x']; sys.path.insert(0, str(R / 'scripts'))
    import emage_infer as E
    import torch.nn.functional as F
    W = R / 'weights/emage_audio'
    face = E.load_pretrained(E.EmageVQVAEConv, W / 'emage_vq/face').to(dev).eval()
    upper = E.load_pretrained(E.EmageVQVAEConv, W / 'emage_vq/upper').to(dev).eval()
    lower = E.load_pretrained(E.EmageVQVAEConv, W / 'emage_vq/lower').to(dev).eval()
    hands = E.load_pretrained(E.EmageVQVAEConv, W / 'emage_vq/hands').to(dev).eval()
    glob = E.load_pretrained(E.EmageVAEConv, W / 'emage_vq/global').to(dev).eval()
    vq = E.EmageVQModel(face_model=face, upper_model=upper, lower_model=lower, hands_model=hands, global_model=glob).to(dev).eval()
    model = E.load_pretrained(E.EmageAudioModel, W).to(dev).eval()
    rows = []
    for k, stem in enumerate(stems):
        st = {}
        n = int(torch.load(DR / 'motion_39d_30fps' / f'{stem}.pt', map_location='cpu', weights_only=True)['real_num_frames'])
        with torch.no_grad():
            t0 = clk.now(); wav = E.read_wav(DR / 'audio' / f'{stem}.wav'); dur = len(wav) / 16000.
            audio = torch.from_numpy(wav).to(dev).unsqueeze(0); st['audio_feat'] = clk.now() - t0
            t0 = clk.now()
            latent = model.inference(audio, torch.zeros(1, 1).long().to(dev), vq, masked_motion=None, mask=None)
            cfg = model.cfg
            fl = latent['rec_face'] if cfg.lf > 0 and cfg.cf == 0 else None
            ul = latent['rec_upper'] if cfg.lu > 0 and cfg.cu == 0 else None
            hl = latent['rec_hands'] if cfg.lh > 0 and cfg.ch == 0 else None
            ll = latent['rec_lower'] if cfg.ll > 0 and cfg.cl == 0 else None
            pick = lambda key, c: torch.max(F.log_softmax(latent[key], dim=2), dim=2)[1] if c > 0 else None
            fi, ui, hi, li = pick('cls_face', cfg.cf), pick('cls_upper', cfg.cu), pick('cls_hands', cfg.ch), pick('cls_lower', cfg.cl)
            st['generate'] = clk.now() - t0
            t0 = clk.now()
            out = vq.decode(face_latent=fl, upper_latent=ul, lower_latent=ll, hands_latent=hl, face_index=fi, upper_index=ui,
                            lower_index=li, hands_index=hi, get_global_motion=True, ref_trans=torch.zeros(1, 3, device=dev))
            out['motion_axis_angle'][0].float().cpu().numpy(); out['trans'][0].float().cpu().numpy()
            st['decode'] = clk.now() - t0
        comp = sum(st.values())
        rows.append(dict(stem=stem, dur=dur, frames30=n, stages=st, compute=comp, schedule=[(dur, comp, 0.0, n, 30.)]))
        print(f'{k + 1}/{len(stems)} {stem} dur={dur:.2f} comp={comp:.3f} {st}', flush=True)
    return rows


def run_lsm(stems, clk, dev):
    sys.path.insert(0, str(R / 'scripts'))
    import lsm_infer as Lm   # sets sys.path for GestureLSM, stubs fasttext
    from omegaconf import OmegaConf
    from utils import config
    from utils.joints import upper_body_mask, hands_body_mask, lower_body_mask
    from utils import rotation_conversions as rc
    REPO = Lm.REPO
    cfg_path = str(REPO / 'configs_new/shortcut_rvqvae_128.yaml'); ckpt = str(R / 'weights/gesturelsm/shortcut_reflow.bin')
    cwd = os.getcwd(); os.chdir(REPO)
    cfg = OmegaConf.load(cfg_path); cfg.model.modality_encoder.params.data_path = str(REPO) + '/'; cfg.exp_name = Path(cfg_path).stem
    flat = Path(f'/tmp/_flat_lsm_{os.getpid()}.yaml'); flat.write_text(f'cfg: {cfg_path}\n')
    sys.argv = ['x']; args, _ = config.parse_args(str(flat))
    for kk, v in cfg.items():
        if kk in ('model', 'data', 'solver', 'validation'): continue
        if not hasattr(v, 'items'): setattr(args, kk, v)
    for kk, v in cfg.data.items(): setattr(args, kk, v)
    args.vqvae_upper_path = str(R / 'weights/gesturelsm/net_300000_upper.pth'); args.vqvae_hands_path = str(R / 'weights/gesturelsm/net_300000_hands.pth')
    args.vqvae_lower_path = str(R / 'weights/gesturelsm/net_300000_lower.pth')
    args.mean_pose_path = str(REPO / 'mean_std/beatx_2_330_mean.npy'); args.std_pose_path = str(REPO / 'mean_std/beatx_2_330_std.npy')
    args.mean_trans_path = str(REPO / 'mean_std/beatx_2_trans_mean.npy'); args.std_trans_path = str(REPO / 'mean_std/beatx_2_trans_std.npy')
    args.use_trans = True
    from models.vq.model import RVQVAE
    mm = __import__(f'models.{cfg.model.model_name}', fromlist=['x'])
    def make_vq(path, dim):
        m = RVQVAE(args, dim, args.nb_code, args.code_dim, args.code_dim, args.down_t, args.stride_t, args.width, args.depth,
                   args.dilation_growth_rate, args.vq_act, args.vq_norm)
        m.load_state_dict(torch.load(path, map_location='cpu')['net']); return m.to(dev).eval()
    vq_u, vq_h, vq_l = make_vq(args.vqvae_upper_path, 78), make_vq(args.vqvae_hands_path, 180), make_vq(args.vqvae_lower_path, 57)
    model = getattr(mm, cfg.model.g_name)(cfg).to(dev).eval(); Lm.load_state(model, ckpt)
    mean = np.load(args.mean_pose_path); std = np.load(args.std_pose_path)
    mean = {p: torch.from_numpy(mean[m]).float().to(dev) for p, m in (('upper', upper_body_mask), ('hands', hands_body_mask), ('lower', lower_body_mask))}
    std = {p: torch.from_numpy(std[m]).float().to(dev) for p, m in (('upper', upper_body_mask), ('hands', hands_body_mask), ('lower', lower_body_mask))}
    tm = torch.from_numpy(np.load(args.mean_trans_path)).float().to(dev); ts = torch.from_numpy(np.load(args.std_trans_path)).float().to(dev)
    vocab = model.modality_encoder.lang_model
    m3u, m3h, m3l = Lm.mask6to3(upper_body_mask), Lm.mask6to3(hands_body_mask), Lm.mask6to3(lower_body_mask)
    scale, sq, pre, plen = args.vqvae_latent_scale, args.vqvae_squeeze_scale, args.pre_frames, args.pose_length
    round_l = plen - pre * sq
    rows = []
    for k, stem in enumerate(stems):
        st = {}
        gt = torch.load(DR / 'motion_39d_30fps' / f'{stem}.pt', map_location='cpu', weights_only=True); n = int(gt['real_num_frames'])
        kk_ = max(1, int(np.ceil((n - pre * sq) / round_l))); N = pre * sq + kk_ * round_l
        with torch.no_grad():
            t0 = clk.now()
            feat = Lm.onset_amplitude(DR / 'audio' / f'{stem}.wav'); dur = len(feat) / 16000.
            need = int(np.ceil(N / 30.0 * 16000))
            if len(feat) < need: feat = np.pad(feat, ((0, need - len(feat)), (0, 0)))
            audio = torch.from_numpy(feat[:need].astype(np.float32)).unsqueeze(0).to(dev); st['audio_feat'] = clk.now() - t0
            t0 = clk.now()
            ann = json.load(open(WORDS / f'{stem}.json')); tg = Lm.build_textgrid(ann['words'], N)
            word = torch.from_numpy(Lm.word_ids(tg, vocab, N)).long().unsqueeze(0).to(dev); st['text_enc'] = clk.now() - t0
            t0 = clk.now()
            seed = torch.zeros(1, N // sq, 3 * 128, device=dev)
            torch.manual_seed((0 * 1000003 + zlib.crc32(stem.encode())) & 0x7fffffff)
            ru, rh, rl = [], [], []; last = None
            for i in range(kk_):
                s0 = i * round_l
                at = audio[:, s0 * (16000 // 30):(s0 + round_l) * (16000 // 30) + (16000 // 30) * pre * sq]
                wt = word[:, s0:s0 + round_l + pre * sq]
                if at.shape[1] < (16000 // 30) * plen: at = torch.nn.functional.pad(at, (0, (16000 // 30) * plen - at.shape[1]))
                if wt.shape[1] < plen: wt = torch.nn.functional.pad(wt, (0, plen - wt.shape[1]), value=vocab.UNK_token)
                sd = seed[:, :pre, :] if i == 0 else last[:, -pre:, :]
                c = {'y': {'audio_onset': at, 'word': wt, 'id': torch.zeros(1, plen, 1, device=dev), 'seed': sd,
                           'style_feature': torch.zeros(1, 512, device=dev)}}
                smp = model(c)['latents'].squeeze().permute(1, 0).unsqueeze(0); last = smp.clone()
                o = 0 if i == 0 else pre
                ru.append(smp[:, o:, :128]); rh.append(smp[:, o:, 128:256]); rl.append(smp[:, o:, 256:])
            st['generate'] = clk.now() - t0
            t0 = clk.now()
            up = vq_u.latent2origin(torch.cat(ru, 1) * scale)[0][0]; hd = vq_h.latent2origin(torch.cat(rh, 1) * scale)[0][0]
            lo = vq_l.latent2origin(torch.cat(rl, 1) * scale)[0][0]
            tv = lo[..., -3:] * ts + tm; lo = lo[..., :-3]; tr = torch.cumsum(tv, dim=0); tr[..., 1] = tv[..., 1]
            up = up * std['upper'] + mean['upper']; hd = hd * std['hands'] + mean['hands']; lo = lo * std['lower'] + mean['lower']
            nl = up.shape[0]
            up = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(up.reshape(1, nl, 13, 6))).reshape(nl, 39).cpu().numpy()
            lo = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(lo.reshape(1, nl, 9, 6))).reshape(nl, 27).cpu().numpy()
            hd = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(hd.reshape(1, nl, 30, 6))).reshape(nl, 90).cpu().numpy()
            rec = np.zeros((nl, 165), np.float32); rec[:, m3u] = up; rec[:, m3l] = lo; rec[:, m3h] = hd; tr.cpu().numpy()
            st['decode'] = clk.now() - t0
        comp = sum(st.values())
        rows.append(dict(stem=stem, dur=dur, frames30=n, windows=kk_, stages=st, compute=comp, schedule=[(dur, comp, 0.0, n, 30.)]))
        print(f'{k + 1}/{len(stems)} {stem} dur={dur:.2f} comp={comp:.3f} {st}', flush=True)
    os.chdir(cwd)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--method', required=True, choices=['emage', 'lsm'])
    ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    mon = LC.GPUMon(); host0 = LC.host_snapshot(); clk = LC.Sync(); dev = 'cuda'
    stems = LC.stems50()[:a.n or None]
    rows = run_emage(stems, clk, dev) if a.method == 'emage' else run_lsm(stems, clk, dev)
    extra = dict(frame_fps=30, host_start=host0, host_end=LC.host_snapshot(), torch=torch.__version__,
                 weights=dict(emage='haiyangliu0/emage_audio', lsm='pliu23/GestureLSM shortcut_reflow.bin')[a.method])
    extra.update(LC.cuda_mem()); extra.update(mon.stop())
    LC.write(a.out, a.method, rows, extra)


if __name__ == '__main__':
    main()
