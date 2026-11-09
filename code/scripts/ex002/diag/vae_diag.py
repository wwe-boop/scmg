"""11-08 read-only VAE isolation (coordinator): is the 4-frame acceleration-spike pattern of EX002 generations made by the vae-r3 decoder?
 A  dev GT (43 clips) -> encode (mu) -> decode: spike statistics (spikes_where.stat / where), reconstruction error
 B  enc-s0 offline generation (seed 0, none, bench_ex002 infer, clips of --shard) with the committed latents captured, decoded three ways:
    streamed (= the emitted output: 4-chunk tail, 1-latent look-ahead), one-shot (all latents at once), no look-ahead (each latent's frames
    decoded with the next latent replaced by a copy of itself); plus the streamed output re-encoded and decoded (projection onto the manifold)
    python vae_diag.py OUT_JSON [shard]      (SCMG_J=34; PYTHONPATH = release dir)"""
import json, glob, os, sys, types
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spikes_where import stat, where, speech_masks, ARM_E
B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; C = '/mnt/cpfs/zbl-cpfs-new/dataset/zehan/gesture'
E = B + '/ex002-sfscope-1108/enc-s0'; VAE = B + '/trained/vae-r3/final.pt'; ST = C + '/scmg/STATS_v36_1.json'
UB = list(range(2, 34))                                     # moving upper body: head, arms, fingers (waist / pedestal excluded)


def main():
    out, shard = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '0/6')
    from scmg.vae import MotionVAE
    dev = 'cuda'; st = json.load(open(ST)); qm = torch.tensor(st['q_mean'], device=dev).float(); qs = torch.tensor(st['q_std'], device=dev).float()
    vck = torch.load(VAE, map_location='cpu'); vae = MotionVAE(**vck['cfg']).to(dev); vae.load_state_dict(vck['ema']); vae.eval()
    R = {}; sm = speech_masks()

    def enc_dec(y):
        T = len(y); Tp = -(-T // 4) * 4; x = np.concatenate([y, np.repeat(y[-1:], Tp - T, 0)]) if Tp > T else y
        with torch.no_grad():
            mu, _ = vae.encode(((torch.as_tensor(x, device=dev).float() - qm) / qs)[None]); r = vae.decode(mu)[0] * qs + qm
        return r[:T].cpu().numpy().astype(np.float64)
    gt = [np.load(f'{E}/offline/gt_{i}.npy').astype(np.float64) for i in range(len(glob.glob(E + '/offline/gt_*.npy')))]
    g, ga = stat(gt, 50, ARM_E); thr = g['p99']; R['A GT'] = g; R['A GT where'] = where(gt, ga, thr, sm)
    rec = [enc_dec(y) for y in gt]; d, da = stat(rec, 50, ARM_E, thr); R['A GT recon (vae-r3 mu)'] = d; R['A GT recon where'] = where(rec, da, thr, sm)
    R['A recon error'] = dict(mean_abs_rad_upper_body=float(np.mean([np.abs(r[:, UB] - y[:, UB]).mean() for r, y in zip(rec, gt)])),
                              mean_abs_rad_arms=float(np.mean([np.abs(r[:, 4:18] - y[:, 4:18]).mean() for r, y in zip(rec, gt)])))
    print(json.dumps({k: R[k] for k in ('A GT', 'A GT recon (vae-r3 mu)', 'A recon error')}), flush=True)
    # B: generation with latent capture
    import scmg.infer as SI
    LAT = {}; orig = SI.Streamer

    class Cap(orig):
        def flush(self):
            fr = super().flush(); LAT[len(LAT)] = torch.stack(self.lat).clone(); return fr
    SI.Streamer = Cap
    sys.path.insert(0, os.environ['REL'] + '/scripts/ex002'); import bench_ex002 as BE
    gdir = os.path.dirname(os.path.abspath(out)) + '/gen_s0'
    a = types.SimpleNamespace(ckpt=E + '/train/final.pt', vae=VAE, out=gdir, view=C + '/train-v1-20261101/views/data-v36_1', stats=ST,
                              text_cache=C + '/scmg/text-e5-v36_1', tok_cache=C + '/scmg/tok-qwen25-v36_1', text_causal=B + '/models/text-e5-causal-v36_1',
                              protocol='offline', seed=0, seed_mode='per_clip', steps=10, cfg_w=2.0, shard=shard, pstats=None, campplus=None, words_dir=None,
                              spk_refresh='grades', causal_feat='inc', plan=None, mode='none', inpaint_dir=None, ctl_lookahead=0.)
    BE.infer(a)
    from scmg.evaluate import load
    m, _ = load(a.ckpt, VAE, dev)
    cis = sorted(int(os.path.basename(f)[:-4]) for f in glob.glob(gdir + '/[0-9]*.npy'))
    streamed, oneshot, nolook, reproj, ref = [], [], [], [], []
    for k, ci in enumerate(cis):
        y = np.load(f'{gdir}/{ci}.npy').astype(np.float64); L = len(y); lat = LAT[k].to(dev)
        z = lat * m.lat_std.reshape(-1) + m.lat_mean.reshape(-1)
        with torch.no_grad():
            full = (vae.decode(z[None].float())[0] * qs + qm).cpu().numpy().astype(np.float64)[:L]
            nl = []
            for i in range(len(z)):
                i0 = max(0, i - 15); zz = torch.cat([z[i0:i + 1], z[i:i + 1]])[None].float()
                fr = vae.decode(zz)[0] * qs + qm; nl.append(fr[4 * (i - i0):4 * (i - i0) + 4])
            nl = torch.cat(nl).cpu().numpy().astype(np.float64)[:L]
        streamed.append(y); oneshot.append(full); nolook.append(nl); reproj.append(enc_dec(y))
        ref.append(np.load(f'{E}/offline/s0/{ci}.npy').astype(np.float64)[:L])
    smb = [sm[ci] for ci in cis]
    R['B clips'] = cis; R['B streamed vs enc-s0 offline s0 file max|diff|'] = float(max(np.abs(a_ - b_).max() for a_, b_ in zip(streamed, ref)))
    R['B oneshot vs streamed max|diff| rad'] = float(max(np.abs(a_ - b_).max() for a_, b_ in zip(streamed, oneshot)))
    gsub = [gt[ci] for ci in cis]; gs, _ = stat(gsub, 50, ARM_E); R['B GT (same clips)'] = gs
    for lab, cl in (('streamed (1-latent look-ahead, emitted)', streamed), ('one-shot decode', oneshot), ('no look-ahead decode', nolook),
                    ('streamed re-encoded + decoded', reproj)):
        d, da = stat(cl, 50, ARM_E, thr); R[f'B {lab}'] = d; R[f'B {lab} where'] = where(cl, da, thr, smb)
    open(out, 'w').write(json.dumps(R, indent=1)); print('VAE_DIAG_DONE')


if __name__ == '__main__':
    main()
