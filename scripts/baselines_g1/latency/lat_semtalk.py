"""SemTalk (Speaker-2 weights) latency with exactly the benchmark pipeline of ../semtalk_infer.py (official _g_test
decoding, official features, demo seed), batch 1.  Whole-utterance pipeline: HuBERT runs on the whole clip and every
64-frame window needs its sentence text / emotion label, so compute starts once the whole clip has arrived.
Stages: audio_feat = waveform loading + rhythm/pause 'beat' features + mel length + HuBERT-large + emotion2vec_plus_large
labels; text_enc = frame word ids + per-window sentences + CLIP ViT-B/32 (sentence and emotion label); generate = Base +
Sparse window loop; decode = final RVQ decode + VAEConvZero global motion.  The official audio-only inference() adds
WhisperX ASR (not used in the benchmark: the transcript is given), so it is not timed.  SMPL-X -> G1 (GMR) is a
separate row (lat_gmr.py)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import lat_common as LC  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    mon = LC.GPUMon(); host0 = LC.host_snapshot()
    import semtalk_infer as S
    dev = 'cuda'; clk = LC.Sync()
    torch.manual_seed(0); np.random.seed(0)
    args, base, sparse, vq, ctx = S.build(dev)
    seed_lat, trans0 = S.seed_latent(args, ctx, dev)
    fx = S.Feat(dev)
    stems = LC.stems50()[:a.n or None]; rows = []
    for k, stem in enumerate(stems):
        st = dict(audio_feat=0., text_enc=0., generate=0., decode=0.); last = [clk.now()]
        def mark(stage):
            t = clk.now(); st[stage] += t - last[0]; last[0] = t
        n = int(torch.load(S.EG / 'motion_39d_30fps' / f'{stem}.pt', map_location='cpu', weights_only=True)['real_num_frames'])
        words = json.load(open(S.EG / 'annotations/words' / f'{stem}.json'))['words']
        last[0] = clk.now()
        f = fx(S.EG / 'audio' / f'{stem}.wav', words, n, mark=mark)
        poses, trans, exps = S.generate(args, base, sparse, vq, ctx, f, seed_lat, trans0, dev, mark=mark)
        mark('decode')
        import soundfile as sf
        dur = sf.info(str(S.EG / 'audio' / f'{stem}.wav')).duration
        comp = sum(st.values())
        rows.append(dict(stem=stem, dur=dur, frames30=n, windows=int(f['k']), stages=st, compute=comp, schedule=[(dur, comp, 0.0, n, 30.)]))
        print(f'{k + 1}/{len(stems)} {stem} dur={dur:.2f} comp={comp:.3f} {st}', flush=True)
    extra = dict(weights='HF X-Zhang/SemTalk weights_semtalk_no_smplx.zip (Speaker 2)', decoding='argmax (deterministic)',
                 frame_fps=30, not_timed='WhisperX ASR of the official audio-only inference() (transcript given in the benchmark)',
                 host_start=host0, host_end=LC.host_snapshot(), torch=torch.__version__)
    extra.update(LC.cuda_mem()); extra.update(mon.stop())
    LC.write(a.out, 'semtalk', rows, extra)


if __name__ == '__main__':
    main()
