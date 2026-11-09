"""RoboGesture official MotionGenerator.generate latency (raw branch, seed 0; Mimi 24 kHz + 50-step flow, 30-frame blocks).
The official generate consumes the PCM block by block (0.96 / 1.04 s, alternating) after 2 blocks of silence history;
block j's 30 frames depend only on audio up to the end of block j (causal), so block j can start once its audio has
arrived (real-time arrival) and block j-1 is done.  The two history blocks need no input audio: they are scheduled at
t = 0 (they finish long before the first speech block has arrived).  The loop below is the official generate with
cuda-synchronised timers only (the output is checked bitwise against the benchmark raw_seed0 export).
Stages: audio_enc = Mimi streaming encode per block; generate = 50 Euler steps of the motion model; decode =
de-normalisation + host copy + 41 -> 29 joint mapping.  Official deployment additionally runs smooth_model_blocks +
CollisionFilter (CPU) on the whole utterance before playback ('filter', reported, not in compute); the robot plays
the utterance only after generation ("playback is whole-utterance") -- the deployment figures are given as
deploy_* (whole-utterance: wait for the full audio + all blocks + filter)."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lat_common as LC  # noqa: E402

ROOT = Path('/mnt/cpfs/zbl-cpfs-new/USERS/zehan/baselines/robogesture')
AUD = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/echog/data_root/audio')
RAW0 = Path('/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/baseline-repro-1104/robogesture/gen/raw_seed0')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
    a = ap.parse_args()
    mon = LC.GPUMon(); host0 = LC.host_snapshot()
    os.environ.setdefault('ROBOGESTURE_WEIGHTS_ROOT', str(ROOT / 'weights'))
    sys.path.insert(0, str(ROOT / 'official'))
    import robogesture.config as config
    config.MOTION_MODEL_PATH = ROOT / 'weights' / 'model.pt'
    from robogesture import model as official
    from robogesture.offline import read_wav
    from robogesture.motion import model_to_robot, smooth_model_blocks
    from robogesture.collision import CollisionFilter
    gen = official.MotionGenerator('cuda'); filt = CollisionFilter(); clk = LC.Sync(); dv = gen.device
    stems = LC.stems50()[:a.n or None]; rows = []
    for k, stem in enumerate(stems):
        pcm = read_wav(AUD / f'{stem}.wav'); total = len(pcm) // 2; dur = total / 24000.
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        st = dict(audio_enc=0., generate=0., decode=0.); sched = []; ticks = []
        with torch.inference_mode():
            mimi_cache = None
            audio_cache = torch.zeros(1, 25, 2, dtype=torch.long, device=dv)
            motion_cache = torch.zeros(1, 60, 41, device=dv)
            pending = bytearray(bytes(sum(gen.CHUNK_SAMPLES) * 2)); pending.extend(pcm)
            chunks = []; index = 0; consumed = 0
            while pending or index < 3:
                sc = gen.CHUNK_SAMPLES[index % 2]; raw = bytes(pending[:sc * 2]); del pending[:sc * 2]
                consumed += sc
                t0 = clk.now()
                samples = np.zeros(sc, dtype=np.float32); dec = np.frombuffer(raw, dtype='<i2')
                samples[:len(dec)] = dec.astype(np.float32) / 32768.0
                wf = torch.from_numpy(samples).reshape(1, 1, -1).to(dv)
                encd = gen.mimi.encode(wf, padding_cache=mimi_cache, num_quantizers=2, use_streaming=True)
                mimi_cache = encd.padding_cache; tokens = encd.audio_codes.transpose(1, 2)
                t1 = clk.now(); td = 0.
                if index < 2:
                    motion = torch.zeros(1, 30, 41, device=dv)
                else:
                    x_t = torch.randn(1, 30, 41, device=dv)
                    for step in torch.linspace(0.0, 0.98, 50, device=dv):
                        window = torch.cat((audio_cache, tokens), dim=1)
                        if window.shape[1] < 38:
                            window = torch.cat((torch.zeros(1, 38 - window.shape[1], 2, dtype=torch.long, device=dv), window), dim=1)
                        x_t = x_t + gen.model(window, x_t, motion_cache, step.expand(1))['pred_velocity'] / 50.0
                    motion = x_t.clamp(-5.0, 5.0)
                    t2 = clk.now()
                    chunks.append(gen.model.denormalize_motion(motion)[0].cpu().numpy())
                    td = clk.now() - t2
                t3 = clk.now()
                audio_cache = torch.cat((audio_cache, tokens), dim=1)[:, -25:]
                motion_cache = torch.cat((motion_cache, motion), dim=1)[:, -60:]
                ta, tg = t1 - t0, clk.now() - t1 - td
                st['audio_enc'] += ta; st['generate'] += tg; st['decode'] += td
                if index < 2:
                    sched.append((0.0, ta + tg, 0.0, 0, 30.))
                else:
                    ready = min(consumed - sum(gen.CHUNK_SAMPLES), total) / 24000.
                    sched.append((ready, ta + tg + td, (index - 2) * 1.0, 30, 30.))
                    ticks.append(dict(t=(consumed - sc - sum(gen.CHUNK_SAMPLES)) / 24000., len=sc / 24000., audio_enc=ta, generate=tg, decode=td))
                index += 1
        raw41 = np.concatenate(chunks).astype(np.float32)
        if k < 5:
            ref = np.load(RAW0 / f'{stem}.npz')['raw_motion41'] if (RAW0 / f'{stem}.npz').exists() else None
            if ref is not None: assert np.array_equal(raw41, ref), f'{stem}: timed loop differs from the official raw_seed0'
        t0 = time.perf_counter()
        res = filt.filter(smooth_model_blocks(model_to_robot(raw41)), np.zeros(12), np.zeros(30, dtype=np.float32))
        tf = time.perf_counter() - t0
        n = int(round(dur * 30))
        sched = [(r, c, t0_, min(nn, max(0, n - int(round(t0_ * 30)))), f) for r, c, t0_, nn, f in sched]   # frames past the audio = block padding
        comp = sum(st.values())
        rows.append(dict(stem=stem, dur=dur, blocks=index, stages=st, filter_s=tf, compute=comp, schedule=sched, ticks=ticks,
                         deploy_schedule=[(dur, comp + tf, 0.0, n, 30.)]))
        print(f'{k + 1}/{len(stems)} {stem} dur={dur:.2f} comp={comp:.3f} filter={tf:.2f} {st}', flush=True)
    use = rows[LC.N_WARM:]
    dep = LC.summarize([dict(r, schedule=r['deploy_schedule'], compute=r['compute'] + r['filter_s']) for r in rows], {})
    extra = dict(seed=0, variant='raw (official generate, --raw branch)', frame_fps=30, parity='first 5 clips bitwise == raw_seed0',
                 filter_ms_per_clip_p50=float(np.median([r['filter_s'] for r in use]) * 1e3),
                 filter_rtf=float(sum(r['filter_s'] for r in use) / sum(r['dur'] for r in use)),
                 deploy_whole_utterance=dict((kk, dep[kk]) for kk in ('algorithmic_latency_s', 'e2e_latency_s', 'first_frame_latency_s', 'rtf')),
                 host_start=host0, host_end=LC.host_snapshot(), torch=torch.__version__)
    extra.update(LC.cuda_mem()); extra.update(mon.stop())
    LC.write(a.out, 'robogesture', rows, extra)


if __name__ == '__main__':
    main()
