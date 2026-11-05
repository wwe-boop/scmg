"""Per-clip motion-energy probe on the official G1 metric definitions (11-05).

Hypothesis: dDiv and dBA are one problem -- the generated motion energy does not follow the speaker / utterance.
  * Div (EMAGE L1div) = within-clip mean absolute deviation of the 30 body positions = motion amplitude of the clip.
  * BA (audio_to_motion GAHR) rises with the density of motion beats and with the number of upper bodies that move
    above the mmae threshold (a body without beats scores 0).
For every clip this writes, for prediction and GT, exactly the official per-clip quantities (Div contribution / frame,
BA) plus their ingredients (motion beats per second, fraction of upper bodies with beats, mean normalized speed), and
audio prosody (RMS dB mean / std, onset rate). Same FK and functions as eval_g1_motion_cls.py (imported, not copied).

    python scripts/echog_g1/energy_probe.py --pred-dir D --ref-dir R --wav-dir W --stems S --out probe.jsonl
Run with the echog python (PYTHONPATH=<official>/src).
"""
import argparse, json, os
import numpy as np

from echo_g.evaluation import g1_motion_cls as E

ap = argparse.ArgumentParser()
ap.add_argument('--pred-dir', required=True); ap.add_argument('--ref-dir', required=True); ap.add_argument('--wav-dir', required=True)
ap.add_argument('--stems', required=True); ap.add_argument('--out', required=True); ap.add_argument('--fps', type=float, default=30.0)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--mmae-file', required=True, help='same per-body mean velocity file as the official evaluator (MMAE)')
a = ap.parse_args()
E._G1_MEAN_VEL_PATH = a.mmae_file   # exactly what eval_g1_motion_cls.py main() does with --mmae-file

import librosa, scipy.signal
ba = E.BeatAlign(a.fps)          # official defaults (sigma 0.3, order 7, threshold 0.10, audio_to_motion)
mmae = E._g1_mean_vel()


def motion_stats(J):
    """J (T,30,3) -> per-clip Div / frame, BA ingredients (same definitions as the official classes)."""
    T = J.shape[0]
    flat = J.reshape(T, -1)
    div = float(np.sum(np.abs(flat - flat.mean(0))) / T)                       # Div.compute contribution per frame
    spd = E.body_speed_central(J, a.fps)
    n_beats, with_beats, nspd = [], 0, []
    for b in ba.upper_bodies:
        v = spd[:, b]; norm = v / (mmae[b] + 1e-8) if (mmae is not None and mmae[b] > 1e-8) else v
        minima = scipy.signal.argrelextrema(v, np.less, order=ba.order)[0]
        beats = np.intersect1d(minima, np.where(norm > ba.bc_threshold)[0])
        n_beats.append(beats.size); with_beats += beats.size > 0; nspd.append(float(norm.mean()))
    return dict(div=div, beats_per_s=float(np.mean(n_beats)) * a.fps / T, frac_bodies_beats=with_beats / len(ba.upper_bodies),
                norm_speed=float(np.mean(nspd)))


stems = [s.strip() for s in open(a.stems) if s.strip()]
if a.limit: stems = stems[:a.limit]
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
n = 0
with open(a.out + '.tmp', 'w') as fo:
    for s in stems:
        pr = E.load_repr(f'{a.pred_dir}/{s}.pt', a.fps); gr = E.load_repr(f'{a.ref_dir}/{s}.pt', a.fps)
        if pr is None or gr is None or pr.shape[0] < 4 or gr.shape[0] < 4: continue
        pr, gr = E.crop_to_common_length(pr, gr)
        pj, gj = E.fk_positions_ba(pr), E.fk_positions_ba(gr)
        wav = E.wav_of(a.wav_dir, s)
        rec = dict(stem=s, spk='_'.join(s.split('_')[:3]), T=int(pj.shape[0]))
        for k, J in (('gen', pj), ('gt', gj)):
            st = motion_stats(J); st['ba'] = ba._beat_consistency(J, wav)
            rec.update({f'{k}_{x}': v for x, v in st.items()})
        y, _ = librosa.load(wav, sr=16000); y = y[: int(rec['T'] / a.fps * 16000)]
        rms = librosa.feature.rms(y=y, hop_length=320)[0]; db = 20 * np.log10(rms + 1e-5)
        on = librosa.onset.onset_detect(y=y, sr=16000, hop_length=512, units='time')
        rec.update(a_db_mean=float(db.mean()), a_db_std=float(db.std()), a_onset_rate=len(on) / max(len(y) / 16000, 1e-3))
        fo.write(json.dumps(rec) + '\n'); n += 1
os.replace(a.out + '.tmp', a.out)
print('wrote', a.out, n, 'clips')
