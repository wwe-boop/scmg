"""11-05 How much does ECHO-G use its text? The official SGDiT sampler (echo_g.flow.sample_euler, 8 Euler steps, guidance 1
as published, official per-clip initial noise stable_seed(stem, seed)) with three TEXT inputs and everything else fixed
(same audio features, same length, same noise):
  own    : the released condition (Qwen3.5-4B token features + word-derived token centres)      == the official export
  notext : ECHO-G's own null text = what its condition dropout / CFG unconditional branch uses: ONE zero token, unmasked,
           time distance 0 (echo_g.flow.apply_condition_dropout / sample_euler; also what decode_condition gives an
           'audio-only' config).  NOTE: ECHO-G drops audio and text TOGETHER in training (one joint dropout draw), so audio +
           null text is a combination it never saw -- unlike SCMG, whose text / audio dropouts are independent.
  swap   : the text tokens and token centres of a DONOR clip (--swap-map {stem: donor}, shared with SCMG infer_test.py
           --text-swap-map), centres rescaled by F / F_donor (proportional alignment to this clip's duration, as for SCMG).
Output: <out>/<mode>_seed_<s>/<stem>.pt in the official prediction format (robot_repr [F, 39] physical ...).
    PYTHONPATH=<official>/src python scripts/echog_g1/echog_textuse.py --checkpoint best.pt --config sgdit_audio_text.yaml \
        --cond-dir condition_30fps --meta <data_root> --stems subset.txt --swap-map map.json --out <dir> --modes own notext swap --seed 0
"""
import argparse, json, time
from pathlib import Path
import torch

from echo_g.condition_provenance import load_condition_manifest
from echo_g.data import decode_condition, load_aligned_lengths, read_stems
from echo_g.flow import sample_euler
from echo_g.inference import load_model_and_config
from echo_g.utils import stable_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--cond-dir', type=Path, required=True); ap.add_argument('--meta', type=Path, required=True)
    ap.add_argument('--stems', type=Path, required=True); ap.add_argument('--swap-map', type=Path)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--modes', nargs='+', default=['own', 'notext', 'swap'])
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--guidance-scale', type=float, default=1.0); ap.add_argument('--shard', default='0/1')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(); dev = torch.device(a.device)
    model, config, mean, std, _ = load_model_and_config(a.checkpoint, a.config, dev)
    if config.data.conditioning != 'audio-text': raise SystemExit('audio-text checkpoint expected')
    fps = float(config.data.fps); mean, std = mean.to(dev), std.to(dev)
    lengths = load_aligned_lengths(a.meta / config.data.aligned_lengths_file)
    man = load_condition_manifest(a.meta / config.data.condition_manifest, config.data.condition_manifest_sha256)
    swap = json.loads(a.swap_map.read_text()) if a.swap_map else None
    if 'swap' in a.modes and swap is None: raise SystemExit('--modes swap needs --swap-map')
    stems = read_stems(a.stems); si, sn = (int(x) for x in a.shard.split('/')); stems = stems[si::sn]

    def cond(stem):
        cp = a.cond_dir / f'{stem}.pt'; F = min(lengths[stem], config.data.max_frames)
        return decode_condition(torch.load(cp, map_location='cpu', weights_only=True), stem, config.data, config.model, F,
                                condition_path=cp, condition_manifest=man)

    t0 = time.time()
    for n, stem in enumerate(stems):
        c = cond(stem); F = c['frames']; audio = c['audio'].to(dev)
        g = torch.Generator(device=dev); g.manual_seed(stable_seed(stem, a.seed))
        noise = torch.randn(F, model.motion_dim, generator=g, device=dev)       # == echo_g.inference initial noise
        ft = torch.arange(F, device=dev, dtype=torch.float32) / fps
        for mode in a.modes:
            o = a.out / f'{mode}_seed_{a.seed:03d}' / f'{stem}.pt'
            if o.exists(): continue
            o.parent.mkdir(parents=True, exist_ok=True); extra = {}
            if mode == 'own':
                tx, centre = c['text'].to(dev), c['token_centers'].to(dev)
            elif mode == 'swap':
                d = cond(swap[stem]); tx = d['text'].to(dev)
                centre = d['token_centers'].to(dev) * (F / d['frames'])                # proportional time alignment
                extra = dict(text_donor=swap[stem], time_scale=F / d['frames'])
            elif mode == 'notext':
                tx = centre = None
            else: raise SystemExit(f'mode {mode}')
            if tx is None:                                                     # the official null text
                text = torch.zeros(1, 1, config.model.text_dim, device=dev); tpm = torch.zeros(1, 1, dtype=torch.bool, device=dev)
                td = torch.zeros(1, F, 1, device=dev)
            else:
                text = tx[None]; tpm = torch.zeros(1, len(tx), dtype=torch.bool, device=dev); td = (ft[:, None] - centre[None])[None]
            batch = dict(audio=audio[None], text=text, motion_mask=torch.ones(1, F, dtype=torch.bool, device=dev),
                         text_padding_mask=tpm, time_distance=td)
            y = sample_euler(model, batch, a.steps, a.guidance_scale, initial_noise=noise[None].clone())
            r = (y[0].float() * std + mean).cpu().contiguous()
            if not torch.isfinite(r).all(): raise ValueError(f'{stem}: non-finite')
            torch.save(dict(robot_repr=r, real_num_frames=F, fps=fps, stem=stem, seed=a.seed, representation_units='physical',
                            representation_schema=config.data.representation_schema, source_checkpoint=str(a.checkpoint),
                            text_mode=mode, steps=a.steps, guidance_scale=a.guidance_scale, **extra), o.with_suffix('.tmp'))
            o.with_suffix('.tmp').rename(o)
        if n % 25 == 0: print(f'{n + 1}/{len(stems)} {stem} F={F} {time.time() - t0:.0f}s', flush=True)
    print('ECHOG_TEXTUSE_DONE', len(stems), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
