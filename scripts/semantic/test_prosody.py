"""audio_prosody_only (11-07 §7.55 item 2) acceptance tests; helpers from test_ctl.py (same env).
 A default off: losses with the flag unset equal the losses of a StageCfg without the field (bitwise, same seed)
 B flag on: the loss is invariant to the 80 mel columns (mel replaced by noise -> identical loss) but not to prosody
 C flag off: the loss does depend on the mel columns (the test can fail)
Prints ALL_OK / SOME_FAIL."""
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ctl as TC


def main():
    ok = {}
    st, v, vae = TC.setup()
    m = TC.model({'adsc': True})
    b = TC.batch_of(v, n=8, seed=3)
    noisy = dict(b, feat=b['feat'].clone()); noisy['feat'][..., :80] = torch.randn_like(noisy['feat'][..., :80])
    pros = dict(b, feat=b['feat'].clone()); pros['feat'][..., 80:88] = torch.randn_like(pros['feat'][..., 80:88])
    L = lambda bb, **kw: TC.grads_losses(m, vae, bb, st, dict(TC.SC, **kw))[0]['loss']
    l0 = L(b); ok['A_default'] = l0 == L(b, audio_prosody_only=False)
    lp = L(b, audio_prosody_only=True)
    ok['B_mel_invariant'] = lp == L(noisy, audio_prosody_only=True)
    ok['B_prosody_used'] = lp != L(pros, audio_prosody_only=True)
    ok['C_mel_used_when_off'] = l0 != L(noisy)
    print(json.dumps(ok)); print('ALL_OK' if all(ok.values()) else 'SOME_FAIL')


if __name__ == '__main__':
    main()
