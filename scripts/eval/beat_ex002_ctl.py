"""11-09 EX002 control / keyframe win rules in code (owner B; companion of A's scripts/eval/beat_ex002.py, same say() / fair() style).
Reads the stored result JSONs and applies the registered gates (docs/criteria/EX002_CTL.md) with the audit reading rules
(refine-logs/EVAL_AUDIT_1108.md): STG gain only with the shuffle-follow check, palm angle SIGNED (root cause row 51; earlier sign-agnostic
palm PASSes are void), release continuity as median + p95 + max, (i) / (ii) against the no-token C + delta of the same base / protocol.
Every verdict is checked against the one recorded in EX002_CTL.md (EXPECTED); a mismatch prints MISMATCH and makes the exit code 1.
    python scripts/eval/beat_ex002_ctl.py [--json OUT]"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

X = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/control-token-1105/ex002'; D = X + '/diag'
OUT, MISMATCH = {}, []


def J(p): return json.load(open(p))


def say(table, item, verdict, reason, expected=None):
    ok = expected is None or expected == verdict
    tag = '' if expected is None else (' [reported: %s]' % expected if ok else ' MISMATCH (reported: %s)' % expected)
    print(f'  {item:34s} {verdict:7s} {reason}{tag}')
    OUT.setdefault(table, {})[item] = dict(verdict=verdict, reason=reason, expected=expected, match=ok)
    if not ok: MISMATCH.append(f'{table}/{item}')


def fair(s): print('  fairness: ' + s)


def header(t, designs): print(f'\n== {t}  (designs / variants tried: {designs})')


def mean(d, row, key, seeds=(0, 1, 2)):
    v = [d[f'{row}_s{s}'][key] for s in seeds if f'{row}_s{s}' in d and d[f'{row}_s{s}'].get(key) is not None]
    return float(np.mean(v)) if v else float('nan')


def gate_cd(F, none, rad, seeds=(0, 1, 2)):
    c = np.array([F[f'{none}_s{s}'][rad]['FGD_v2'] for s in seeds]); C = c.mean(); return C + max(0.03 * C, 2 * c.std(ddof=1)), C


def fgd(F, row, rad, seeds=(0, 1, 2)): return float(np.mean([F[f'{row}_s{s}'][rad]['FGD_v2'] for s in seeds]))


def stg_ok(G, S, row, none, gain_min=0.10, seeds=(0, 1, 2)):
    gain = mean(G, row, 'stg_g', seeds) - mean(G, none, 'stg_g', seeds)
    follow = S is not None and mean(S, row.replace('given', 'shuffle'), 'stg_g', seeds) > mean(S, row.replace('given', 'shuffle'), 'stg_o', seeds)
    return gain >= gain_min and follow, gain, follow


def lanes():
    header('CONTROL LANES 1-4 (sibag-s2 base)', 'lane 1 (ctl fine-tune), lane 2 (rcp recipe control + ctl2 light), lane 3 (base lr 1e-6), lane 4 (true freeze); finger override (module 2) on each')
    for name, d, g, s, row, none, fo, exp in [('lane 1', 'lane1', 'EVAL_ex002ctl.json', 'EVAL_ex002ctl_shuffle.json', 'ex002ctl_given', 'ex002ctl_none', None, 'FAIL'),
                                               ('lane 2 ctl2', 'lane2', 'EVAL_given.json', 'EVAL_shuffle.json', 'ctl2_given', 'ctl2_none', 'ctl2fo_given', 'FAIL'),
                                               ('lane 3 ctl3', 'lane3', 'EVAL_given.json', None, 'ctl3_given', 'ctl3_none', 'ctl3fo_given', 'FAIL')]:
        G = J(f'{X}/{d}/{g}'); S = J(f'{X}/{d}/{s}') if s else None
        gain = mean(G, row, 'stg_g') - mean(G, none, 'stg_g'); fh = mean(G, fo or row, 'fh_g'); fg = mean(G, row, 'FGD_v2')
        ok = gain >= 0.10 and fh >= 0.9 and fg <= 99.37
        say('lanes', name, 'PASS' if ok else 'FAIL', f'STG gain {gain:+.3f}; finger hit {fh:.3f}{" (override)" if fo else ""}; FGD v2 (dense given) {fg:.1f} vs gate 99.37', exp)
    F = J(f'{X}/lane4/FGDOUT_sparse.json'); G = J(f'{X}/lane4/EVAL_given.json')
    gain = mean(G, 'ctl4_given', 'stg_g') - mean(G, 'ctl4_none', 'stg_g'); i_ = fgd(F, 'ctl4sp_given', 'R1'); ii_ = fgd(F, 'ctl4spfo_given', 'R0')
    say('lanes', 'lane 4 ctl4 (true freeze)', 'PASS' if (gain >= .10 and i_ <= 100.44 and ii_ <= 99.37) else 'FAIL',
        f'STG gain {gain:+.3f}; (i) sparse outside +-1 s {i_:.2f} vs 100.44; (ii) whole clip (override) {ii_:.2f} vs 99.37', 'FAIL')
    fair('lanes 1-4 share base sibag-s2, data-v36_1 train, the same plans / scorer; they differ only in the fine-tune recipe (frozen vs not).')


def m5():
    header('MODULE 5 / 5b / 5c (command-local blend)', '5 (symmetric), 5b (asymmetric), 5b fresh-plan confirmation, port to czm, 5c (smooth composition, enc-s0)')
    G = J(f'{X}/m5/EVAL_given.json'); gain = mean(G, 'm5_given', 'stg_g') - mean(G, 'none', 'stg_g')
    say('m5', 'module 5', 'PASS' if gain >= .10 else 'FAIL', f'STG gain {gain:+.3f} (gate +0.10)', 'FAIL')
    G = J(f'{X}/m5b/EVAL_given.json'); S = J(f'{X}/m5b/EVAL_shuffle.json'); F = J(f'{X}/m5b/FGDOUT_sparse.json')
    gain = mean(G, 'm5b_given', 'stg_g') - mean(G, 'none', 'stg_g'); fol = mean(S, 'm5b_shuffle', 'stg_g') > mean(S, 'm5b_shuffle', 'stg_o') if 'm5b_shuffle_s0' in S else True
    i_, ii_, iifo = fgd(F, 'm5bsp_given', 'R1'), fgd(F, 'm5bsp_given', 'R0'), fgd(F, 'm5bspfo_given', 'R0'); fh = mean(G, 'm5bfo_given', 'fh_g')
    ok = gain >= .10 and fol and i_ <= 100.44 and ii_ <= 99.37 and iifo <= 99.37 and fh >= .9
    say('m5', 'module 5b (old plans)', 'PASS' if ok else 'FAIL', f'STG gain {gain:+.3f} (shuffle follows {fol}); (i) {i_:.2f}/100.44; (ii) {ii_:.2f} / fo {iifo:.2f} vs 99.37; finger (fo) {fh:.3f}', 'PASS')
    P = f'{X}/m5b-confirm/fresh'; G = J(P + '/EVAL_given.json'); S = J(P + '/EVAL_shuffle.json'); F = J(P + '/FGDOUT_sparse.json'); B = J(P + '/BOOT_m5b.json'); sd = (3, 4, 5)
    gain = mean(G, 'm5b_given', 'stg_g', sd) - mean(G, 'none', 'stg_g', sd); fol = mean(S, 'm5b_shuffle', 'stg_g', sd) > mean(S, 'm5b_shuffle', 'stg_o', sd)
    gi, Ci = gate_cd(F, 'none', 'R1', sd); gii, Cii = gate_cd(F, 'none', 'R0', sd)
    i_, ii_, iifo = fgd(F, 'm5bsp_given', 'R1', sd), fgd(F, 'm5bsp_given', 'R0', sd), fgd(F, 'm5bspfo_given', 'R0', sd)
    ok = gain >= .10 and fol and i_ <= gi and ii_ <= gii and iifo <= gii
    say('m5', 'module 5b fresh-plan confirmation', 'PASS' if ok else 'FAIL', f'STG gain {gain:+.3f} CI [{B["ci95_forearm"][0]:+.3f}, {B["ci95_forearm"][1]:+.3f}] (shuffle {fol}); (i) {i_:.2f}/{gi:.2f}; (ii) {ii_:.2f} / fo {iifo:.2f} vs {gii:.2f}', 'PASS')
    for pr in ('offline', 'psync'):
        Bp = J(f'{X}/port-czm/{pr}/BOOT_m5b.json')
        say('m5', f'port to czm ({pr})', 'PASS' if Bp['gain_forearm'] >= .10 else 'FAIL', f'STG gain {Bp["gain_forearm"]:+.3f}', 'FAIL')
    for pr in ('offline', 'psync'):
        R = f'{X}/m5c-enc/{pr}'; G = J(R + '/EVAL_given.json'); S = J(R + '/EVAL_shuffle.json'); F = J(R + '/FGDOUT_sparse.json')
        ok, gain, fol = stg_ok(G, S, 'tok5c', 'none') if False else (None, mean(G, 'tok5c', 'stg_g') - mean(G, 'none', 'stg_g'), mean(S, 'tok5c', 'stg_g') > mean(S, 'tok5c', 'stg_o'))
        gii, _ = gate_cd(F, 'none', 'R0'); ii_ = fgd(F, 'tok5c', 'R0')
        say('m5', f'module 5c on enc-s0 ({pr})', 'PASS' if (gain >= .10 and fol and ii_ <= gii) else 'FAIL', f'STG gain {gain:+.3f} (shuffle {fol}); (ii) {ii_:.2f} vs {gii:.2f}', 'FAIL')
    fair('5b, the 5b confirmation and 5c reuse the same control-module generations of their base and change only the post-hoc blend weight.')


def present_signed(f, row):
    d = J(f)[row]; return d.get('present_hand_palm_angle_signed_median'), d.get('present_hand_raised_share_mean')


def m78():
    header('MODULES 7 / 8 / 7b / 8b', '7 / 8 on czm, 7b / 8b on sibag-s2')
    for pr in ('offline', 'psync'):
        A = J(f'{X}/m78-czm/m7/{pr}/AIMERR.json')['m5b_given']['fa']['total_median']
        say('m78', f'module 7 czm ({pr})', 'PASS' if A <= 25 else 'FAIL', f'apex angle {A:.1f} (gate 25)', 'FAIL')
        pal, rai = present_signed(f'{D}/palm_signed/palm/m8_{pr}.json', 'm8')
        say('m78', f'module 8 czm ({pr})', 'PASS' if (rai >= .8 and pal <= 45) else 'FAIL', f'presenting hand raised {rai:.3f}; SIGNED palm {pal:.1f} (sign-agnostic metric void)', 'FAIL')
        A = J(f'{X}/m78b-sibag/m7b/{pr}/AIMERR.json')['m5b_given']['fa']['total_median']
        say('m78', f'module 7b sibag ({pr})', 'PASS' if A <= 25 else 'FAIL', f'apex angle {A:.1f}', 'FAIL')
        pal, rai = present_signed(f'{D}/palm_signed/palm/m8b_{pr}.json', 'm8b')
        say('m78', f'module 8b sibag ({pr})', 'PASS' if (rai >= .8 and pal <= 45) else 'FAIL', f'presenting hand raised {rai:.3f}; SIGNED palm {pal:.1f}', 'FAIL')


def rel(f, row):
    r = J(f)[row]; return f'release med / p95 / max {r["median"]:.1f} / {r["p95"]:.1f} / {r.get("max", float("nan")):.1f}', r['median'] <= 29.8


def keyframes():
    header('KEYFRAME MODULES 9 / 9b / 9c / 9c\'\' / 9d', "9 (sibag, re-conditioning), 9b (enc), m9b5c, 9c (hindsight pool + control), 9c' (quick), 9c'' (T + palm fix), 9d (quick, 5 rows)")
    for pr in ('offline', 'psync'):
        R = f'{X}/m9-sibag/{pr}'; A = J(R + '/AIMERR.json')['m9']['fa']['total_median']; pal, rai = present_signed(f'{D}/palm_signed/palm/m9_{pr}.json', 'm9')
        F = J(R + '/FGDOUT_sparse.json'); gi, _ = gate_cd(F, 'none', 'R1'); i_ = fgd(F, 'm9', 'R1'); rs, rok = rel(R + '/RELEASE_dense.json', 'm9')
        say('kf', f'module 9 sibag ({pr})', 'PASS' if (A <= 25 and pal <= 45 and i_ <= gi and rok) else 'FAIL', f'angle {A:.1f}; SIGNED palm {pal:.1f} (was reported 1.0, void); (i) {i_:.1f}/{gi:.1f}; {rs}', 'FAIL')
    for name, base, row, pfile, prow in (('module 9b', 'm9b-enc', 'm9b', 'm9b', 'm9b'), ('m9b5c', 'm5c-enc', 'm9b5c', 'm9b', 'm9b5c')):
        for pr in ('offline', 'psync'):
            R = f'{X}/{base}/{pr}'; A = J(R + '/AIMERR.json')[row]['fa']['total_median']; pal, rai = present_signed(f'{D}/palm_signed/palm/{pfile}_{pr}.json', prow)
            say('kf', f'{name} ({pr})', 'PASS' if (A <= 25 and pal <= 45 and rai >= .8) else 'FAIL', f'angle {A:.1f}; raised {rai:.3f}; SIGNED palm {pal:.1f}', 'FAIL')
    for pr in ('offline', 'psync'):
        R = f'{X}/m9c-enc/{pr}'; A = J(R + '/AIMERR.json'); P = J(f'{D}/palm_signed/PRESENT_m9c_{pr}.json'); G = J(R + '/EVAL_given.json'); S = J(R + '/EVAL_shuffle.json'); F = J(R + '/FGDOUT_sparse.json')
        gii, _ = gate_cd(F, 'none', 'R0'); gain = mean(G, 'm9c', 'stg_g') - mean(G, 'none', 'stg_g'); fol = mean(S, 'm9c', 'stg_g') > mean(S, 'm9c', 'stg_o'); rs, rok = rel(R + '/RELEASE_dense.json', 'm9c')
        a9 = A['m9c']['fa']['total_median']; pal = P['m9c']['present_hand_palm_angle_signed_median']; ii_ = fgd(F, 'm9c', 'R0')
        say('kf', f'module 9c ({pr})', 'PASS' if (a9 <= 25 and pal <= 45 and gain >= .10 and fol and ii_ <= gii and rok) else 'FAIL',
            f'angle {a9:.1f} (control row {A["m9cctl"]["fa"]["total_median"]:.1f}); SIGNED palm {pal:.1f}; STG {gain:+.3f} (shuffle {fol}); (ii) {ii_:.1f}/{gii:.1f}; {rs}', 'FAIL')
        R = f'{X}/m9cpp-enc/{pr}'; A = J(R + '/AIMERR.json'); P = J(R + '/PRESENT.json'); G = J(R + '/EVAL_given.json'); S = J(R + '/EVAL_shuffle.json'); F = J(R + '/FGDOUT_sparse.json')
        gii, _ = gate_cd(F, 'none', 'R0'); gi, _ = gate_cd(F, 'none', 'R1'); gain = mean(G, 'm9c', 'stg_g') - mean(G, 'none', 'stg_g'); fol = mean(S, 'm9c', 'stg_g') > mean(S, 'm9c', 'stg_o')
        rs, rok = rel(R + '/RELEASE_dense.json', 'm9c'); a9 = A['m9c']['fa']['total_median']; el = A['m9c']['fa']['elevation_median_target_above_shoulder']
        pal, rai = P['m9c']['present_hand_palm_angle_signed_median'], P['m9c']['present_hand_raised_share_mean']; ii_, i_ = fgd(F, 'm9c', 'R0'), fgd(F, 'm9c', 'R1')
        others = gain >= .10 and fol and abs(el) <= 5 and pal <= 45 and rai >= .8 and ii_ <= gii and i_ <= gi and rok
        say('kf', f"module 9c'' ({pr})", 'PASS' if (a9 <= 25 and others) else 'FAIL', f'angle {a9:.1f} (gate 25); all other gates {others}: STG {gain:+.3f} shuffle {fol}, elev {el:+.1f}, palm {pal:.1f}, raised {rai:.3f}, (i) {i_:.1f}/{gi:.1f}, (ii) {ii_:.1f}/{gii:.1f}, {rs}', 'FAIL')
    H = {pr: J(f'{D}/quick_9c/q9cp/{pr}/H9d.json') for pr in ('offline', 'psync')}; P = {pr: J(f'{D}/quick_9c/q9cp/{pr}/P9d.json') for pr in ('offline', 'psync')}
    sp = [H[pr]['d1']['hold_speed_median'] for pr in ('offline', 'psync')]; pal = [P[pr]['d12']['present_hand_palm_angle_signed_median'] for pr in ('offline', 'psync')]
    kill = min(sp) < 0.2 or min(pal) > 45
    say('kf', '9d quick lane (kill rule)', 'KILLED' if kill else 'GO', f'diff-IK hold speed {sp[0]:.3f} / {sp[1]:.3f} (< 0.2 kills); (1)+(2) signed palm {pal[0]:.0f} / {pal[1]:.0f} (worse than 9c\'\' kills)', 'KILLED')
    fair("9b / 9c / 9c'' / 9d are post-processing on the same enc-s0 control generations (plans 1112, seeds 0-2); rows differ only in the insertion settings.")


def m10():
    header('MODULE 10 (enc-s0)', '(a) library shapes, (b) 4000-step ctl modules, (a)+(b)')
    for row, f, exp in (('tok10a', 'tok10a', 'FAIL'), ('tok10b', 'tok10b-tok10ab', 'FAIL'), ('tok10ab', 'tok10b-tok10ab', 'PASS')):
        res = []
        for pr in ('offline', 'psync'):
            R = f'{X}/m10-enc/{pr}'; G = J(f'{R}/EVAL_given_{f}.json'); S = J(f'{R}/EVAL_shuffle_{f}.json'); F = J(f'{R}/FGDOUT_sparse_{f}.json'); B = J(f'{R}/BOOT_{row}.json')
            gain = mean(G, row, 'stg_g') - mean(G, 'none', 'stg_g'); fol = mean(S, row, 'stg_g') > mean(S, row, 'stg_o'); fh = mean(G, row, 'fh_g')
            gi, _ = gate_cd(F, 'none', 'R1'); gii, _ = gate_cd(F, 'none', 'R0'); i_, ii_ = fgd(F, row, 'R1'), fgd(F, row, 'R0')
            ok = gain >= .10 and fol and fh >= .9 and i_ <= gi and ii_ <= gii; res.append(ok)
            print(f'    {row} {pr}: STG {gain:+.3f} CI [{B["ci95_forearm"][0]:+.3f}, {B["ci95_forearm"][1]:+.3f}] shuffle {fol}; finger {fh:.3f} (library rows: circular); (i) {i_:.2f}/{gi:.2f}; (ii) {ii_:.2f}/{gii:.2f}')
        say('m10', row, 'PASS' if all(res) else 'FAIL', 'offline AND psync (see lines above)', exp)
    fair('module-10 rows share the enc-s0 none stream and plans 1112; (b) / (a)+(b) use the 4000-step control modules, (a) the 2000-step ones.')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--json', default=None); a = ap.parse_args()
    for f in (lanes, m5, m78, keyframes, m10):
        try: f()
        except Exception as ex: print(f'  ERROR in {f.__name__}: {ex!r}'); MISMATCH.append(f'{f.__name__}: error')
    print('\nMISMATCH:', MISMATCH if MISMATCH else 'none')
    if a.json: Path(a.json).write_text(json.dumps(OUT, indent=1))
    sys.exit(1 if MISMATCH else 0)


if __name__ == '__main__':
    main()
