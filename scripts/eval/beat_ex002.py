"""11-08 EX002 win rules in code (autonomous-ml-research stage 4; coordinator). Reads the stored result JSONs and applies, per table,
exactly the registered gates (docs/criteria/*.md) and the audit reading rules (refine-logs/EVAL_AUDIT_1108.md): STG only with the
shuffle-follow check, FKD never alone (FGD v2 beside it), v3 for abruptness only (not a freeze detector), RKD never without the yield
components, no single yield gate after the audit, release / insertion continuity as median + p95 + max. Prints PASS / FAIL with the
reason, the design count per table, one fairness sentence per comparison, and checks every verdict against the one already reported
(EXPECTED); a mismatch is printed as MISMATCH and makes the exit code 1.
Tables encoded: base selection (EX002_SFSCOPE_1108), safety layer (EX002_CBF_1108 3c-1 .. 3c-7), interruption (EX002_INTERRUPT_1107 rule
addendum, full and half-B references, F2 baselines), yield-metric validity (yield3d, v4 it1 / it2, activity-only, PSS), design round
11-08 (D1, D2, D3, M1), control (EX002_CTL module 5b confirmation on fresh plans). NOT encoded here (owner B, exp/ctl-port): the other
control / keyframe modules (lanes 1-4, 5, 5c, port, 7 / 8, 7b / 8b, 9, 9b, 9c, 9c'', 9d, 10) -- listed, not judged.
    python scripts/eval/beat_ex002.py [--json OUT]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

B = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg'; X = B + '/ex002-interrupt-1107'; S = B + '/ex002-sfscope-1108'
R = os.path.expanduser('~/bulk/causal-train-1105/runs'); A = os.path.expanduser('~/bulk/causal-train-1105/audit-1108')
CT = B + '/control-token-1105/ex002'
OUT, MISMATCH = {}, []
GT_LIVE = dict(still=0.0166, wrist_speed=0.411, wrist_amp=0.120, held_share=0.023, stutter=1.026)   # lively_floor_ex002 GT (dev)


def J(p): return json.load(open(p))


def say(table, item, verdict, reason, expected=None):
    ok = expected is None or expected == verdict
    tag = '' if expected is None else (' [reported: %s]' % expected if ok else ' MISMATCH (reported: %s)' % expected)
    print(f'  {item:38s} {verdict:8s} {reason}{tag}')
    OUT.setdefault(table, {})[item] = dict(verdict=verdict, reason=reason, expected=expected, match=ok)
    if not ok: MISMATCH.append(f'{table}/{item}')


def fair(s): print('  fairness: ' + s)


def header(t, designs):
    print(f'\n== {t}  (designs / variants tried: {designs})')


# ------------------------------------------------------------------------------------------------- base selection
def base_selection():
    header('BASE SELECTION (EX002_SFSCOPE_1108)', 'self-forcing scope sf0 / all / encoders x 2 training seeds, + causal match czm-enc-s0 (7 lanes)')

    def lane(d):
        e = J(d + '/EVAL.json'); o = {}
        for P in ('offline', 'psync'):
            rows = [e[f'{P}-s{s}'] for s in range(3)]
            o[P] = dict(FGD=[r['FGD_v2'] for r in rows], FKD=float(np.mean([r['FKD'] for r in rows])),
                        **{k: float(np.mean([r['lively'][k] for r in rows])) for k in GT_LIVE})
        return o

    def guards(m):
        return all([m['still'] <= 2 * GT_LIVE['still'], m['wrist_speed'] >= .75 * GT_LIVE['wrist_speed'], m['wrist_amp'] >= .75 * GT_LIVE['wrist_amp'],
                    m['held_share'] <= .10, m['stutter'] <= 1.10 * GT_LIVE['stutter']])
    c, e = lane(S + '/czm-enc-s0'), lane(S + '/enc-s0')
    a_ = max(c['psync']['FGD']) < min(e['psync']['FGD']); b_ = np.mean(c['offline']['FGD']) <= 1.05 * np.mean(e['offline']['FGD'])
    c_ = c['psync']['FKD'] <= 1.10 * e['psync']['FKD']; e_ = guards(c['offline']) and guards(c['psync']); ge = guards(e['offline']) and guards(e['psync'])
    czm_ok = a_ and b_ and c_ and e_
    say('base', 'czm-enc-s0 vs enc-s0', 'PASS' if czm_ok else 'FAIL',
        f'(a) psync FGD range below {a_} ({min(c["psync"]["FGD"]):.1f}-{max(c["psync"]["FGD"]):.1f} vs {min(e["psync"]["FGD"]):.1f}-{max(e["psync"]["FGD"]):.1f}); '
        f'(b) offline <= 1.05x {b_}; (c) psync FKD <= 1.10x {c_} (FKD read with FGD, never alone); (e) guards {e_}', 'FAIL')
    say('base', 'enc-s0 guards (offline AND psync)', 'PASS' if ge else 'FAIL', 'liveliness floor (read as a floor only)', 'PASS')
    say('base', 'selected base', 'enc-s0' if (not czm_ok and ge) else ('czm-enc-s0' if czm_ok else 'sibag-s2'), 'frozen rule', 'enc-s0')
    fair('czm-enc-s0 and enc-s0 have the same recipe / init / data / seed / steps / release / bench protocol because czm-enc-s0 = enc-s0 + '
         'the causal_front / spk / text switches only (EX002_SFSCOPE_1108 "Causal match"), both benched with bench_ex002 per_clip seeds 0-2.')


# ------------------------------------------------------------------------------------------------- safety layer
def p99_idle(d):
    t = []
    for P in ('psync', 'offline'):
        f = Path(d) / P / 's0' / 'stats.jsonl'
        if f.exists(): t.append(float(np.percentile(np.concatenate([np.array(json.loads(l)['times']) for l in open(f)]) * 1e3, 99)))
    return t


def safety():
    header('SAFETY LAYER (EX002_CBF_1108)', '3c-1 cbf_qp, 3c-2 slack (+2 pre-emptive sub-variants), 3c-3 proj_l2, 3c-3b, 3c-3c, 3c-4, 3c-4b, 3c-5, 3c-6, 3c-7 (2 orders); '
           'perceptual projection (negatives 19); KKT backends auto / qdldl, OSQP compared')
    czm = J(B + '/ex002-sweep-1107/czm-s12000-lr1e-4-w0.1/EVAL.json'); enc = J(S + '/enc-s0/EVAL.json')

    def g3(ev, raw, fb_json, jerk=True):
        out = []
        for P in ('offline', 'psync'):
            rf = [raw[f'{P}-s{s}']['FGD_v2'] for s in range(3)]; rk = [raw[f'{P}-s{s}']['FKD'] for s in range(3)]
            lf = [ev[f'{P}-s{s}']['FGD_v2'] for s in range(3)]; lk = [ev[f'{P}-s{s}']['FKD'] for s in range(3)]
            dF, dK = np.mean(np.subtract(lf, rf)), np.mean(np.subtract(lk, rk)); sF, sK = np.std(rf, ddof=1), np.std(rk, ddof=1)
            out.append((P, dF, sF, dK, sK, abs(dF) <= sF and abs(dK) <= sK))
        jk = max(v['jerk_over_limit_frames_pct'] for v in fb_json.values())          # always shown; a gate only where registered (jerk)
        return all(o[-1] for o in out) and (jk == 0 or not jerk), out, jk

    mods = [('3c-1 cbf_qp', 'ex002-cbf', 'EVAL_cbf.json', czm, 'run', False), ('3c-3 proj_l2 B25', 'ex002-proj', 'EVAL_proj.json', czm, 'run', True),
            ('3c-3b B13', 'ex002-proj13', 'EVAL_proj13.json', czm, 'ex002-proj13-idle', True), ('3c-3c qdldl + tail 1e-3', 'ex002-proj13c', 'EVAL_proj13c.json', czm, 'ex002-proj13c-idle', True),
            ('3c-4 recursive feasibility', 'ex002-proj4-enc', 'EVAL_proj4.json', enc, None, True), ('3c-4b cut cap 8', 'ex002-proj4b-enc', 'EVAL_proj4b.json', enc, 'ex002-proj4b-idle', True),
            ('3c-5 predictive rows + carry', 'ex002-proj5-enc', 'EVAL_proj5.json', enc, 'ex002-proj5-idle', True)]
    exp = {'3c-1 cbf_qp': ('PASS (registered 3 gates) / not adopted', None), '3c-3 proj_l2 B25': ('FAIL', 'g2,g4'), '3c-3b B13': ('FAIL', 'g2,g4'),
           '3c-3c qdldl + tail 1e-3': ('FAIL', 'g2,g4'), '3c-4 recursive feasibility': ('FAIL', 'g2'), '3c-4b cut cap 8': ('FAIL', 'g4'), '3c-5 predictive rows + carry': ('FAIL', 'g4')}
    for name, d, ev, raw, idle, jerk in mods:
        ck = J(f'{R}/{d}/CHECK.json'); e = J(f'{R}/{d}/{ev}'); fb = J(f'{R}/{d}/FALLBACK.json')
        g1 = all(v['all_clear'] for v in ck.values()); fbf = [v['fallback_frames'] for v in ck.values()]; g2 = sum(fbf) == 0
        g3_, parts, jk = g3(e, raw, fb, jerk)
        if idle == 'run': p = [v['t_p99'] for v in ck.values()]; g4 = max(p) <= 100; p4 = f'p99 (as registered for this module: the run itself) {min(p):.1f}-{max(p):.1f} ms'
        elif idle: p = p99_idle(f'{R}/{idle}'); g4 = max(p) <= 100; p4 = f'idle p99 {"/".join(f"{x:.1f}" for x in p)} ms'
        else: g4 = True; p4 = 'g4 not measured idle for this module (reported: superseded by 3c-4b before timing)'
        if name.startswith('3c-1'):           # registered 3c-1 gates: clear, dFGD within 1 raw SD, p99 <= 100 (no zero-fallback / jerk gate)
            dfgd_ok = all(abs(x[1]) <= x[2] for x in parts); v = 'PASS (registered 3 gates) / not adopted' if (g1 and dfgd_ok and g4) else 'FAIL'
            say('safety', name, v, f'g1 clear {g1}; dFGD within SD {dfgd_ok}; {p4}; BUT fallback frames {min(fbf)}-{max(fbf)} per run and jerk-over-table '
                f'{jk:.0f} % -> rejected by a main-agent judgement beyond the registered gates; the later modules registered zero fallback and jerk as gates', exp[name][0]); continue
        fails = [g for g, ok in (('g1', g1), ('g2', g2), ('g3', g3_), ('g4', g4)) if not ok]
        say('safety', name, 'PASS' if not fails else 'FAIL', f'fails {",".join(fails) or "none"} | g1 clear {g1}; g2 fallback frames {fbf}; g3 dFGD/dFKD within raw SD '
            + ', '.join(f'{P} {dF:+.2f}/{sF:.2f} {dK:+.3f}/{sK:.3f}' for P, dF, sF, dK, sK, _ in parts) + f', jerk-over-table {jk:.1f} %; g4 {p4}',
            exp[name][0])
        if exp[name][1] and set(fails) != set(exp[name][1].split(',')): MISMATCH.append(f'safety/{name} failing gates {fails} vs reported {exp[name][1]}'); print('    MISMATCH failing-gate set')
    p13d = p99_idle(f'{R}/ex002-proj13d-idle')
    say('safety', '3c-3c + memo (gate 4 only)', 'PASS' if max(p13d) <= 100 else 'FAIL', f'idle p99 {"/".join(f"{x:.1f}" for x in p13d)} ms (outputs bit-identical to 3c-3c)', 'PASS')
    p6 = p99_idle(f'{R}/ex002-proj6-idle')
    say('safety', '3c-6 = 3c-5 + 4 workers (gate 4)', 'PASS' if max(p6) <= 100 else 'FAIL', f'idle p99 {"/".join(f"{x:.1f}" for x in p6)} ms (outputs bit-identical to 3c-5)', 'FAIL')
    q = J(f'{R}/ex002-3c7-quick/QUICK.json')
    for o in ('fingers_first', 'arm_first'):
        r = q[o]; k = r['wbc_violating'] or r['pos_frames'] or r['vel_frames'] or r['fallback_frames'] or not r['p99'] < 0.7 * q['none']['p99']
        say('safety', f'3c-7 (D1) {o} quick lane', 'KILL' if k else 'PASS', f'fallback {r["fallback_frames"]}, p99 {r["p99"]:.1f} vs 0.7 x 3c-4b {0.7 * q["none"]["p99"]:.1f} ms (same session)', 'KILL')
    say('safety', 'final layer', '3c-4b', 'gates 1-3 PASS with zero fallback, gate 4 FAIL (stated); optimisation stopped (coordinator)', '3c-4b')
    fair('3c-4 / 4b / 5 / 6 / 7 have the same input streams (enc-s0 bench outputs, offline / psync seeds 0-2), embodiment / limits / WBC table '
         '(release06 EX002RM, runtime-rm003), certificate (exact GJK test) and host (9950X, 1 thread unless stated) because they run through the '
         'same ex002_cbf_dev.py runner on the same files; 3c-1 .. 3c-3c ran on the voided czm pick, so they are compared with each other only.')


# ------------------------------------------------------------------------------------------------- interruption
def rule(r):
    """registered (09:1x addendum): yield3d >= 90 %, v3 <= 10 %, unsafe <= 1 %."""
    f = [n for n, ok in (('yield3d', r['yield3d_rate'] >= .90), ('v3', r['abrupt_v3'] <= .10), ('unsafe', r['unsafe_any'] <= .01)) if not ok]
    return not f, f


def interruption():
    header('INTERRUPTION (EX002_INTERRUPT_1107 + DUPLEX_LADDER)', 'bases 3 (sibag-s2, enc-s0, r3d); post methods B3 / R0causal / Funnel (+ L2 diagnostic, L1 running, '
           'L2d killed); chains raw / 3c-4b / ISC; references full 757 / half B; yield metrics 5 versions')
    for lab, f, exp_sel, exp_q in (('full 757-tail ref', X + '/eval-enc-s0/EVAL_isc.json', 'Funnel_isc', dict(R0causal_isc='qualifies', Funnel_isc='qualifies', B3_isc='fails')),
                                   ('half-B ref', X + '/eval-enc-s0-halfB/EVAL_isc.json', 'R0causal_isc', dict(R0causal_isc='qualifies', Funnel_isc='fails', B3_isc='fails'))):
        e = J(f); q = {}
        for m in ('B3_isc', 'R0causal_isc', 'Funnel_isc'):
            ok, why = rule(e[m]); q[m] = ok
            say('interruption', f'{lab}: {m}', 'qualifies' if ok else 'fails', f'yield3d {e[m]["yield3d_rate"]:.3f} v3 {e[m]["abrupt_v3"]:.3f} unsafe {e[m]["unsafe_any"]:.3f} '
                f'RKD-post {e[m]["rkd_post"]:.3f} (RKD read with the yield components){" fails " + ",".join(why) if why else ""}', exp_q[m])
        cand = [m for m in ('R0causal_isc', 'Funnel_isc') if q[m]]
        sel = min(cand, key=lambda m: e[m]['rkd_post']) if cand else 'none'
        say('interruption', f'{lab}: selection', sel, 'lower RKD-post among the qualifying post-hoc methods', exp_sel)
    f2 = J(X + '/eval-enc-s0-halfB/EVAL_f2.json')
    for m in ('B1_isc', 'B2_isc', 'FRONT_isc'):
        ok, why = rule(f2[m])
        say('interruption', f'half-B baseline {m}', 'qualifies' if ok else 'fails', f'yield3d {f2[m]["yield3d_rate"]:.3f} v3 {f2[m]["abrupt_v3"]:.3f} unsafe {f2[m]["unsafe_any"]:.3f} '
            f'RKD-post {f2[m]["rkd_post"]:.3f}{" fails " + ",".join(why) if why else ""}', {'B1_isc': 'fails', 'B2_isc': 'qualifies', 'FRONT_isc': 'fails'}[m])
    sd = J(A + '/B3_SEED_SD.json')['B1_isc']['v3']['vals']
    print(f'  reading rule (v3 is not a freeze detector): B1_isc v3 per inference seed {sd} vs the 10 % gate -> B1_isc is rejected by yield3d / RKD, not by v3.')
    print('  reading rule (no single yield gate, audit D6 / D7 / D8): yield3d is recorded INVALID on human tails; the selection above is the registered '
          'verdict and is read only beside activity / withdrawal / RKD (V4_SCORE_postchain.json, BOOT_halfB_isc_v4.json).')
    b = J(A + '/BOOT_halfB_isc.json')['R0causal_isc-Funnel_isc']
    print('  clustered paired CI (H1) R0causal_isc - Funnel_isc: ' + ', '.join(f'{k} {v["point"]:+.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]' for k, v in b.items()))
    fair('B3_isc / R0causal_isc / Funnel_isc / B1-B2-FRONT_isc have the same base (enc-s0 psync s0), decoder (vae-r3), safety chain (3c-4b + ISC, '
         'safety_int_ex002.py --isc, prefix = projected U) and human reference because every row is built from the same int_psync_s0 continuations '
         '(or, for the baselines, the same uninterrupted psync s0) and scored by one eval_interrupt_ex002.py call per reference.')


# ------------------------------------------------------------------------------------------------- metric validity
def metric_validity():
    header('YIELD METRIC VALIDITY (audit D6-D8, M1)', 'yield-height, yield3d, v4 it1, v4 it2, activity-only, PSS it0 / it1')
    a1, a2 = J(A + '/V4_AUDIT_it1.json'), J(A + '/V4_AUDIT_it2.json')
    y3 = a2['tails_B']['yield3d_all']
    say('metrics', 'yield3d on half-B human tails', 'INVALID' if y3 < .85 else 'valid', f'human pass {y3:.3f} (needs >= 0.85)', 'INVALID')
    for lab, a in (('v4 it1', a1), ('v4 it2', a2)):
        key = 'v4_needed' if lab == 'v4 it2' else 'v4'
        hb = a['tails_B'][key]; b1 = a['B1']['v4']
        ok = hb >= .85 and a['GT_speech']['v4'] <= .15 and b1 <= .15
        say('metrics', lab, 'valid' if ok else 'INVALID', f'half-B {key} {hb:.3f} (>= 0.85), GT speech {a["GT_speech"]["v4"]:.3f}, B1 {b1:.3f} (<= 0.15)', 'INVALID')
    fr = J(A + '/V4_FRESH_it2.json')['activity']; act_ok = a2['tails_A']['activity'] >= .85 and a2['tails_B']['activity'] >= .85 and fr >= .85 and a2['GT_speech']['activity'] <= .15
    say('metrics', 'activity-only yield (D8)', 'valid' if act_ok else 'INVALID', f'A {a2["tails_A"]["activity"]:.3f} B {a2["tails_B"]["activity"]:.3f} fresh {fr:.3f} (>= 0.85); '
        f'GT speech {a2["GT_speech"]["activity"]:.3f} (<= 0.15)', 'INVALID')
    p = J(A + '/m1/PSS_VALID_it1.json')
    say('metrics', 'PSS (M1) it1, last iteration', 'valid' if p['gates']['all'] else 'INVALID', f'half-B {p["tails_B"]["pass_rate"]:.3f}, fresh {p["fresh"]["pass_rate"]:.3f} '
        f'(>= 0.85), GT speech {p["GT_speech"]["pass_rate"]:.3f} (<= 0.15), mismatched ref {p["mismatched_ref_B"]["delta_pp"]:+.1f} pp', 'INVALID')


# ------------------------------------------------------------------------------------------------- design round 11-08
def design_round():
    header('DESIGN ROUND 11-08 (FABLE_DESIGN_1108; A)', 'D1 (2 orders), D2 (poly / affine / agnostic), D3 (l2d + controls a-d), M1 (2 iterations)')
    q = J(A + '/d2-quick/QUICK.json')['dev']
    kill = q['poly']['rt_l1'] > q['r3']['rt_l1'] or q['poly']['rt_max_over_min'] > 1.15
    say('design', 'D2 polyphase equaliser quick lane', 'KILL' if kill else 'PASS', f'round-trip L1 {q["poly"]["rt_l1"]:.4f} vs r3 {q["r3"]["rt_l1"]:.4f}; '
        f'phase max/min {q["poly"]["rt_max_over_min"]:.3f} (kill > 1.15; GT {q["GT"]["max_over_min"]:.3f})', 'KILL')
    e = J(A + '/l2d-quick41/EVAL.json')['l2d']; d = J(A + '/l2d-quick41/DRIFT.json')['l2d']
    kill = e['rkd_post'] >= .518 or e['abrupt_v3'] > .10
    say('design', 'D3 L2d quick lane (41 events)', 'KILL' if kill else 'PASS', f'RKD-post {e["rkd_post"]:.3f} (kill >= 0.518), v3 {e["abrupt_v3"]:.3f}; yield3d {e["yield3d_rate"]:.3f}, '
        f'drift median / p95 / max {d["median"]:.3f} / {d["p95"]:.3f} / {d["max"]:.3f} m', 'KILL')
    print('  (D1 and M1: see SAFETY 3c-7 and METRIC VALIDITY)')
    fair('l2d and its controls (a)-(d), R0causal raw and B3 raw have the same base (enc-s0 psync s0 continuations), decoder and raw chain because '
         'all rows post-process the same int_psync_s0 files of the same 41 interruptions and are scored by one eval_interrupt call (half-B reference).')


# ------------------------------------------------------------------------------------------------- control (module 5b confirmation)
def control():
    header('CONTROL (EX002_CTL; only module 5b confirmation encoded here)', 'lanes 1-4, modules 2, 5, 5b, 5c, port, 6 (not run), 7 / 8, 7b / 8b, 9, 9b, 9c, 9c\'\', 9d, 10')
    F = CT + '/m5b-confirm/fresh'
    try:
        g, sh, fo = J(F + '/EVAL_given.json'), J(F + '/EVAL_shuffle.json'), J(F + '/FGDOUT_sparse.json'); ex = J(F + '/EXEC_dense.json')
    except FileNotFoundError as err:
        say('control', 'module 5b confirmation', 'NO DATA', str(err)); return
    s = (3, 4, 5); m = lambda d, k, f: float(np.mean([d[k.format(x)][f] for x in s]))
    gain = m(g, 'm5b_given_s{}', 'stg_g') - m(g, 'none_s{}', 'stg_g')
    follow = m(sh, 'm5b_shuffle_s{}', 'stg_g') > m(sh, 'm5b_shuffle_s{}', 'stg_o')
    say('control', '5b gate 1 STG gain (forearm) + shuffle', 'PASS' if gain >= .10 and follow else 'FAIL',
        f'gain {gain:+.3f} (>= +0.10); shuffle follows {follow} ({m(sh, "m5b_shuffle_s{}", "stg_g"):.3f} vs original {m(sh, "m5b_shuffle_s{}", "stg_o"):.3f})', 'PASS')
    fh = m(g, 'm5bfo_given_s{}', 'fh_g'); ffol = m(sh, 'm5bfo_shuffle_s{}', 'fh_g') > m(sh, 'm5bfo_shuffle_s{}', 'fh_o')
    say('control', '5b gate 2 finger hit (override) + shuffle', 'PASS' if fh >= .9 and ffol else 'FAIL', f'hit {fh:.3f} (>= 0.9); shuffle follows {ffol}', 'PASS')
    none_r1 = [fo[f'none_s{x}']['R1']['FGD_v2'] for x in s]; none_r0 = [fo[f'none_s{x}']['R0']['FGD_v2'] for x in s]
    C1, C0 = float(np.mean(none_r1)), float(np.mean(none_r0)); d1 = max(.03 * C1, 2 * np.std(none_r1, ddof=1)); d0 = max(.03 * C0, 2 * np.std(none_r0, ddof=1))
    i_ = float(np.mean([fo[f'm5bsp_given_s{x}']['R1']['FGD_v2'] for x in s])); ii = float(np.mean([fo[f'm5bspfo_given_s{x}']['R0']['FGD_v2'] for x in s]))
    say('control', '5b gate (i) sparse outside +-1 s', 'PASS' if i_ <= C1 + d1 else 'FAIL', f'{i_:.2f} <= C + delta = {C1:.2f} + {d1:.2f}', 'PASS')
    say('control', '5b gate (ii) sparse whole clip (+ override)', 'PASS' if ii <= C0 + d0 else 'FAIL', f'{ii:.2f} <= C + delta = {C0:.2f} + {d0:.2f}', 'PASS')
    exo = all(v['exec_after_cbf'] == 1.0 for v in ex.values())
    say('control', '5b executability after CBF', 'PASS' if exo else 'FAIL', 'all runs 100 % of clips', 'PASS')
    fair('m5b given / shuffle / none have the same model (lane-4 ctl4 on sibag-s2), plans (fresh 1108), seeds 3-5 and scorer because they are '
         'blends / generations of one DLC job (dlci2zmkrgtpnpye) scored by one eval call.')
    print('  NOT ENCODED (owner B, exp/ctl-port; verdicts as reported in EX002_CTL.md): lanes 1-4, 5, 5c, port, 7 / 8, 7b / 8b, 9, 9b, 9c, 9c\'\', 9d, 10.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--json', default=None); a = ap.parse_args()
    for f in (base_selection, safety, interruption, metric_validity, design_round, control):
        try: f()
        except Exception as err:                                     # a missing / changed result file is reported, not hidden
            print(f'  ERROR in {f.__name__}: {type(err).__name__}: {err}'); MISMATCH.append(f'{f.__name__}: {err}')
    print('\nBEAT_EX002', 'ALL VERDICTS REPRODUCED' if not MISMATCH else f'MISMATCH {MISMATCH}')
    if a.json: Path(a.json).write_text(json.dumps(dict(tables=OUT, mismatches=MISMATCH), indent=1))
    sys.exit(1 if MISMATCH else 0)
