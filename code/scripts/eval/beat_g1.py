"""11-08 G1 win rules in code (counterpart of beat_ex002.py). Registered criteria: refine-logs/EXPERIMENT_PLAN.md blocks B1 / B2 / B4
(B3 G1 interruption: dropped by the user's decision 11-08, DUPLEX_1107.md; B5 G1 control: owner B, not encoded). Reads the stored
JSONs (FINAL_TABLE_G1_1107.json, BOOT_MAIN_1107.json, FINAL_TABLE_G1_ABL_1108.json, BOOT_ABL.json when present), prints PASS / FAIL /
N-A with the reason, the design count per table and a fairness sentence per comparison, and checks the verdicts already reported.
Reading rules: FGD (official encoder) is the primary column and is read with the paired recording-level bootstrap, never a point
estimate alone; executability = ours after CBF-QP, baselines raw (registered convention, stated); test data were read once per final
method after TEST_USAGE entries (abltest-1108/TEST_USAGE.jsonl, final3-1106/TEST_USAGE.jsonl).
    python scripts/eval/beat_g1.py [--boot-abl BOOT_ABL.json] [--json OUT]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
C = '/mnt/cpfs/zbl-cpfs-new/CKPT/zehan/gesture-scmg/causal-train-1105'; RUNS = str(Path.home() / 'bulk/causal-train-1105/runs')
OUT, MISMATCH = {}, []


def say(table, item, verdict, reason, expected=None):
    ok = expected is None or expected == verdict
    tag = '' if expected is None else (' [reported: %s]' % expected if ok else ' MISMATCH (reported: %s)' % expected)
    print(f'  {item:44s} {verdict:6s} {reason}{tag}'); OUT.setdefault(table, {})[item] = dict(verdict=verdict, reason=reason, expected=expected, match=ok)
    if not ok: MISMATCH.append(f'{table}/{item}')


def rows(path):
    return {r['label']: r for r in json.load(open(path))}


def main_table():
    print('\n== B1 MAIN TABLE (G1 test, val_common 349 recordings)  (designs: final recipe czm + 6000 steps + sf0 + w_safe 300 + CBF-QP; '
          'baselines EMAGE / GestureLSM / SemTalk + GMR, ECHO-G official 5 seeds, ECHO-G stream_prefix)')
    pairs = json.load(open(C + '/BOOT_MAIN_1107.json'))['pairs']; key = ['SCMG_psync - ECHOG_offline']
    p = pairs[key[0]]; d, lo, hi = p['diff'], p['lo'], p['hi']
    say('g1_main', 'psync FGD < ECHO-G official, CI excludes 0', 'PASS' if hi < 0 else 'FAIL', f'{key[0]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (2000 recording draws)', 'PASS')
    t = rows(C + '/FINAL_TABLE_G1_1107.json')
    ex = [r['exec_cbf_clean'] for lab, row in t.items() if lab.startswith('SCMG') for r in row['runs']]
    say('g1_main', 'executability after CBF-QP (ours)', 'PASS' if min(ex) == 1.0 else 'FAIL', f'{len(ex)} runs, min clean share {min(ex):.4f}', 'PASS')
    print('  fairness: SCMG and ECHO-G official are scored on the same 349 val_common recordings with the same official FGD encoder, the same '
          'leg handling (train-split mean legs for every method) and the same paired bootstrap, because all predictions go through one '
          'bench.sh / boot_main_g1.py call; ECHO-G is offline (not causal) and its executability is raw (no safety layer), stated in the table.')


def streaming():
    print('\n== B2 STREAMING ISOLATION (training match, test)  (designs: czmws vs ncws; offline / psync; chunk / look-ahead sweep is dev only)')
    t = rows(RUNS + '/FINAL_TABLE_G1_ABL_1108.json')
    def fgd(lab): return float(np.mean([r['FGD'] for r in t[lab]['runs']]))
    gz = fgd('SCMG psync (final recipe)') - fgd('SCMG offline (final recipe)'); gn = fgd('no causal training match (ncws) psync') - fgd('no causal training match (ncws) offline')
    if gn <= 0: say('g1_stream', 'training match halves the psync-offline gap', 'N/A', f'no gap to halve on test: ncws psync - offline {gn:+.3f}, czmws {gz:+.3f} (psync not worse in either)')
    else: say('g1_stream', 'training match halves the psync-offline gap', 'PASS' if gz <= 0.5 * gn else 'FAIL', f'gap ncws {gn:+.3f} -> czmws {gz:+.3f}')
    print('  fairness: czmws and ncws have the same data, init, steps, sf0, w_safe and test protocol because ncws is the same recipe with the '
          'causal training switches off (abltest-1108 lanes), scored by the same final_table_g1.py call.')


def ablation(boot):
    print('\n== MAIN-TABLE ABLATIONS vs czmws (test)  (designs: ncws, training-level no text / no audio, inference-level no text / no audio; offline + psync)')
    if not boot or not Path(boot).exists():
        print('  paired bootstrap vs czmws not available yet (BOOT_ABL.json) -> no verdicts printed'); return
    b = json.load(open(boot))['pairs']
    for name, p in b.items():
        if 'SCMG' not in name: continue
        d, lo, hi = p['diff'], p['lo'], p['hi']
        v = 'worse' if lo > 0 else ('better' if hi < 0 else 'n.s.')
        say('g1_ablation', name[:44], v, f'FGD diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}] (positive = the ablation is worse than czmws)')
    print('  fairness: every ablation row shares the czmws recipe except the one switch named in its label, uses the same 3 training seeds x '
          'per_clip inference seeds 0 / 1 and the same 349 recordings, because all rows were generated by abltest-1108 / final3-1106 with one '
          'causal_g1.sh test entry point and scored by one boot_main_g1.py call.')


def not_encoded():
    print('\n== NOT JUDGED HERE')
    print('  B3 G1 interruption test: dropped (user decision 11-08, DUPLEX_1107.md); B5 G1 control: owner B; B4 FGD change after CBF: '
          'not in these JSONs (the main table FGD is of the raw generation).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--boot-abl', default=None); ap.add_argument('--json', default=None); a = ap.parse_args()
    for f in (main_table, streaming, lambda: ablation(a.boot_abl), not_encoded):
        try: f()
        except Exception as err: print(f'  ERROR: {type(err).__name__}: {err}'); MISMATCH.append(str(err))
    print('\nBEAT_G1', 'ALL VERDICTS REPRODUCED' if not MISMATCH else f'MISMATCH {MISMATCH}')
    if a.json: Path(a.json).write_text(json.dumps(dict(tables=OUT, mismatches=MISMATCH), indent=1))
    sys.exit(1 if MISMATCH else 0)
