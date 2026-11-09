"""Leak check of the semantic line (11-06 rule 3; cf. check_gest_leak.py). Exit 0 = PASS, 1 = FAIL. Checks:
 1 speaker split  : R (recogniser) and K (retrieval bank / synthesis) disjoint; neither has a test speaker (1_wayne,
                    3_solomon, 21_ayana) or a dev speaker (2_scott, 6_carla)
 2 recogniser data: every TRAIN event of the recogniser data dir is an R speaker; no test / dev speaker trains
 3 control tokens : CTL.json train-split tokens only on the view's train_ids, dev tokens only on val_ids, none on the
                    ECHO-G test items (<view>/test/train.json); no dev / test speaker among train tokens
 4 retrieval bank : (--bank BANK.json: [{stem, spk, recording}]) every entry a K speaker, a train_ids item, not test / dev
 5 retrieval log  : (--retrieval-log: [{query, retrieved}] stems) never the query's own recording
--selftest: injects one leak per check into copies and requires each to FAIL.
    python scripts/semantic/check_sem_leak.py --split SPLIT_RK.json --rec-data <dir> --ctl <CTL dir> --view <hos view> [--bank B] [--retrieval-log L]
"""
import argparse, copy, json, re, sys
from pathlib import Path

TEST_SPK = {'1_wayne', '3_solomon', '21_ayana'}; DEV_SPK = {'2_scott', '6_carla'}
spk_of = lambda stem: '_'.join(stem.split('_')[1:3])
rec_of = lambda stem: re.sub(r'_utt_\d+$', '', stem)


def checks(S, E, C, items, train_ids, val_ids, test_ids, bank, rlog):
    errs = []
    R, K = set(S['R']), set(S['K'])
    if R & K: errs.append(f'1: R and K share {sorted(R & K)}')
    for n, X in (('R', R), ('K', K)):
        if X & (TEST_SPK | DEV_SPK): errs.append(f'1: {n} contains test/dev speakers {sorted(X & (TEST_SPK | DEV_SPK))}')
    bad = {e['spk'] for e in E['events'] if e['split'] == 'train' and e['spk'] not in R}
    if bad: errs.append(f'2: recogniser train events of non-R speakers {sorted(bad)[:5]}')
    for s, ts in C.items():
        sp = {t['split'] for t in ts}
        if s in test_ids: errs.append(f'3: tokens on test item {s}'); continue
        if 'train' in sp and s not in train_ids: errs.append(f'3: train tokens on non-train item {s}')
        if 'dev' in sp and s not in val_ids: errs.append(f'3: dev tokens on non-val item {s}')
        if 'train' in sp and (items.get(s, {}).get('speaker') in TEST_SPK | DEV_SPK): errs.append(f'3: train tokens of test/dev speaker {s}')
    for b in bank or []:
        if b['spk'] not in K: errs.append(f'4: bank entry {b["stem"]} speaker {b["spk"]} not in K')
        if b['stem'] not in train_ids: errs.append(f'4: bank entry {b["stem"]} not a train item')
        if b['spk'] in TEST_SPK | DEV_SPK: errs.append(f'4: bank entry {b["stem"]} test/dev speaker')
    for r in rlog or []:
        if rec_of(r['query']) == rec_of(r['retrieved']): errs.append(f'5: {r["query"]} retrieved its own recording')
    return errs


def main():
    ap = argparse.ArgumentParser()
    for k in ('split', 'rec_data', 'ctl', 'view'): ap.add_argument('--' + k.replace('_', '-'), required=True)
    ap.add_argument('--bank'); ap.add_argument('--retrieval-log'); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
    S = json.loads(Path(a.split).read_text()); E = json.loads((Path(a.rec_data) / 'EVENTS.json').read_text())
    C = json.loads((Path(a.ctl) / 'CTL.json').read_text()); v = Path(a.view)
    sp = json.loads((v / 'SPLIT.json').read_text()); train_ids, val_ids = set(sp['train_ids']), set(sp['val_ids'])
    items = {it['id']: it for it in json.loads((v / 'train.json').read_text())}
    test_ids = {x['id'] for x in json.loads((v / 'test' / 'train.json').read_text())}
    bank = json.loads(Path(a.bank).read_text()) if a.bank else None
    rlog = json.loads(Path(a.retrieval_log).read_text()) if a.retrieval_log else None
    errs = checks(S, E, C, items, train_ids, val_ids, test_ids, bank, rlog)
    rep = dict(n_R=len(S['R']), n_K=len(S['K']), rec_train_events=sum(e['split'] == 'train' for e in E['events']),
               ctl_items=len(C), bank=len(bank) if bank else 0, retrievals=len(rlog) if rlog else 0, errors=errs[:20], n_errors=len(errs))
    ok = not errs
    if a.selftest:                                  # every injected leak must be caught
        st = {}
        S1 = copy.deepcopy(S); S1['K'] = S1['K'] + ['1_wayne']; st['K_has_test_speaker'] = bool(checks(S1, E, C, items, train_ids, val_ids, test_ids, bank, rlog))
        E1 = copy.deepcopy(E); e0 = next(e for e in E1['events'] if e['split'] == 'train'); e0['spk'] = S['K'][0]
        st['rec_trains_on_K'] = bool(checks(S, E1, C, items, train_ids, val_ids, test_ids, bank, rlog))
        C1 = copy.deepcopy(C); C1[next(iter(test_ids))] = [dict(type=1, t=0, split='train')]
        st['token_on_test_item'] = bool(checks(S, E, C1, items, train_ids, val_ids, test_ids, bank, rlog))
        B1 = (bank or []) + [dict(stem=next(iter(val_ids)), spk='2_scott', recording='x')]
        st['bank_has_dev'] = bool(checks(S, E, C, items, train_ids, val_ids, test_ids, B1, rlog))
        L1 = (rlog or []) + [dict(query='english_5_stewart_0_1_1_utt_0001', retrieved='english_5_stewart_0_1_1_utt_0007')]
        st['own_recording'] = bool(checks(S, E, C, items, train_ids, val_ids, test_ids, bank, L1))
        rep['selftest'] = st; ok = ok and all(st.values())
    print(json.dumps(rep, indent=1)); print('LEAK_CHECK', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
