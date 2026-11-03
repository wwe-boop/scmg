"""Summarise actual result files only; never invent missing metrics."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[]
    for lane in ('bagger-null','bagger-reverse'):
        ledger=a.root/lane/'timing.jsonl'
        timings=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
        for r in range(1,4):
            root=a.root/lane/f'round-{r}'
            for protocol in ('default','x2'):
                f=root/f'rollout-{protocol}.jsonl'
                if not f.exists(): continue
                x=json.loads(f.read_text());free,oracle=x['free'],x['oracle']
                passed=(free['still']<=.08 and .9<=free['speed']<=1.1 and .9<=free['copy']<=1.1 and oracle['speed']<=1.3)
                rows.append(dict(lane=lane,round=r,protocol=protocol,free=free,oracle=oracle,passed=passed,
                    seconds=sum(t['seconds'] for t in timings if t['stage'] in (f'collect-{r}',f'train-{r}',f'eval-{r}-default',f'eval-{r}-x2')),
                    result=str(f)))
    out=dict(rows=rows,complete=len(rows)==12,historical_control=dict(still=.15,speed=.71,copy=.80),
             note='Historical control from task, not a fresh paired run. Default/x2 use unchanged rollout_probe protocol.')
    tmp=Path(str(a.out)+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(out,indent=2));tmp.replace(a.out);print(json.dumps(out),flush=True)


if __name__=='__main__': main()
