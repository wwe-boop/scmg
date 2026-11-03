"""Run one stage with an append-only timing/exit-code ledger."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--ledger',type=Path,required=True);p.add_argument('--stage',required=True)
p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
start=time.monotonic(); stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(cmd)
with a.ledger.open('a') as f:
    f.write(json.dumps(dict(stage=a.stage,started=stamp,seconds=time.monotonic()-start,returncode=r.returncode,command=cmd))+'\n')
raise SystemExit(r.returncode)
