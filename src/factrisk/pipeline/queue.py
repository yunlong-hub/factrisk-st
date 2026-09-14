"""Dependency-aware single-GPU lanes; no multi-node or distributed training."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from factrisk.core.io import write_json


def wait_for(path, hours=12):
    start=time.monotonic()
    while not Path(path).is_file():
        if time.monotonic()-start>hours*3600:
            raise TimeoutError(f'Dependency did not finish: {path}')
        time.sleep(30)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--role',choices=['direct','evidence'],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--after-status');p.add_argument('--status',required=True)
    a=p.parse_args();base=dict(pid=os.getpid(),role=a.role,config=a.config,shard=a.shard,shards=a.shards)
    write_json(a.status,dict(**base,status='waiting'))
    try:
        wait_for(a.config)
        if a.after_status:
            started=time.monotonic()
            while True:
                wait_for(a.after_status)
                status=json.loads(Path(a.after_status).read_text())
                if status['status']=='failed': raise RuntimeError('Upstream inference failed')
                if status['status']=='complete': break
                if time.monotonic()-started>12*3600: raise TimeoutError('Upstream inference timeout')
                time.sleep(30)
        command=[sys.executable,'-u','-m','factrisk.backends.runner','--config',a.config,
                 '--role',a.role,'--shard',str(a.shard),'--shards',str(a.shards)]
        write_json(a.status,dict(**base,status='running',command=command))
        subprocess.run(command,check=True,timeout=24*3600)
        write_json(a.status,dict(**base,status='complete'))
    except BaseException as exc:
        write_json(a.status,dict(**base,status='failed',error=repr(exc)));raise


if __name__=='__main__': main()
