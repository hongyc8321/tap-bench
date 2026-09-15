"""Detached single-run supervisor; no timeout and no automatic retraining."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    config=json.loads((root/'authorized.json').read_text())
    status=root/'process_status.json'
    def save(value):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(status)
    state=dict(status='starting',supervisor_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),timeout_seconds=None)
    save(state)
    env=os.environ.copy();env.update(config['environment']);env['PYTHONUNBUFFERED']='1'
    try:
        with (root/'train.log').open('x') as log:
            process=subprocess.Popen(config['command'],env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root)
            state.update(status='running',training_pid=process.pid);save(state)
            code=process.wait()
        state.update(status='exited',exit_code=code,ended_at=datetime.now(timezone.utc).isoformat());save(state)
    except Exception as error:
        state.update(status='supervisor_failed',error=repr(error));save(state);raise


if __name__=='__main__':main()
