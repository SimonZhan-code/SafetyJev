"""Trusted, CPU-only admission check. Never forward process errors to the model."""
import json
import os
from pathlib import Path
import pwd
import subprocess
from .codex_worker import validate_attestation


def preflight(config):
    account=pwd.getpwnam(config['runtime_user'])
    if account.pw_uid==0:raise ValueError('Dedicated harness user required')
    home=Path(config['runtime_home'])
    if any((home/'work').iterdir()):raise ValueError('Harness cwd must be empty')
    env={'PATH':'/usr/bin:/bin','HOME':str(home),'CODEX_HOME':str(home/'.codex'),
         'PYTHONPATH':config['runtime_root'],'PYTHONNOUSERSITE':'1','LANG':'C.UTF-8'}
    proc=subprocess.run([config['runtime_python'],'-m','safetyjev.codex_worker','--preflight',
        '--model',config['model'],'--reasoning-effort',config['reasoning_effort'],
        '--tool-sequence',config['tool_sequence'],'--decision-timeout',str(config['decision_timeout_s'])],
        cwd=home/'work',env=env,user=account.pw_uid,group=account.pw_gid,extra_groups=[],
        capture_output=True,text=True,timeout=100,umask=0o077)
    try:report=json.loads(proc.stdout)
    except ValueError:raise RuntimeError('Codex preflight failed') from None
    if proc.returncode or report.get('status')!='passed':
        raise RuntimeError('Codex tool-isolation/authentication preflight did not pass')
    validate_attestation(report['attestation'], config['tool_sequence'])
    return report
