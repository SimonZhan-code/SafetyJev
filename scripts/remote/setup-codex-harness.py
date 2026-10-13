"""Install only the restricted CPU harness; never start simulator/model jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MODULES = ('__init__.py', 'codex_worker.py', 'codex_tools.py', 'codex_bridge.py',
           'codex_inventory.py', 'codex_dispatch_audit.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk-python', default='/usr/bin/python3',
                        help='Python executable readable by the dedicated account')
    args=parser.parse_args()
    if os.getuid()!=0: raise SystemExit('Run setup as root on the prepared NVIDIA node')
    user='safetyjev-codex'; home=Path('/home')/user
    try: account=pwd.getpwnam(user)
    except KeyError:
        subprocess.run(['useradd','--create-home','--shell','/usr/sbin/nologin',user],check=True)
        account=pwd.getpwnam(user)
    if account.pw_uid==0:raise SystemExit('Dedicated account must not be root')
    for path in (home,home/'.codex',home/'work'):
        path.mkdir(parents=True,exist_ok=True);os.chown(path,account.pw_uid,account.pw_gid);path.chmod(0o700)
    if any((home/'work').iterdir()):raise SystemExit('Harness work directory must be empty')
    for path in (Path('/opt/safetyjev-codex'),Path('/run/safetyjev-codex')):
        path.mkdir(parents=True,exist_ok=True);os.chown(path,0,account.pw_gid);path.chmod(0o750)
    runtime=Path('/opt/safetyjev-codex/safetyjev');runtime.mkdir(exist_ok=True)
    os.chown(runtime,0,account.pw_gid);runtime.chmod(0o750)
    hashes={}
    for name in MODULES:
        source=ROOT/'safetyjev'/name;target=runtime/name
        shutil.copyfile(source,target);os.chown(target,0,account.pw_gid);target.chmod(0o640)
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    # A named-user denial preserves access for the existing root-owned sweep.
    for path in ('/workspace/SafetyJev', '/workspace/logs'):
        subprocess.run(['setfacl', '-m', 'u:safetyjev-codex:---', path], check=True)
    python='/workspace/codex-harness-env/bin/python'
    if not Path(python).exists():
        subprocess.run(['uv','venv','--python',args.sdk_python,'/workspace/codex-harness-env'],
                       check=True,umask=0o022)
    subprocess.run(['uv','pip','install','--python',python,'openai-codex==0.162.1',
                    'openai-codex-cli-bin==0.162.1','zstandard==0.25.0'],check=True,umask=0o022)
    # Only SDK authentication store belongs to this account; no HF/OpenRouter secrets.
    env={'PATH':'/usr/bin:/bin','HOME':str(home),'CODEX_HOME':str(home/'.codex'),
         'PYTHONPATH':'/opt/safetyjev-codex','PYTHONNOUSERSITE':'1','LANG':'C.UTF-8'}
    proc=subprocess.run([python,'-m','safetyjev.codex_worker','--preflight'],env=env,
        cwd=home/'work',user=account.pw_uid,group=account.pw_gid,extra_groups=[],
        capture_output=True,text=True,timeout=100,umask=0o077)
    try:report=json.loads(proc.stdout)
    except ValueError:report={'status':'setup_failed'}
    report.update(source_sha256=hashes,sdk_version='0.162.1',runtime_version='0.162.1',
                  scope='CPU/auth/tool inventory only; no robot or GPU inference',exit_code=proc.returncode)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');args.output.chmod(0o600)
    print(json.dumps(report))
    raise SystemExit(proc.returncode)


if __name__=='__main__':main()
