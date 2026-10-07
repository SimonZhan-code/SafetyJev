"""Install a policy launch reservation at a completed episode boundary.

Only the orchestration parent pauses; its active capture child finishes normally.
The policy service is reconfigured after that child exits, then the parent resumes.
Active hashed evaluation files and completed cases are unchanged.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path('/workspace/SafetyJev-ood-20261006')
STATUS=ROOT/'artifacts/clutter-handoff-status.json'


def save(value):
    p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(STATUS)


def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''


def main():
    if os.environ.get('CONTAINER_ID')!='54533396':raise RuntimeError('Wrong coordinator instance')
    pid=int(subprocess.check_output(['supervisorctl','pid','safetyjev-base-sweep'],text=True).strip())
    if 'scripts/remote/base-task-sweep.py' not in command(pid):raise RuntimeError('Unexpected evaluator process')
    save({'status':'waiting_for_active_cabinet_capture','parent_pid':pid})
    deadline=time.monotonic()+300
    while True:
        progress=json.loads(Path('/workspace/SafetyJev/artifacts/base-sweep-20261006/progress.json').read_text())
        if progress.get('current',{}).get('family')!='cabinet':raise RuntimeError('Cabinet boundary no longer available')
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        captures=[int(c) for c in children if 'safetyjev.visual_runtime capture' in command(c)]
        if len(captures)==1:break
        if time.monotonic()>deadline:raise RuntimeError('No capture boundary found')
        time.sleep(.2)
    capture=captures[0]
    os.kill(pid,signal.SIGSTOP)
    try:
        save({'status':'capture_finishing_parent_paused','parent_pid':pid,'capture_pid':capture,'scene':progress['current']['scene']})
        deadline=time.monotonic()+7200
        while True:
            path=Path(f'/proc/{capture}/status')
            if not path.exists() or any(line.startswith('State:') and '\tZ' in line for line in path.read_text().splitlines()):break
            if time.monotonic()>deadline:raise RuntimeError('Capture did not exit in time')
            time.sleep(1)
        # No capture client is using the policy now; the parent cannot launch the next one.
        subprocess.run(['supervisorctl','stop','safetyjev-sweep-policy'],check=True)
        config=Path('/etc/supervisor/conf.d/safetyjev-recovery.conf')
        old=config.read_text();old_command='/workspace/SafetyJev/scripts/remote/serve-sweep-policy.py'
        if old_command not in old:raise RuntimeError('Expected service command missing')
        config.with_suffix('.before-clutter-reservation').write_text(old)
        new=old.replace(old_command,str(ROOT/'scripts/remote/serve-reserved-base-policy.py'))
        config.write_text(new)
        subprocess.run(['supervisorctl','reread'],check=True)
        subprocess.run(['supervisorctl','update','safetyjev-sweep-policy'],check=True)
        subprocess.run(['supervisorctl','start','safetyjev-sweep-policy'],check=True)
        save({'status':'clutter_reserved_on_new_node','coordinator_instance':'54533396','clutter_instance':'54566989','parent_pid':pid,'unix_s':time.time()})
    finally:
        os.kill(pid,signal.SIGCONT)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save({'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise
