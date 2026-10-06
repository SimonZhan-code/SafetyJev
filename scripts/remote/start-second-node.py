import json,os,subprocess,time
from pathlib import Path
setup=Path('/workspace/bootstrap-copy');root=Path('/workspace/SafetyJev-ood-20261006')
def run(cmd,**kwargs):
 print('RUN',cmd,flush=True);return subprocess.run(cmd,check=True,**kwargs)
while not (setup/'complete.json').exists():
 state=subprocess.run(['supervisorctl','status','safetyjev-runtime-copy'],capture_output=True,text=True).stdout
 if any(x in state for x in ['FATAL','EXITED','STOPPED']):raise RuntimeError('Runtime transfer failed: '+state)
 time.sleep(20)
run(['tar','-xzf',str(setup/'final-payload.tar.gz'),'-C',str(root)])
(root/'artifacts').mkdir(exist_ok=True);Path('/workspace/SafetyJev/artifacts').mkdir(exist_ok=True)
resource=root/'docs/results/2026-10-06-id-ood-queue/domain-sweep-resources.json'
(root/'artifacts/domain-sweep-resources.json').write_bytes(resource.read_bytes())
env=dict(os.environ,PYTHONPATH=str(root)+':/workspace/ManiGuard',SAFETYJEV_MANIGUARD_ROOT='/workspace/ManiGuard',OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_HEADLESS='1',OMNIGIBSON_DATA_PATH='/workspace/ManiGuard/behavior-1k/datasets')
run(['/workspace/conda/behavior51/bin/python','-m','unittest','discover','-s','tests','-q'],cwd=root,env=env)
for python in ['/workspace/conda/behavior51/bin/python','/workspace/openpi/.venv/bin/python','/workspace/Open-Jev-visual-pinned/.venv/bin/python']:
 run([python,'-c','import torch; x=torch.ones((16,16),device="cuda"); assert (x@x).sum().item()==4096; print(torch.__version__,torch.cuda.get_device_name(0))'],env=env)
vulkan=subprocess.run(['vulkaninfo','--summary'],capture_output=True,text=True)
(setup/'vulkan-summary.txt').write_text(vulkan.stdout+vulkan.stderr)
if vulkan.returncode or 'NVIDIA RTX PRO 6000' not in vulkan.stdout:raise RuntimeError('Vulkan device unavailable')
run(['supervisorctl','reread']);run(['supervisorctl','update'])
run(['supervisorctl','start','safetyjev-domain-queue'])
(setup/'worker-started.json').write_text(json.dumps({'status':'worker_queue_started','worker':'node-b','unix_s':time.time()},indent=2))
