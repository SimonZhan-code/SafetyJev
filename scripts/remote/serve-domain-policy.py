"""Supervisor entrypoint; read the sweep-selected pinned policy identity."""
import json,os
from pathlib import Path
root=Path(__file__).resolve().parents[2]
f=json.loads((root/'artifacts/active-domain-policy.json').read_text())['family']
spec=json.loads((root/'configs/domain-sweep-policies.json').read_text())['families'][f]
os.environ.update(PYTHONPATH='/workspace/ManiGuard',XLA_PYTHON_CLIENT_PREALLOCATE='false',CUDA_VISIBLE_DEVICES='0')
os.chdir('/workspace/ManiGuard')
python='/workspace/openpi/.venv/bin/python'
os.execv(python,[python,'-u','-m','maniguard.serve.openpi_native','--config',f'pi05-base_datagen_v1_{f}_joint_2cam_lora','--checkpoint',f'/workspace/checkpoints/pi05-{f}/'+spec['step'],'--host','127.0.0.1','--port','8001'])
