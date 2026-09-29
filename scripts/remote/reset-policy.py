"""Warm the policy and ensure the next benchmark episode seed triggers reseeding."""
import json
from pathlib import Path
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
policy=WebsocketClientPolicy(host='127.0.0.1',port=8000)
meta=policy.get_server_metadata()
assert meta['serve_config']=='pi05-base_datagen_v1_jar_joint_2cam_lora',meta
assert meta['checkpoint']=='/workspace/checkpoints/pi05-jar/7400',meta
path=Path('/workspace/data/maniguard-bench/jar_transport/task_0000/base/diagnostics.jsonl')
prompt=json.loads(path.read_text().splitlines()[0])['prompt']
obs={'observation/state':np.array([0.,-.4,0.,-2.,0.,1.6,.8,.04],dtype=np.float32),'observation/image_left':np.zeros((256,256,3),dtype=np.uint8),'observation/wrist_image':np.zeros((256,256,3),dtype=np.uint8),'prompt':prompt,'episode_seed':0}
a=np.asarray(policy.infer(obs)['actions'])
assert a.shape==(16,8) and np.isfinite(a).all()
print('policy_warmup_seed_0_complete',flush=True)
