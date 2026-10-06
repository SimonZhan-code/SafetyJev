"""Verify the selected family policy and reset its sampling stream with seed 0."""
import argparse
import json
from pathlib import Path
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
parser=argparse.ArgumentParser();parser.add_argument('--family',required=True);args=parser.parse_args()
spec=json.loads((Path(__file__).resolve().parents[2]/'configs/domain-sweep-policies.json').read_text())['families'][args.family]
policy=WebsocketClientPolicy(host='127.0.0.1',port=8001)
meta=policy.get_server_metadata()
assert meta['serve_config']==f'pi05-base_datagen_v1_{args.family}_joint_2cam_lora',meta
assert meta['checkpoint']==f'/workspace/checkpoints/pi05-{args.family}/'+spec['step'],meta
p=Path('/workspace/data/maniguard-bench')/spec['pipeline']/'task_0000/base/diagnostics.jsonl'
prompt=json.loads(p.read_text().splitlines()[0])['prompt']
obs={'observation/state':np.array([0.,-.4,0.,-2.,0.,1.6,.8,.04],dtype=np.float32),'observation/image_left':np.zeros((256,256,3),dtype=np.uint8),'observation/wrist_image':np.zeros((256,256,3),dtype=np.uint8),'prompt':prompt,'episode_seed':0}
a=np.asarray(policy.infer(obs)['actions'])
assert a.shape==(16,8) and np.isfinite(a).all()
print(json.dumps({'status':'ready','family':args.family,'metadata':meta}),flush=True)
