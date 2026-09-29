import json
import time
from pathlib import Path
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
metadata = client.get_server_metadata()
assert metadata["serve_config"] == "pi05-base_datagen_v1_jar_joint_2cam_lora", metadata
assert metadata["checkpoint"] == "/workspace/checkpoints/pi05-jar/7400", metadata
rows=[]
for index in range(3):
    observation = {
        "observation/state": np.array([0., -0.4, 0., -2., 0., 1.6, 0.8, 0.04],dtype=np.float32),
        "observation/image_left": np.zeros((256,256,3),dtype=np.uint8),
        "observation/wrist_image": np.zeros((256,256,3),dtype=np.uint8),
        "prompt": "Transport the jar to the target while keeping it upright.",
        "episode_seed": 0,
    }
    start=time.perf_counter()
    response=client.infer(observation)
    elapsed=time.perf_counter()-start
    actions=np.asarray(response["actions"])
    assert actions.shape==(16,8), actions.shape
    assert np.isfinite(actions).all()
    rows.append({"request":index,"latency_s":elapsed,"action_shape":list(actions.shape),"all_finite":True})
    print(json.dumps(rows[-1]),flush=True)
result={"test":"synthetic_policy_inference","synthetic_inputs":True,"simulator_executed":False,"safety_accuracy_measured":False,"server_metadata":metadata,"requests":rows}
Path("/workspace/SafetyJev/artifacts/synthetic-policy-smoke.json").write_text(json.dumps(result,indent=2)+"\n")
