"""Compile the pinned Jar policy before capture without consuming its episode RNG."""
import argparse
import json
from pathlib import Path
import time

# ManiGuard masks rollout seeds to 31 bits, so this warm-up seed cannot collide.
WARMUP_SEED = 1 << 31


def validate_metadata(metadata):
    if metadata.get('serve_config') != 'pi05-base_datagen_v1_jar_joint_2cam_lora':
        raise ValueError('Unexpected policy config')
    if metadata.get('checkpoint') != '/workspace/checkpoints/pi05-jar/7400':
        raise ValueError('Unexpected policy checkpoint')


def warm_policy(scene):
    import numpy as np
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect

    prompt = json.loads((Path(scene)/'diagnostics.jsonl').read_text().splitlines()[0])['prompt']
    observation = {
        'observation/state':np.array([0.,-.4,0.,-2.,0.,1.6,.8,.04],dtype=np.float32),
        'observation/image_left':np.zeros((256,256,3),dtype=np.uint8),
        'observation/wrist_image':np.zeros((256,256,3),dtype=np.uint8),
        'prompt':prompt, 'episode_seed':WARMUP_SEED}
    started = time.monotonic()
    # Cold JAX compilation blocks the server event loop longer than the default
    # 20-second pong deadline. Bound warm-up separately from the episode itself.
    with connect('ws://127.0.0.1:8000',compression=None,max_size=None,
                 ping_timeout=300,open_timeout=10) as socket:
        metadata = msgpack_numpy.unpackb(socket.recv(timeout=30))
        validate_metadata(metadata)
        socket.send(msgpack_numpy.Packer().pack(observation))
        reply = socket.recv(timeout=300)
        if isinstance(reply,str):
            raise RuntimeError('Policy warm-up failed')
        result = msgpack_numpy.unpackb(reply)
        actions = np.asarray(result['actions'])
        if actions.shape != (16,8) or not np.isfinite(actions).all():
            raise ValueError('Invalid warm-up actions')
    return {'status':'ready','metadata':metadata,'warmup_seed':WARMUP_SEED,
            'seconds':time.monotonic()-started,
            'note':'No simulator actions; the real episode seed resets the policy sampler.'}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',required=True)
    args = parser.parse_args()
    print(json.dumps(warm_policy(args.scene)),flush=True)
