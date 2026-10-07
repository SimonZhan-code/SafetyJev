"""Enforce the recovery family reservation before launching a VLA service."""
import json
import os
from pathlib import Path

ROOT=Path('/workspace/SafetyJev')
RESERVATION=Path('/workspace/SafetyJev-ood-20261006/artifacts/base-worker-reservation.json')


def authorize_family(family,instance_id,reservation):
    if reservation.get('schema')!=1 or reservation.get('families',{}).get(family)!=str(instance_id):
        raise ValueError('Family is reserved for another instance: '+family)


def main():
    family=json.loads((ROOT/'artifacts/active-sweep-policy.json').read_text())['family']
    authorize_family(family,os.environ.get('CONTAINER_ID'),json.loads(RESERVATION.read_text()))
    python='/workspace/conda/behavior51/bin/python'
    os.execv(python,[python,'-u',str(ROOT/'scripts/remote/serve-sweep-policy.py')])


if __name__=='__main__':main()
