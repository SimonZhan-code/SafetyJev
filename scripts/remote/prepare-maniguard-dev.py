"""Prepare the validated dev snapshot without modifying an existing checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from safetyjev.maniguard import DEV_MANIFEST, SOURCE_PROFILE, verify_sources, source_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('/workspace/ManiGuard-dev'))
    args = parser.parse_args()
    os.umask(0o077)
    if SOURCE_PROFILE != 'dev':
        raise ValueError('Preparation requires the dev profile')
    remote = DEV_MANIFEST['repository']
    latest = subprocess.check_output(['git', 'ls-remote', remote, 'refs/heads/dev'], text=True).split()[0]
    if latest != DEV_MANIFEST['commit']:
        raise RuntimeError('Upstream dev advanced to '+latest+'; review and validate the adapter before starting a new run')
    if not args.destination.exists():
        subprocess.run(['git', 'clone', '--single-branch', '--branch', 'dev', remote, str(args.destination)], check=True)
    # Never reset/pull an existing checkout, including an active historical run.
    verify_sources(args.destination)
    print(json.dumps({'status': 'prepared', 'upstream_dev_at_preparation': latest,
                      **source_provenance()}, indent=2))


if __name__ == '__main__':
    main()
