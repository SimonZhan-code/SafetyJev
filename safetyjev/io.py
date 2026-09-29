import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
