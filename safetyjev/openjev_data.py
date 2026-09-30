"""Noul records aligned with Zefan-Cai/Open-Jev, plus an explicit visual adapter."""
import copy

from .data_preparation import model_payload

UPSTREAM_REVISION = "3308a15ccd7eea1df7a37d6ddc39b023b801ba16"
FAST_BACKEND_REVISION = "c52b8bb958c1f0d241d4eb7fce4ecd8d885bf1e4"


def to_openjev_record(sample, *, split, source, provenance):
    """Use the upstream ten-field schema; video decoding requires our visual path."""
    if split not in ("train", "calibration", "validation", "test", "ood"):
        raise ValueError("Assign an upstream dataset split explicitly")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Specify the dataset source")
    for key in ("input_sha256", "source_url", "license", "split_policy"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError("Missing import provenance: " + key)
    answer = sample["target"]["answer"]
    if type(answer) is not int or answer not in (0, 1):
        raise ValueError("Binary supervision must be 0=no or 1=yes")
    metadata = copy.deepcopy(sample["metadata"])
    metadata["provenance"] = {**provenance, "type": "import", "original_id": sample["sample_id"]}
    metadata["input_adapter"] = "safetyjev_visual_observations_v1"
    return {"id": sample["sample_id"], "group_id": metadata["split_group"], "split": split,
            "source": source, "state": {"observation_window": copy.deepcopy(sample["media"])},
            "question": sample["input"]["question"], "kind": "noul", "options": ["no", "yes"],
            "target": [float(1 - answer), float(answer)], "metadata": metadata}


def visual_model_payload(record, decoded_observations):
    """Translate media references to decoded observations, never textify filenames."""
    if record.get("kind") != "noul" or record.get("options") != ["no", "yes"]:
        raise ValueError("Expected an Open-Jev Noul record")
    sample = {"input": {"question": record["question"]},
              "media": record["state"]["observation_window"]}
    return model_payload(sample, decoded_observations)
