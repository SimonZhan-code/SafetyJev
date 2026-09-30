"""CPU end-to-end validation of packaged Noul records, video batches and loss shapes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from safetyjev.visual_dataset import APWindowDataset, make_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.package)
    expected = json.loads((root / "dataset_metadata.json").read_text())
    groups = json.loads((root / "group_splits.json").read_text())
    seen_ids, seen_groups, windows = set(), {}, 0
    shapes = {"python": sys.version.split()[0], "torch": torch.__version__, "splits": {}}
    for split, count in expected["splits"].items():
        started = time.perf_counter()
        data = APWindowDataset(root, split)
        assert len(data) == count
        file_hash = hashlib.sha256()
        with data.path.open("rb") as stream:
            for line in stream:
                file_hash.update(line)
                row = json.loads(line)
                assert row["id"] not in seen_ids
                seen_ids.add(row["id"])
                group = row["group_id"]
                assert groups[group] == split
                assert group not in seen_groups or seen_groups[group] == split
                seen_groups[group] = split
                media = row["state"]["observation_window"]
                assert media["frame_indices"][-1] == row["metadata"]["query_step"]
                assert len(media["frame_indices"]) == expected["window"]["history_frames"]
                assert set(media["videos"]) == {"overview", "wrist"}
                assert row["options"] == ["no", "yes"]
                windows += 1
        assert file_hash.hexdigest() == expected["file_sha256"][split]
        loader = make_dataloader(data, batch_size=args.batch_size, num_workers=args.num_workers, seed=42,
                                 balance="query_answer" if split == "train" else "uniform")
        batch = next(iter(loader))
        assert set(batch["inputs"]) == {"questions", "observations"}
        assert torch.all(batch["targets"].sum(dim=-1) == 1)
        for camera, images in batch["inputs"]["observations"].items():
            assert images.dtype == torch.uint8 and images.ndim == 5
            assert images.shape[:3] == (args.batch_size, expected["window"]["history_frames"], 3)
        # Check the shared-scalar Noul loss interface, not a trained model.
        scalar = torch.zeros(len(batch["targets"]), requires_grad=True)
        logits = torch.stack([torch.zeros_like(scalar), scalar], dim=-1)
        loss = -(batch["targets"] * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(scalar.grad).all()
        shapes["splits"][split] = {"samples": len(data), "batch_size": len(batch["targets"]),
                                   "cameras": {c: list(v.shape) for c, v in batch["inputs"]["observations"].items()},
                                   "target_shape": list(batch["targets"].shape),
                                   "num_workers": args.num_workers, "scalar_loss_backward": True,
                                   "elapsed_seconds": round(time.perf_counter()-started, 2)}
        del loader
    shapes.update(samples_verified=windows, unique_sample_ids=len(seen_ids),
                  groups_without_leakage=len(seen_groups), trained_model_tested=False,
                  single_class_queries={key: values for key, values in expected["by_query"].items()
                                        if not values.get("0", 0) or not values.get("1", 0)})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(shapes, indent=2)+"\n")
    print(json.dumps(shapes, indent=2))


if __name__ == "__main__":
    main()
