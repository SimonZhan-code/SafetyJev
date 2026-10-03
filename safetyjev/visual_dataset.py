"""Map-style video dataset and batches for a multimodal Noul training adapter."""
from array import array
from collections import Counter, OrderedDict
import json
import math
import os
from pathlib import Path

import numpy as np

from .openjev_data import visual_model_payload


class VideoWindowDecoder:
    def __init__(self, cache_windows=16):
        if type(cache_windows) is not int or cache_windows < 0:
            raise ValueError("cache_windows must be a nonnegative integer")
        self.cache_windows = cache_windows
        self.cache = OrderedDict()

    def decode(self, path, frame_indices, expected_fps):
        indices = tuple(frame_indices)
        if not indices or any(type(i) is not int or i < 0 for i in indices) or any(
            a >= b for a, b in zip(indices, indices[1:])
        ):
            raise ValueError("Frame indices must be nonnegative and strictly increasing")
        key = (str(Path(path).resolve()), indices, expected_fps)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        import av
        frames = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 1
            fps = float(stream.average_rate)
            if not math.isfinite(expected_fps) or not math.isclose(fps, expected_fps, abs_tol=1e-6):
                raise ValueError("Video rate does not match the indexed frame clock")
            start_pts = stream.start_time or 0
            seek_pts = start_pts + math.floor(indices[0] / fps / float(stream.time_base))
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            wanted = set(indices)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Video frame lacks a timestamp")
                position = float((frame.pts - start_pts) * stream.time_base) * fps
                index = round(position)
                if abs(position - index) > .01:
                    raise ValueError("Video timestamps are not aligned to the indexed constant frame clock")
                if index in wanted:
                    frames[index] = frame.to_ndarray(format="rgb24")
                if index >= indices[-1]:
                    break
        if set(frames) != set(indices):
            raise ValueError("Requested observation frames are missing: " + str(path))
        result = np.stack([frames[i] for i in indices])
        if self.cache_windows:
            self.cache[key] = result.copy()
            while len(self.cache) > self.cache_windows:
                self.cache.popitem(last=False)
        return result


class APWindowDataset:
    """Torch-compatible map-style dataset with a compact JSONL byte-offset index."""
    def __init__(self, package, split, cache_windows=16, frame_cache=None):
        self.package = Path(package).resolve()
        if (self.package / "BUILDING").exists():
            raise ValueError("Dataset package build has not finished")
        self.summary = json.loads((self.package / "dataset_metadata.json").read_text())
        if split not in ("train", "calibration", "validation", "test", "ood"):
            raise ValueError("Unknown dataset split")
        self.split = split
        self.path = self.package / (split + ".jsonl")
        self.offsets, self.strata = array("Q"), array("I")
        self.frame_cache = None
        if frame_cache is not None:
            from .visual_cache import FrameCache
            self.frame_cache = FrameCache(frame_cache, self.package)
            for offset, code in self.frame_cache.connection().execute(
                    'SELECT offset,stratum FROM samples WHERE split=? ORDER BY idx', (split,)):
                self.offsets.append(offset); self.strata.append(code)
            self.frame_cache.close()
            from .visual_cache import sample_index_digest
            digest=sample_index_digest((idx,offset,code) for idx,(offset,code) in enumerate(zip(self.offsets,self.strata)))
            if digest!=self.frame_cache.metadata['sample_index_sha256'].get(split):raise ValueError('Cached sample index checksum differs')
            if len(self.offsets)!=self.frame_cache.metadata['splits'].get(split):raise ValueError('Cached sample count differs')
            if not self.offsets: raise ValueError("Selected split has no cached samples")
            self.decoder = VideoWindowDecoder(cache_windows=cache_windows)
            return
        strata_ids = {}
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") != "noul" or row.get("options") != ["no", "yes"] or row.get("target") not in ([1, 0], [0, 1]):
                    raise ValueError("Expected a hard-label Noul training record")
                if row.get("split", split) != split:
                    raise ValueError("Record belongs to another split")
                answer = int(row["target"][1])
                key = (row["metadata"]["family"], row["metadata"]["query_id"], answer)
                code = strata_ids.setdefault(key, len(strata_ids))
                self.offsets.append(offset)
                self.strata.append(code)
        if not self.offsets:
            raise ValueError("Selected split has no samples")
        self.decoder = VideoWindowDecoder(cache_windows=cache_windows)

    def close(self):
        self._reader_pid=None
        reader=getattr(self,'_reader',None)
        if reader is not None:reader.close();self._reader=None
        cache=getattr(self,'frame_cache',None)
        if cache is not None:cache.close()
    def __del__(self):
        self.close()
    def __len__(self):
        return len(self.offsets)

    def record(self, index):
        if getattr(self,"_reader_pid",None)!=os.getpid():
            if getattr(self,"_reader",None) is not None:self._reader.close()
            self._reader=self.path.open("rb");self._reader_pid=os.getpid()
        self._reader.seek(self.offsets[index])
        return json.loads(self._reader.readline())

    def __getstate__(self):
        state=self.__dict__.copy();state['_reader']=None;state['_reader_pid']=None
        return state

    def media_path(self, record, camera):
        media = record["state"]["observation_window"]
        resource = self.summary["resources"][media["resource_id"]]
        root = (self.package / resource["raw_root"]).resolve()
        path = (root / media["videos"][camera]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Media path escapes its registered raw dataset")
        return path

    def __getitem__(self, index):
        row = self.record(index)
        media = row["state"]["observation_window"]
        if self.frame_cache is None:
            images = {camera: self.decoder.decode(self.media_path(row, camera), media["frame_indices"], media["video_fps"])
                      for camera in media["videos"]}
        else:
            from .visual_cache import frame_key
            images = {camera: np.stack([self.frame_cache.read(frame_key(media["resource_id"], video, frame, media["video_fps"]))
                                      for frame in media["frame_indices"]]) for camera, video in media["videos"].items()}
        inputs = visual_model_payload(row, images)
        return {"inputs": inputs, "target": np.asarray(row["target"], dtype=np.float32), "sample_id": row["id"],
                "query_id": row["metadata"]["query_id"]}

    def training_weights(self, mode="uniform"):
        if self.split != "train":
            raise ValueError("Evaluation splits must retain their natural distribution")
        if mode == "uniform":
            return np.ones(len(self), dtype=np.float64)
        if mode != "query_answer":
            raise ValueError("Balance must be uniform or query_answer")
        counts = Counter(self.strata)
        return np.asarray([1.0 / counts[code] for code in self.strata], dtype=np.float64)


def collate_numpy(items):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    cameras = set(items[0]["inputs"]["observations"])
    if any(set(item["inputs"]["observations"]) != cameras for item in items):
        raise ValueError("All items in a batch need the same camera streams")
    observations = {}
    for camera in sorted(cameras):
        images = [item["inputs"]["observations"][camera] for item in items]
        if any(x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8 or x.shape != images[0].shape for x in images):
            raise ValueError("Expected matching T,H,W,3 uint8 RGB histories")
        observations[camera] = np.stack(images)
    targets = np.stack([item["target"] for item in items])
    if targets.shape != (len(items), 2):
        raise ValueError("Expected two-class Noul targets")
    return {"inputs": {"questions": [item["inputs"]["question"] for item in items], "observations": observations},
            "targets": targets, "sample_ids": [item["sample_id"] for item in items],
            "query_ids": [item.get("query_id", "unspecified") for item in items]}


def collate_torch(items):
    import torch
    batch = collate_numpy(items)
    batch["inputs"]["observations"] = {
        camera: torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()
        for camera, images in batch["inputs"]["observations"].items()
    }
    batch["targets"] = torch.from_numpy(batch["targets"])
    return batch


def make_dataloader(dataset, *, batch_size, num_workers=0, balance="uniform", seed=42):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    if dataset.split != "train" and balance != "uniform":
        raise ValueError("Do not rebalance an evaluation split")
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if balance != "uniform":
        sampler = WeightedRandomSampler(dataset.training_weights(balance), len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_torch,
                      shuffle=dataset.split == "train" and sampler is None, sampler=sampler,
                      generator=generator, persistent_workers=num_workers > 0)


class PreparedVisualCollator:
    """Spawn-safe worker-local processor; metadata never enters the model input."""
    def __init__(self, model_config):
        self.config=dict(model_config)
        self.processor=None

    def __call__(self, items):
        from transformers import AutoProcessor
        from jev.visual_model import encode_visual_inputs
        if self.processor is None:
            import os
            os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
            self.processor=AutoProcessor.from_pretrained(self.config['model_id'],revision=self.config['revision'],
                min_pixels=self.config['min_pixels'],max_pixels=self.config['max_pixels'],local_files_only=True)
            self.processor.tokenizer.padding_side='right'
            if self.processor.tokenizer.pad_token_id is None:self.processor.tokenizer.pad_token=self.processor.tokenizer.eos_token
        batch=collate_torch(items)
        encoded=encode_visual_inputs(self.processor,batch['inputs']['questions'],batch['inputs']['observations'],
                                     ('overview','wrist'),self.config['max_length'])
        batch['inputs']={'encoded':encoded}
        return batch
