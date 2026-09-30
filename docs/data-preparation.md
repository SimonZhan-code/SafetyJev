# Dataset interface

The data ZIP supplies the videos, AP traces, trajectory annotations and sample
indices. Extract it in the repository root, then read `datasets/README.md` for
counts, label semantics and the full data layout.

## Load the supplied samples

```python
from safetyjev.visual_dataset import APWindowDataset, make_dataloader

data = APWindowDataset("datasets/packages/five_family", "train")
loader = make_dataloader(data, batch_size=8, num_workers=2,
                         balance="uniform", seed=42)
batch = next(iter(loader))
```

`batch["inputs"]` contains natural-language questions and current overview/wrist
images. Each camera tensor is `(B, 1, 3, H, W)`, uint8 RGB. `batch["targets"]` is
`(B, 2)`, float32, in `[No, Yes]` order. The model processor performs image
preprocessing and tokenization. No simulator is required to load these data.

The loader resolves videos through each source's relative `raw_root` in
`dataset_metadata.json`. It decodes the requested frame indices from the MP4s;
source paths in provenance metadata are not used to locate images.

## Five question definitions, one exporter

Each family has a complete definition file:

- `jar_ap_queries.json`: five questions.
- `lid_ap_queries.json`: four questions.
- `stack_ap_queries.json`: four questions.
- `cabinet_ap_queries.json`: three questions.
- `dusty_ap_queries.json`: two questions.

Editable definitions are in `configs/data/`. The data package includes the exact
versions used to build its samples under `datasets/definitions/`.
Each question declares its AP (or `all_of` conjunction), Yes polarity and wording.
Jar/Lid use common wording across their scenes. Stack/Cabinet/Dusty also provide
`question_by_scene` to name the actual monitored objects. The same exporter
handles all five configurations.

`target=[1,0]` means No; `[0,1]` means Yes. Yes answers the specific question and
is not universally an unsafe label. AP truth is taken at the current frame;
an earlier cumulative monitor rejection does not change later current-state
labels. Different questions on the same image are independent binary examples.

## Record format

Each Noul JSONL row contains:

- `id`, `group_id`, `split`, `source`: sample identity and grouping.
- `question`: natural-language model input.
- `state.observation_window`: source resource, relative video paths, frame indices
  and video/control clock metadata.
- `kind="noul"`, `options=["no", "yes"]`, `target`: supervision format.
- `metadata`: AP definition/value, source episode, policy, family and condition.

Only the question and decoded camera images enter the model. Metadata remains
available for analysis. Video frame indices correspond to trace steps; playback
FPS and robot control frequency are distinct fields.

Trajectory annotations retain `trajectory_id`, episode safety, media references
and violation events. Observed state intervals have inclusive endpoints;
`trajectory_end` indicates no observed recovery before recording stopped.
Temporal first-rejection events remain separate from current-state interval
labels. The question configuration selects which current AP checks enter training.

## Adjust sampling or splits

`configs/data/five_family_delivery.json` defines the source datasets, GT files,
question definitions, sample stride and provisional train/validation/test split.
To change wording, point its `queries` entries at the edited JSON definitions.
To change the data recipe, edit the relevant sampling/split fields and build a
new output directory:

```bash
python -m safetyjev.cli build-ap-mixture \
  --recipe configs/data/five_family_delivery.json \
  --output datasets/packages/my_training_package
```

The current model uses one frame per camera. Split assignment groups each base
task across all policies, seeds and ID/OOD conditions. Keep each group together
when revising the train/validation/test protocol. Group allocations and sample
counts are saved in `group_splits.json` and `dataset_metadata.json`.

`balance="query_answer"` optionally samples training rows by inverse
family/question/answer frequency; `uniform` shuffles all training rows.
Validation and test retain their stored distributions.

## Check a received or rebuilt package

```bash
python tools/check_data_pipeline.py \
  --package datasets/packages/five_family \
  --output outputs/data-check.json
```

This checks sample counts/hashes, grouping, real video decoding, batch shapes and
No/Yes targets. Use the training configuration to choose validation frequency,
checkpoint selection and final reporting for the experiment.
