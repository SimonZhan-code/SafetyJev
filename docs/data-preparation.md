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

## Semantic safety labels from source recordings

The optional semantic mode derives supervision offline from ManiGuard source HDF5
recordings. It leaves the source monitor GT and existing AP datasets unchanged.
In this mode **Yes means the queried safety condition is violated**. Each example
asks about one resolved object scope, so simultaneous violations produce separate
positive examples.

Use `configs/data/semantic_safety.json` and an explicit task-group split. The six
reviewed rules cover tilt strictly above 15 degrees, object-origin height strictly
more than 2 cm below the support surface, direct food contact, an open Jar off
support, an uncovered Lid-task container off support, and endpoint liquid net
loss of at least 20%. Object scopes and evidence requirements are in the config.
Engineering checks may use a separate fixture manifest; the final collection's
frozen manifest is required when preparing the formal dataset, not to implement
or test this pipeline.

From the repository root, build the judge package first and reuse its annotations
for the classifier. Use new output directories and the same immutable sources,
definitions, split and episode-selection arguments for both commands:

```bash
python -m safetyjev.predictor_judge_commands build \
  --episodes datasets/predictor_judge/raw \
  --output /path/to/semantic-judge \
  --split-manifest /path/to/task-group-splits.json \
  --semantic-definitions configs/data/semantic_safety.json \
  --maniguard-root /path/to/ManiGuard \
  --liquid-asset-root /path/to/licensed-datasets \
  --semantic-task predictor_judge --history-frames 3

python -m safetyjev.predictor_judge_commands build \
  --episodes datasets/predictor_judge/raw \
  --output /path/to/semantic-classifier \
  --split-manifest /path/to/task-group-splits.json \
  --semantic-definitions configs/data/semantic_safety.json \
  --maniguard-root /path/to/ManiGuard \
  --semantic-task classifier \
  --reuse-annotations /path/to/semantic-judge/annotations
```

Both use the existing loaders, cache tools and training configurations;
set `data.package` and the corresponding `data.frame_cache` to the new outputs.
The command does not overwrite an existing package. `--review` permits a marked
development build from explicitly selected candidate parameters. A build without
`--review` requires every selected definition to be reviewed.

When preparing the other model from the same sources and definitions, add
`--reuse-annotations /path/to/first-package/annotations`. This skips repeated
physical-snapshot extraction and copies the verified annotation files into the
new package, so either package can be moved independently with its raw tree.
The reuse check hashes metadata, definitions and annotation implementation, and
checks HDF5 sizes and modification times. It assumes finalized raw files are
immutable; it is not a full HDF5 content checksum. Rebuild annotations after
changing raw files or copying them without preserving timestamps. Media decoding
and validation still run for each package.

The package contains `annotations/`, the exact semantic definitions and their
hashes, derived source indices, split JSONL files and composition reports. Images
remain in the original HDF5: classifier rows use
`state.observation_window.image_refs` instead of MP4 paths. Keep the package and
its relatively referenced raw tree together when moving them. The classifier
uses one current frame per camera; the predictor judge uses adjacent history,
current robot state and the known remaining committed actions with a padding mask.
Neither receives physical labeling evidence, source AP truth or future images.

State supervision can recover from Yes to No. Predictor state labels refer to
`(start_step, end_step]` under the recorded actions; replaced actions and truncated
unobserved tails cannot supply negative labels. Interval liquid-loss supervision
requires valid container occupancy counts at the specified times. Unavailable
labels go to `excluded.jsonl`; current-frame classifier inputs are not paired
with interval liquid-loss questions. Liquid labels compare occupancy at the window
start and end: `(start_count - end_count) / start_count >= 0.20`. The starting
count must be positive. A temporary loss that recovers by the endpoint is not a
positive net-loss label. The question stays qualitative; counts and thresholds
are supervision metadata, not model inputs.

For fresh liquid annotations, use the source-preparation environment with SciPy,
cryptography and importable USD (`pxr`). Set `--liquid-asset-root /path/to/datasets`
or `OMNIGIBSON_DATA_PATH` to the locally licensed source assets. The builder checks
the encrypted asset against its recorded identity, resolves rigid fillable mesh
volumes, and counts recorded particles offline without launching a simulator.
Geometry is reused across episodes and asset/geometry hashes accompany annotations.
Unvalidated articulated or ambiguous geometry stops preparation for inspection;
missing per-step evidence yields excluded labels, never invented negative labels.
Assets and decryption keys are not included in packages. Reusing verified
annotations and running model training need neither assets nor USD.

Episode selection uses the new annotations before applying the training mixture.
`legacy_safety` retains the old result, and `semantic_coverage` reports missing
supervision. A positive is sufficient to establish unsafe; incomplete evidence
without a positive does not establish a fully safe episode. Validation/test keep
their frozen groups and are not balanced by outcome.

### Inspect, cache and check training

Before scaling up, inspect `episode_inventory.jsonl`, `DATA_SUMMARY.md` and
`composition.json` on a bounded complete-source package. `semantic_windows`
reconciles each split/family/semantic combination: unique contributing episodes,
positive, negative, eligible and excluded windows, and exclusion reasons.
`candidates = eligible + excluded`; `eligible = positive + negative`. Episodes
can contribute to multiple semantics, so do not sum these episode counts as a
corpus total. Unselected or invalid episodes remain in the episode inventory.
The first-positive summaries are not counts of independent physical events.

Create and verify disposable caches with the source-reading environment (including
HDF5 and the public ManiGuard reader on `PYTHONPATH`). Set a measured disk budget
with `--max-gib` before a large cache build. Both caches can substantially exceed
compressed raw media size; they need not both be retained permanently.

```bash
export PYTHONPATH="$PWD/third_party/Open-Jev:$PWD:/path/to/ManiGuard${PYTHONPATH:+:$PYTHONPATH}"
python tools/prepare_predictor_judge_cache.py \
  --package /path/to/semantic-judge --output /path/to/semantic-judge-cache
python tools/prepare_predictor_judge_cache.py \
  --package /path/to/semantic-judge --output /path/to/semantic-judge-cache --verify-only
python tools/prepare_visual_cache.py \
  --package /path/to/semantic-classifier --output /path/to/semantic-classifier-cache
python tools/prepare_visual_cache.py \
  --package /path/to/semantic-classifier --output /path/to/semantic-classifier-cache --verify-only
```

Cache verification checks integrity and package compatibility, not semantic
correctness or complete real-data coverage. `tools/check_data_pipeline.py` is for
the legacy AP/video layout and is not the validator for these source packages.

For a bounded model check, copy `configs/training/five_family_visual_smoke.json`
and `configs/training/predictor_judge_smoke.json` to separate semantic configs.
Set `data.package` and `data.frame_cache` to the corresponding outputs, and set
`evaluation.run_test=false`. Use the existing model environment and available
checkpoint; model downloads and full training are separate resource decisions.

```bash
bash scripts/train_visual.sh \
  --config /path/to/semantic-classifier-smoke.json \
  --output /path/to/new-classifier-smoke-run --device cuda:0
bash scripts/train_predictor_judge.sh \
  --config /path/to/semantic-judge-smoke.json \
  --output /path/to/new-judge-smoke-run --device cuda:0
```

These runs exercise loss/backpropagation and capped validation, not final model
quality. Normalization uses selected training records only. The decision threshold
is fixed at 0.5 by default; automatic validation-based threshold selection is not
implemented. Test evaluation stays separate. See `classifier-training.md`,
`predictor-judge-training.md` and `evaluation-protocol.md` for checkpoint/resume
and evaluation contracts. Review the final inventory, provenance, composition
and resource budget before a formal batch build.


#When distinct monitored scopes contain the same object category, questions use
static task wording to identify the target versus obstacles or the stack. Target
exclusion is derived from resolved object sets, not from an AP name alone: scopes
can overlap. The recorded task instruction may accompany such a query for object
identification; object IDs, AP truth, measured poses and future evidence remain
outside model inputs. Missing or unsupported ambiguous grounding fails explicitly
rather than producing a broader question with a narrower label. Changing query
construction invalidates annotation provenance; regenerate derived annotations
and indexes from the same read-only raw data before building the new release.

## Fixed-cohort chunk build

For an explicitly delivered source cohort, select the immutable manifest rather
than scanning a live collection directory. The JSON is a list with `episode_id`,
`source_node`, absolute `source_path`, `source_metadata_sha256`, `group_id` and
`split` for each source. Build one node-local shard per source node. The builder
verifies metadata identity, rejects duplicate sources or inconsistent split input,
and records provenance in the package. It never includes newly collected episodes
outside the list.

```bash
python -m safetyjev.predictor_judge_commands build \
  --source-manifest /path/to/source_manifest.json --source-node NODE_ID \
  --split-manifest /path/to/explicit-group-splits.json \
  --semantic-definitions configs/data/semantic_safety.json \
  --maniguard-root /path/to/ManiGuard --liquid-asset-root /path/to/licensed-datasets \
  --input-contract chunk_start_v2 --train-episodes all \
  --output /path/to/node-shard/predictor_judge

python -m safetyjev.predictor_judge_commands build \
  --source-manifest /path/to/source_manifest.json --source-node NODE_ID \
  --split-manifest /path/to/explicit-group-splits.json \
  --semantic-definitions configs/data/semantic_safety.json \
  --maniguard-root /path/to/ManiGuard --semantic-task classifier \
  --train-episodes all --reuse-annotations /path/to/node-shard/predictor_judge/annotations \
  --output /path/to/node-shard/classifier
```

The v2 judge includes `history_robot_states` with shape `(8,16)`: each state is
read from the same post-action observation as its paired images. The history
action mask also masks historical states. Current state remains explicit, and
all states use training-only normalization. Rebuild old v1 chunk indexes/caches;
GT annotations and original media can be reused unchanged.

Annotation reuse hardlinks immutable files on the same filesystem; cross-filesystem
reuse copies the small annotations. Each package still owns a portable directory
entry. Images remain in original HDF5. `query_windows` adds split/family/query
counts, exclusion reasons and starting-state breakdowns to `composition.json`.
Retain split-provenance discrepancies beside the manifest and describe the actual
experimental split status; renaming a split does not erase development exposure.

Each shard records train-only `normalization_statistics` (sums, squared sums and
counts). `safetyjev.source_manifest.combine_training_normalization` combines these
with count weighting; never average per-shard means/stds or fit held-out data.
Before training over multiple shards, use one globally fitted train normalization
and retain its cohort provenance. Node-local raw paths are not remotely readable
from another machine: actual training requires an explicit data-access layout.
A catalog of shards alone does not establish cross-node DataLoader throughput.
Do not generate large caches before measuring their cost on the intended host.
