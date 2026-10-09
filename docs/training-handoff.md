# Training handoff

This is the entry point for training SafetyJev on the Unsafe600 release. It links
the current architecture, data contract and execution steps; the two model guides
maintain the runnable commands.

## Release and scope

| Component | Training baseline |
|---|---|
| Code | `feat/data-preparation`; record the exact published commit containing this workflow and its data-loader implementation |
| Dependency | Open-Jev at the commit recorded by the checkout's submodule |
| Dataset | [IDEAS-Lab-Northwestern/SafetyJev-unsafe600](https://huggingface.co/datasets/IDEAS-Lab-Northwestern/SafetyJev-unsafe600) |
| Dataset revision | `598da866544f0ea218853e3b0d8fe64c2b000ab5` |
| Split | 480/60/60 episodes, partitioned by complete task group |
| Initial experiment | Predictor Judge with the pinned Qwen3.8-27B configuration |
| Subsequent experiments | Qwen3.5-9B backbone comparison and the independent Classifier pipeline |

The dataset contains prepared indices, labels, train-only normalization and shared
training images. Downloading the separate raw backup or repeating semantic data
construction is unnecessary. Both loaders support direct JPEG reading with
`data.frame_cache=null`; decoded caches are optional.

## Execution sequence

1. **Read the architecture.** The [README model table](../README.md#model-pipelines)
   describes both pipelines. The Predictor Judge's
   [chunk-boundary contract](predictor-judge-training.md#chunk-boundary-semantic-judge)
   specifies history actions, post-action observations and robot states, current
   state, next eight committed actions, queries and masks. Future outcomes and
   annotation evidence are supervision only.
2. **Prepare the host and data.** Follow the README's
   [code setup](../README.md#get-the-code),
   [environment](../README.md#training-environment) and
   [fixed dataset download](../README.md#download-the-current-training-dataset).
   Record code, submodule, model and dataset revisions along with GPU resources.
   The dataset occupies approximately 302.81 GB; weights, checkpoints and optional
   caches require additional space. Authentication remains on the training host.
3. **Validate the training interface.** Follow the selected guide's Unsafe600
   workflow: [Predictor Judge](predictor-judge-training.md#unsafe600-start-to-finish)
   or [Classifier](classifier-training.md#unsafe600-start-to-finish). Each provides
   configuration generation and a separate four-step pilot, stopped after step two
   and resumed from its checkpoint. Verify forward/backward, finite loss,
   save/resume, model loading, validation and actual device memory use. DDP requires
   the model to fit on each device.
4. **Run the experiment.** Use a separate formal output directory. Select and
   record the training budget, global batch, evaluation/save intervals and seed
   using the host measurements. Reference step counts are starting recipes rather
   than established convergence budgets. Training samples approximately 60% Yes
   through the sampler; labels remain unchanged. Validation and test retain their
   natural distributions, and test stays disabled during development.
5. **Evaluate the selected model.** Run full window-level validation, then test
   after model and threshold selection. Report accuracy, recall, precision,
   confusion counts and inference duration. For Predictor Judge, also run the
   guide's ordered episode shadow replay at recorded chunk starts. It evaluates
   module behavior along recorded actions; closed-loop intervention is a separate
   experiment. Report current-safe-to-future-violation results separately from
   ongoing violations. Unsafe-only source episodes do not measure false alarms on
   an independent safe-episode population.
6. **Record the result.** Retain the resolved configuration, revisions, hardware,
   training budget, sampling report, logs, selected checkpoint and evaluation
   outputs. Keep pilot diagnostics distinct from formal results. The model guides
   also cover optional tracking, caching, resume and model export.

Data and loader acceptance have been completed for this release. A real training
pilot on the target hardware, trained-model performance and closed-loop safety
remain separate validation stages. Offline training and evaluation use SafetyJev
and its pinned Open-Jev dependency; a running simulator or collection service is
not required.
