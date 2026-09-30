# Open-Jev integration

SafetyJev is the main project for robot data, experiments and runtime integration.
The Open-Jev dependency is a separately versioned fork, checked out at
`third_party/Open-Jev`.

- Fork: https://github.com/666harrypeng/Open-Jev
- Upstream: https://github.com/Zefan-Cai/Open-Jev
- Initial reference: `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`

```bash
git submodule update --init --recursive
git submodule status
```

The parent repository records an exact submodule commit, not a moving branch.
The shared submodule URL uses HTTPS for cloning. The local checkout is configured
to push to the fork using SSH and has an `upstream` remote for the original
project; these local remote settings are not inherited by other clones.

## Development boundary

Data selection, AP annotation, family query definitions, Dataset/DataLoader and
robot integration belong in SafetyJev. Generic visual-input model, processor,
training-state and checkpoint support live in the fork's `jev/visual_model.py`
and `jev/visual_training.py`, preserving the existing text decision path.
SafetyJev supplies the robotics training launcher and current-image inference
entrypoint.

The fork's `main` branch carries the visual model and training extensions for
SafetyJev. The model uses one current overview image and one current wrist image
per question, retaining the scalar Noul head and `[No, Yes]` target convention.
See [visual training](visual-training.md).

Maintain dependency changes on the fork's `main` branch and commit them there
first. Then record that exact child commit in SafetyJev. Publish the child commit
before publishing the parent pointer so collaborators can clone the recorded
version. These extensions are maintained for SafetyJev; an upstream PR is not
part of this workflow.

## Dependencies

The tested training dependencies are pinned in `requirements-visual.txt`.
See [environment setup](visual-training.md#environment) for installation.
