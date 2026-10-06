"""Version-checked in-memory instrumentation. No edits to the upstream checkout."""
import hashlib
import sys
from pathlib import Path

COMMIT = "be97624e0acbec6b6f9260a08891b04168eb8e6c"
RUNNER_SHA256 = "db7e7d37f564bf26b56c1bc1bfe9182161118edffc29cd0cf6f90bbc4e01ebd0"
SOURCE_HASHES = {
    "maniguard/eval/benchmark.py": RUNNER_SHA256,
    "maniguard/utils/ltl_utils.py": "1f3cc04a0a6d9ff8db639f76128b5eae79468c7e74f569895fc51cebc40ad774",
    "maniguard/utils/safety_monitor.py": "690e12205c2dc068c37e2acf25ec9e05821d7581a016929720062f9379e6e37e",
}


def verify_sources(repo):
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Unsupported source " + name + "; use commit " + COMMIT)


def instrument(source, execution_mode="shadow"):
    if hashlib.sha256(source.encode()).hexdigest() != RUNNER_SHA256:
        raise ValueError("Unsupported ManiGuard runner. Use pinned commit " + COMMIT)
    if execution_mode not in ("shadow", "guard_regenerate", "visual_classification"):
        raise ValueError("Unknown execution mode")
    episode_import = ("from safetyjev.guard import GuardedEpisode as ShadowEpisode\n"
                      if execution_mode == "guard_regenerate"
                      else "from safetyjev.capture import ShadowEpisode\n")
    if execution_mode == "visual_classification":
        episode_import = "from safetyjev.visual_runtime import VisualClassificationEpisode as ShadowEpisode\n"
    edits = [
        ("import json\n", "import json\n" + episode_import),
        ("        step_idx = 0\n", "        sj_shadow = ShadowEpisode(scene_info, cfg, monitor, obs, episode_seed)\n\n        step_idx = 0\n"),
        ("                for ci in range(chunk_len):\n",
         "                for ci in range(chunk_len):\n                    sj_shadow.before_action(step_idx, obs, chunk, ci, chunk_len, action_space)\n"),
        ("                    if goal_checker is not None:\n",
         "                    sj_shadow.after_step(step_idx, monitor, obs)\n\n                    if goal_checker is not None:\n"),
        ("        all_results.append(result)\n        _ltl_str",
         "        sj_shadow.finish(result)\n        all_results.append(result)\n        _ltl_str"),
    ]
    if execution_mode == "guard_regenerate":
        anchor = "                chunk = query_policy(policy, obs, client_type, cfg)\n"
        edits.insert(2, (anchor, anchor +
            "                chunk = sj_shadow.select_chunk(\n"
            "                    step_idx, obs, chunk,\n"
            "                    lambda current_obs: query_policy(policy, current_obs, client_type, cfg),\n"
            "                    action_space, min(cfg.execute_horizon, cfg.max_steps - step_idx))\n"
            "                if chunk is None:\n"
            "                    # A guard stop is an unsuccessful task outcome, not an infrastructure crash.\n"
            "                    success = False\n"
            "                    done = True\n"
            "                    break\n"))
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("Integration anchor changed: " + repr(old))
        source = source.replace(old, new, 1)
    compile(source, "instrumented_maniguard", "exec")
    return source


def launch(repo, benchmark_args, options):
    from . import capture
    verify_sources(repo)
    path = Path(repo).resolve() / "maniguard/eval/benchmark.py"
    source = instrument(path.read_text(), options.get("execution_mode", "shadow"))
    capture.OPTIONS = options
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.argv = [str(path)] + benchmark_args
    # __file__ preserves ManiGuard's own REPO_ROOT calculations.
    namespace = {"__name__": "safetyjev_instrumented_runner", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    namespace["main"]()
