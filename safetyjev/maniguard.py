"""Version-checked in-memory instrumentation. No edits to the upstream checkout."""
import hashlib
import json
import subprocess
import os
import sys
from pathlib import Path

COMMIT = "be97624e0acbec6b6f9260a08891b04168eb8e6c"
RUNNER_SHA256 = "db7e7d37f564bf26b56c1bc1bfe9182161118edffc29cd0cf6f90bbc4e01ebd0"
SOURCE_HASHES = {
    "maniguard/eval/benchmark.py": RUNNER_SHA256,
    "maniguard/utils/ltl_utils.py": "1f3cc04a0a6d9ff8db639f76128b5eae79468c7e74f569895fc51cebc40ad774",
    "maniguard/utils/safety_monitor.py": "690e12205c2dc068c37e2acf25ec9e05821d7581a016929720062f9379e6e37e",
}


SOURCE_PROFILE = os.environ.get("SAFETYJEV_MANIGUARD_PROFILE", "dev")
if SOURCE_PROFILE not in ("dev", "pinned", "clutter-exact-v1"):
    raise ValueError("Unsupported ManiGuard source profile: " + SOURCE_PROFILE)
if SOURCE_PROFILE == "clutter-exact-v1":
    # Explicit, version-checked correction for vacuous exact-name Clutter scopes.
    SOURCE_HASHES["maniguard/utils/safety_monitor.py"] = "e03bb12b39c7e3b27155d760bd72ecb7c3f492f22199393f131a7b19a8653b1f"


DEV_MANIFEST = json.loads(Path(__file__).with_name("maniguard-dev.json").read_text())
if SOURCE_PROFILE == "dev":
    COMMIT = DEV_MANIFEST["commit"]
    SOURCE_HASHES = DEV_MANIFEST["source_hashes"]
    RUNNER_SHA256 = SOURCE_HASHES["maniguard/eval/benchmark.py"]


def source_provenance():
    return {"maniguard_reference_commit": COMMIT, "maniguard_source_profile": SOURCE_PROFILE,
            "maniguard_source_branch": "dev" if SOURCE_PROFILE == "dev" else "historical",
            "source_hashes": dict(SOURCE_HASHES)}


def verify_sources(repo):
    if SOURCE_PROFILE == "dev":
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT:
            raise ValueError("ManiGuard dev revision differs; prepare the validated commit " + COMMIT)
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Unsupported source " + name + "; use commit " + COMMIT)


def verify_latest_dev():
    if SOURCE_PROFILE != "dev":
        return
    latest = subprocess.check_output(
        ["git", "ls-remote", DEV_MANIFEST["repository"], "refs/heads/dev"],
        text=True, timeout=60).split()[0]
    if latest != COMMIT:
        raise ValueError("Upstream dev advanced to " + latest + "; validate the new adapter before starting another run")


def instrument(source, execution_mode="shadow", persistent_contacts=False):
    if hashlib.sha256(source.encode()).hexdigest() != RUNNER_SHA256:
        raise ValueError("Unsupported ManiGuard runner. Use pinned commit " + COMMIT)
    if execution_mode not in ("shadow", "guard_regenerate", "visual_classification", "oracle_only", "agentic_native", "codex_harness"):
        raise ValueError("Unknown execution mode")
    episode_import = ("from safetyjev.guard import GuardedEpisode as ShadowEpisode\n"
                      if execution_mode == "guard_regenerate"
                      else "from safetyjev.capture import ShadowEpisode\n")
    if execution_mode == "visual_classification":
        episode_import = "from safetyjev.visual_runtime import VisualClassificationEpisode as ShadowEpisode\n"
    if execution_mode == "oracle_only":
        episode_import = "from safetyjev.visual_runtime import OracleEpisode as ShadowEpisode\n"
    if execution_mode == "agentic_native":
        episode_import = "from safetyjev.agentic import AgenticEpisode as ShadowEpisode\n"
    if execution_mode == "codex_harness":
        episode_import = "from safetyjev.codex_episode import CodexEpisode as ShadowEpisode\n"
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
    if SOURCE_PROFILE == "dev":
        edits[1] = ('            eef2target_min_dist = float("inf")\n',
                    '            eef2target_min_dist = float("inf")\n'
                    "            sj_shadow = ShadowEpisode(scene_info, cfg, monitor, obs, episode_seed)\n")
        edits[-1] = ("        all_results.append(result)\n        if recorder is not None:",
                     "        sj_shadow.finish(result)\n        all_results.append(result)\n        if recorder is not None:")
    if execution_mode in ("agentic_native", "codex_harness"):
        old, new = edits[1]
        edits[1] = (old, new.replace("obs, episode_seed)", "obs, episode_seed, robot)"))
    if execution_mode in ("guard_regenerate", "agentic_native", "codex_harness"):
        anchor = "                chunk = query_policy(policy, obs, client_type, cfg)\n"
        proposal = anchor
        if execution_mode == "agentic_native":
            proposal = ("                policy_obs = sj_shadow.policy_observation(step_idx, obs)\n"
                        "                chunk = query_policy(policy, policy_obs, client_type, cfg)\n")
        if execution_mode == "codex_harness":
            # Codex owns all generation; never spend an implicit pi0.5 proposal.
            proposal = "                chunk = None\n"
        edits.insert(2, (anchor, proposal +
            "                chunk = sj_shadow.select_chunk(\n"
            "                    step_idx, obs, chunk,\n"
            "                    lambda current_obs: query_policy(policy, current_obs, client_type, cfg),\n"
            "                    action_space, min(cfg.execute_horizon, cfg.max_steps - step_idx))\n"
            "                if chunk is None:\n"
            "                    # A guard stop is an unsuccessful task outcome, not an infrastructure crash.\n"
            "                    success = False\n"
            "                    done = True\n"
            "                    break\n"))
    if persistent_contacts:
        if SOURCE_PROFILE != 'dev':
            raise ValueError('Persistent contact adapter requires pinned dev runner')
        edits.extend([
            ('            env.reset()\n', '            env.reset()\n'
             '            from safetyjev.contact_reports import prepare_contact_reports\n'
             '            from safetyjev.capture import OPTIONS as sj_options\n'
             '            sj_options["provenance"]["contact_reporting"] = prepare_contact_reports(env)\n'),
            ('            if og.sim is not None:\n', '            if og.sim is not None:\n'
             '                from safetyjev.contact_reports import close_contact_reports\n'
             '                close_contact_reports()\n'),
        ])
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
    source = instrument(path.read_text(), options.get("execution_mode", "shadow"), options.get("persistent_contacts", False))
    capture.OPTIONS = options
    sys.path.insert(0, str(Path(repo).resolve()))
    sys.argv = [str(path)] + benchmark_args
    # __file__ preserves ManiGuard's own REPO_ROOT calculations.
    namespace = {"__name__": "safetyjev_instrumented_runner", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    try:
        namespace["main"]()
    finally:
        if options.get("persistent_contacts"):
            from .contact_reports import close_contact_reports
            close_contact_reports()
