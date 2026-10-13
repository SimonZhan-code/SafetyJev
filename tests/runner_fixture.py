"""Controlled boundaries for both supported upstream runner versions."""
import ast
import os
from pathlib import Path


def prepare_scope(scope):
    source = Path(os.environ['SAFETYJEV_MANIGUARD_ROOT']) / 'maniguard/eval/benchmark.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == '_record_action_finiteness']
    if functions:
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
        scope['cfg'].action_dim = 8
        scope['recorder'] = None
        scope['rollout_diagnostics'] = {
            'action_attempts': 0, 'env_steps_returned': 0, 'observations_returned': 0,
            'first_nonfinite_action': None, 'termination_reason': None,
        }
