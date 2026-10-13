"""Validate upstream dev protocol and compatibility without importing Isaac."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from safetyjev import maniguard


@unittest.skipUnless(os.environ.get('SAFETYJEV_MANIGUARD_ROOT') and maniguard.SOURCE_PROFILE == 'dev',
                     'Requires the validated dev checkout')
class DevProtocolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ['SAFETYJEV_MANIGUARD_ROOT'])
        self.source = (self.root / 'maniguard/eval/benchmark.py').read_text()

    def functions(self, *names):
        tree = ast.parse(self.source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'upstream-dev', 'exec'), scope)
        return scope

    def test_every_mode_verifies_and_hooks_remain_inside_execution(self):
        maniguard.verify_sources(self.root)
        for mode in ('shadow', 'guard_regenerate', 'agentic_native', 'codex_harness',
                     'visual_classification', 'oracle_only'):
            tree = ast.parse(maniguard.instrument(self.source, mode))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == 'sj_shadow']
            for method in ('before_action', 'after_step', 'finish'):
                self.assertEqual(sum(c.func.attr == method for c in calls), 1)
        self.assertEqual(maniguard.source_provenance()['maniguard_source_branch'], 'dev')

    def test_changed_checkout_or_goal_checker_is_rejected(self):
        with patch('safetyjev.maniguard.subprocess.check_output', return_value='wrong\n'):
            with self.assertRaisesRegex(ValueError, 'revision differs'):maniguard.verify_sources(self.root)
        original = Path.read_bytes
        def modified(path):
            return b'changed' if str(path).endswith('eval/goal_checker.py') else original(path)
        with patch.object(Path, 'read_bytes', modified):
            with self.assertRaisesRegex(ValueError, 'goal_checker.py'):maniguard.verify_sources(self.root)

    def test_initial_safety_rejected_before_policy_and_unavailable_verdicts(self):
        fn = self.functions('_validate_initial_safety', '_initialization_failure')
        for state in ({'state':None}, {'state':0,'doomed':True,'ap':{'x':False}}):
            monitor = SimpleNamespace(step=Mock(return_value=state))
            with self.assertRaises((ValueError, RuntimeError)):
                fn['_validate_initial_safety'](monitor, 'task')
            monitor.step.assert_called_once_with(0)
        result = fn['_initialization_failure']({'name':'task'},SimpleNamespace(seed=0),0,
                                               'both','initial_safety',ValueError('invalid'))
        self.assertEqual(result['status'],'initialization_failed')
        for key in ('success','ltl_violated','counted_violation'):
            self.assertIsNone(result[key])
        self.assertFalse(result['safety_evaluated'])

    def test_default_profile_and_manifest_are_dev(self):
        env = dict(os.environ);env.pop('SAFETYJEV_MANIGUARD_PROFILE',None)
        result = subprocess.check_output([sys.executable,'-c',
            'from safetyjev.maniguard import SOURCE_PROFILE, COMMIT; print(SOURCE_PROFILE,COMMIT)'],env=env,text=True)
        self.assertEqual(result.strip(), 'dev '+maniguard.DEV_MANIFEST['commit'])

    def test_unresolved_scope_fails_in_upstream_monitor(self):
        tree = ast.parse((self.root/'maniguard/utils/safety_monitor.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TaskLTLMonitor')
        init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        scope = {'ObjectResolver':lambda *a:SimpleNamespace(resolve_patterns=lambda p:{}),
                 'SafetyPropositionEvaluator':Mock(), '_merge_safety_data':lambda task,scene:task,
                 '_auto_generate_scene_propositions':lambda task,resolver:task}
        exec(compile(ast.Module(body=[init],type_ignores=[]),'upstream-monitor','exec'),scope)
        with self.assertRaisesRegex(ValueError,'matched no active objects'):
            scope['__init__'](SimpleNamespace(),None,ltl_safety={'combined_ltl':'G x',
                'propositions':{'x':{'over':['missing_999']}}})


class DevHeadTests(unittest.TestCase):
    def test_fresh_run_rejects_advanced_dev_without_modifying_checkout(self):
        with patch.object(maniguard, 'SOURCE_PROFILE', 'dev'), patch.object(
                maniguard.subprocess, 'check_output', return_value='f'*40+'\trefs/heads/dev\n'):
            with self.assertRaisesRegex(ValueError, 'advanced'):
                maniguard.verify_latest_dev()

    def test_historical_profile_does_not_contact_dev(self):
        with patch.object(maniguard, 'SOURCE_PROFILE', 'pinned'), patch.object(
                maniguard.subprocess, 'check_output') as remote:
            maniguard.verify_latest_dev()
            remote.assert_not_called()
