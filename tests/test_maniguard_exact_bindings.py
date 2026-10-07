"""Exercise the actual ManiGuard resolver without importing the GPU simulator."""
import ast
import fnmatch
import os
from pathlib import Path
from types import SimpleNamespace
import typing
import unittest

SOURCE=Path(os.environ.get('MANIGUARD_SAFETY_MONITOR_SOURCE',str(Path(os.environ.get('SAFETYJEV_MANIGUARD_ROOT','/workspace/ManiGuard'))/'maniguard/utils/safety_monitor.py')))


@unittest.skipUnless(SOURCE.exists(),'ManiGuard source required')
class ExactObjectBindingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree=ast.parse(SOURCE.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['ObjectResolver','build_active_objects_for_ltl']]
        cls.ns={'fnmatch':fnmatch,'Any':typing.Any,'category_synset_lemma':lambda c:c}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),cls.ns)

    def env(self):
        objects=[SimpleNamespace(name='teacup_178',category='teacup'),SimpleNamespace(name='bowl_176',category='bowl')]
        return SimpleNamespace(robots=[],task=SimpleNamespace(object_scope={}),scene=SimpleNamespace(objects=objects,object_registry=lambda key,name:next((o for o in objects if o.name==name),None)))

    def test_exact_target_and_obstacle_scopes_are_nonempty(self):
        env=self.env();spec={'propositions':{'target':{'over':['teacup_178']},'obstacle':{'over':['bowl_176']}}}
        active=self.ns['build_active_objects_for_ltl'](env,spec,None)
        resolver=self.ns['ObjectResolver'](env,active)
        self.assertIs(resolver.resolve_patterns(['teacup_178'])['teacup_178'],env.scene.objects[0])
        self.assertIs(resolver.resolve_patterns(['bowl_176'])['bowl_176'],env.scene.objects[1])

    def test_existing_wildcard_scope_still_resolves(self):
        env=self.env();active=self.ns['build_active_objects_for_ltl'](env,{'propositions':{'x':{'over':['teacup_*']}}},None)
        matches=self.ns['ObjectResolver'](env,active).resolve_patterns(['teacup_*'])
        self.assertEqual(list(matches.values()),[env.scene.objects[0]])

    def test_missing_exact_object_fails_instead_of_vacuous_truth(self):
        with self.assertRaisesRegex(ValueError,'Unresolved LTL'):
            self.ns['build_active_objects_for_ltl'](self.env(),{'propositions':{'x':{'over':['missing_999']}}},None)
