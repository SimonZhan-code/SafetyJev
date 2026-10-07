import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('reservation',Path(__file__).parents[1]/'scripts/remote/serve-reserved-base-policy.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class FamilyReservationTests(unittest.TestCase):
    def test_only_assigned_instance_can_launch_each_family(self):
        reservation={'schema':1,'families':{'cabinet':'54533396','clutter':'54566989'}}
        m.authorize_family('cabinet','54533396',reservation)
        m.authorize_family('clutter','54566989',reservation)
        for family,instance in [('clutter','54533396'),('cabinet','54566989'),('jar','54566989'),('clutter',None)]:
            with self.subTest(family=family,instance=instance):
                with self.assertRaises(ValueError):m.authorize_family(family,instance,reservation)
        with self.assertRaises(ValueError):m.authorize_family('clutter','54566989',{})

class ClutterCompletionTests(unittest.TestCase):
    def test_merge_requires_exact_terminal_scene_set(self):
        s=importlib.util.spec_from_file_location('merge',Path(__file__).parents[1]/'scripts/reporting/merge-clutter-on-completion.py')
        merge=importlib.util.module_from_spec(s);s.loader.exec_module(merge)
        progress={'status':'finished','planned':55}
        cases=[{'family':'clutter','scene':f'task_{i:04d}/base','status':'completed'} for i in range(55)]
        self.assertTrue(merge.terminal_55(progress,cases))
        self.assertFalse(merge.terminal_55(progress,cases[:-1]))
        self.assertFalse(merge.terminal_55(progress,cases[:-1]+[cases[0]]))
        cases[-1]={**cases[-1],'status':'running'}
        self.assertFalse(merge.terminal_55(progress,cases))
