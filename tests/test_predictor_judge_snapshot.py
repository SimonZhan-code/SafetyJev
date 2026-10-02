import copy,unittest
from types import SimpleNamespace
import numpy as np
from safetyjev.predictor_judge_snapshot import capture_snapshot,restore_snapshot

class Scene:
    def __init__(self,robot):self.robots=[robot];self.state={'value':np.array([1.,2.])}
    def dump_state(self,serialized=False):
        self.robots[0]._ag_obj_constraint_params['arm']['position'][0]=999
        return copy.deepcopy(self.state)
    def load_state(self,value,serialized=False):self.state=copy.deepcopy(value)
class PredictorJudgeSnapshotTests(unittest.TestCase):
    def test_capture_protects_assisted_grasp_python_state_and_copies_arrays(self):
        robot=SimpleNamespace(_ag_obj_constraint_params={'arm':{'position':[1.,2.,3.]}})
        scene=Scene(robot);env=SimpleNamespace(scene=scene,robots=[robot])
        monitor=SimpleNamespace(_monitor=SimpleNamespace(_state=3),_violation_step=None,_violation_count=0,_error=None,_ltl_log=[],_prop_fns={})
        snapshot=capture_snapshot(env,monitor)
        self.assertEqual(robot._ag_obj_constraint_params['arm']['position'],[1.,2.,3.])
        scene.state['value'][0]=5
        self.assertEqual(snapshot['scene']['value'][0],1)
        restore_snapshot(env,monitor,snapshot,convert_state=lambda x:x)
        self.assertEqual(scene.state['value'][0],1);self.assertEqual(monitor._monitor._state,3)
