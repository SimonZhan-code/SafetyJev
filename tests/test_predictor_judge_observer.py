import copy,unittest
from types import SimpleNamespace
from safetyjev.predictor_judge_observer import constraint_question

class PredictorJudgeObserverTests(unittest.TestCase):
    def test_group_identity_is_independent_of_scene_selection_root(self):
        from safetyjev.predictor_judge_observer import base_task_identity
        scene={'scene_file':'/bench/clutter_pickup/task_0013/base/scene_ep1.json'}
        for relative in ['task_0013/base','base','clutter_pickup/task_0013/base']:
            self.assertEqual(base_task_identity({**scene,'name':relative}),'clutter_pickup/task_0013')
    def test_question_includes_object_scope_and_relative_objects(self):
        constraint={'description':'The target must remain upright.'}
        spec={'propositions':{'p':{'state':'upright','over':['teacup_147'],'params':{'max_tilt_deg':30}}}}
        first=constraint_question(constraint,spec,['p'])
        other=copy.deepcopy(spec);other['propositions']['p']['over']=['bowl_143']
        second=constraint_question(constraint,other,['p'])
        self.assertNotEqual(first,second);self.assertIn('teacup',first);self.assertIn('bowl',second)
        relation={'propositions':{'p':{'state':'ontop','over':['jar_1'],'relative_to':['table_2']}}}
        question=constraint_question({'description':'Stay supported.'},relation,['p'])
        self.assertIn('jar',question);self.assertIn('table',question)
    def robot(self):
        arm=type('JointController',(),{})();arm._motor_type='position';arm._use_delta_commands=False
        arm._command_input_limits=None;arm._command_output_limits=None;arm._use_impedances=False;arm.command_dim=7
        gripper=type('MultiFingerGripperController',(),{})();gripper._mode='binary';gripper._command_input_limits=None;gripper.command_dim=1
        return SimpleNamespace(default_arm='0',controllers={'arm_0':arm,'gripper_0':gripper},_action_normalize=False)
    def test_effective_controller_must_match_action_units(self):
        from safetyjev.predictor_judge_observer import validate_loaded_controller
        validate_loaded_controller(self.robot())
        for field,value in [('_motor_type','velocity'),('_use_delta_commands',True),('_command_input_limits',(-1,1)),('_command_output_limits',(-2,2))]:
            robot=self.robot();setattr(robot.controllers['arm_0'],field,value)
            with self.assertRaises(ValueError):validate_loaded_controller(robot)
    def test_missing_required_ap_cannot_become_a_negative_label(self):
        from safetyjev.predictor_judge_observer import ManiGuardObserver
        observer=ManiGuardObserver.__new__(ManiGuardObserver);observer.invalid=False
        observer.machines={'no_drop':SimpleNamespace(ap_list=['dropped'],state=0,step=lambda ap:{'doomed':False})}
        observer.violated={'no_drop':False};observer._measurements=lambda:{}
        monitor=SimpleNamespace(_ltl_log=[{'step':0,'ap':{},'state':0}],violated=False)
        self.assertFalse(observer._oracle(0,monitor)['valid'])
    def test_capture_timing_preserves_return_values_and_counts_failures(self):
        from safetyjev.predictor_judge_observer import timed_capture
        class Probe:
            def __init__(self):self.timings={}
            @timed_capture
            def work(self,fail=False):
                if fail:raise ValueError('expected')
                return 17
        p=Probe();self.assertEqual(p.work(),17)
        with self.assertRaises(ValueError):p.work(True)
        self.assertEqual(p.timings['work']['calls'],2)
        self.assertGreaterEqual(p.timings['work']['seconds'],0)

class CameraCalibrationTests(unittest.TestCase):
    def sensor(self,matrix):
        import numpy as np
        return SimpleNamespace(intrinsic_matrix=np.asarray(matrix),image_width=256,image_height=256,
            get_attribute=lambda name:{'projection':'perspective','focalLength':17.,'horizontalAperture':20.995,
                'verticalAperture':20.995,'horizontalApertureOffset':0.,'verticalApertureOffset':0.,'clippingRange':[.01,100.]}[name])
    def test_unready_intrinsics_are_not_marked_available(self):
        from safetyjev.predictor_judge_observer import camera_calibration
        result=camera_calibration(self.sensor([[0,0,0],[0,0,0],[0,0,1]]))
        self.assertFalse(result['intrinsic_matrix']['available'])
        self.assertEqual(result['usd_camera']['value']['projection'],'perspective')
        self.assertEqual(result['resolution_px'],[256,256])
    def test_valid_intrinsics_and_usd_attributes_are_retained(self):
        from safetyjev.predictor_judge_observer import camera_calibration
        matrix=[[207.,0.,128.],[0.,207.,128.],[0.,0.,1.]]
        result=camera_calibration(self.sensor(matrix))
        self.assertEqual(result['intrinsic_matrix'],{'available':True,'value':matrix})
        self.assertEqual(result['usd_camera']['value']['focalLength'],17.)
