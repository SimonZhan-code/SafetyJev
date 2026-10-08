"""Consumer contract for generic ManiGuard HDF5 source recordings."""
import json
from pathlib import Path

import numpy as np
import unittest
import tempfile


def source_fixture(path, task=0):
    from maniguard.data.recording.writer import EpisodeWriter
    metadata = {
        'episode_id':f'source{task}', 'camera_keys':['image_left','wrist_image'], 'resolution':[16,16],
        'rgb_encoding':{'codec':'jpeg','quality':95,'subsampling':'444','color_order':'RGB'},
        'action_hz':20, 'physics_hz':120, 'joint_groups':{'arm':list(range(7)), 'gripper':[7,8]},
        'configuration':{'state_mode':'joint','action_dim':8,'ik_eef_to_joint':False,
                         'gripper_binarize':True,'controller_preset':'joint_position_raw','execute_horizon':8},
        'scene':{'scene_file':f'/bench/jar_transport/task_{task:04d}/base/scene_ep1.json',
                 'prompt':'Carry the jar', 'target_name':'jar_1', 'ltl_safety':{
                    'constraints':[{'id':'upright','ltl':'G upright','description':'Keep the jar upright'}],
                    'propositions':{'upright':{'state':'upright','over':['jar*'],'params':{'max_tilt_deg':30}}}}},
        'constraint_ap_names':{'upright':['upright']},
        'proposition_bindings':{'upright':{'over':[{'name':'jar_1','category':'jar'}]}},
        'ap_names':['upright'], 'provenance':{},
    }
    raw=np.tile(np.arange(8,dtype=np.float32)/10,(16,1));cmd=raw[:8].copy();cmd[:,-1]=1
    with EpisodeWriter(path,metadata,max_output_bytes=10000000) as writer:
        writer.append_event({'type':'proposal','proposal_id':0,'source_observation_id':0,
                             'actions':raw,'planned_steps':8,'action_low':[-2]*8,'action_high':[2]*8})
        # Terminates after four actions; future commanded tail must remain censored.
        for t in range(5):
            if t:
                writer.append_transition({'source_observation_id':t-1,'target_observation_id':t,
                    'proposal_id':0,'proposal_offset':t-1,'source_boundary_id':-1,'issued':True,
                    'raw':raw[t-1],'transformed':cmd[t-1],'applied':cmd[t-1],'duration_s':.05})
            writer.append_observation({'observation_id':t,'physics_tick':60+t*6,'sim_time_s':.5+t*.05,
                'camera_frames':{c:np.full((16,16,3),t*20,np.uint8) for c in metadata['camera_keys']},
                'robot':{'q':np.arange(9,dtype=np.float32)+t/10,'dq':np.arange(9,dtype=np.float32)/10},
                'safety':{'monitor_valid':True,'monitor_rejected':t>=3,'ap_values':np.array([t<3]),'ap_valid':np.array([True])},
                'constraint_records':{'upright':{'doomed':t>=3}},'monitor_record':{'valid':True}})
        writer.finish('complete',{'status':'completed','success':False})


class SourceEpisodeTests(unittest.TestCase):
    def setUp(self):
        try:
            from maniguard.data.recording.writer import EpisodeWriter
        except ImportError as exc:
            self.skipTest('ManiGuard source-reader integration environment required: '+str(exc))
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        self.root=Path(directory.name)


    def test_source_index_preserves_commands_and_original_jpeg(self):
        tmp_path=self.root
        from safetyjev.source_episodes import load_source_record, MediaReader
        from safetyjev.predictor_judge_data import build_predictor_judge_samples
        raw=tmp_path/'raw';source_fixture(raw)
        e=load_source_record(raw)
        assert e['group_id']=='jar_transport/task_0000'
        assert len(e['observations'])==5 and len(e['execution'])==4
        assert len(e['proposals'][0]['planned_commands'])==8
        np.testing.assert_allclose(e['observations'][0]['robot_state'][-2:],[7.5,.75])
        assert e['oracle'][3]['violated']=={'upright':True}
        assert 'target jar' in e['constraints'][0]['question']
        rows=list(build_predictor_judge_samples(e))
        assert rows[0]['target']==[0.,1.]
        assert rows[3]['target'] is None
        with MediaReader(raw) as media:
            blob=media.read(e['observations'][0]['images']['overview'])
        assert blob[:2]==b'\xff\xd8'
        assert not list(raw.rglob('*.jpg')) and not (raw/'record.json').exists()


    def test_source_package_cache_and_loader_agree(self):
        tmp_path=self.root
        from safetyjev.predictor_judge_dataset import build_package, PredictorJudgeWindowDataset
        from safetyjev.predictor_judge_cache import build_cache
        dirs=[];splits={}
        for t,split in enumerate(['train','validation','test']):
            d=tmp_path/'raw'/str(t);source_fixture(d,t);dirs.append(d);splits[f'jar_transport/task_{t:04d}']=split
        package=tmp_path/'package';build_package(dirs,package,group_splits=splits)
        direct=PredictorJudgeWindowDataset(package,'train');a=direct[0]
        build_cache(package,tmp_path/'cache')
        cached=PredictorJudgeWindowDataset(package,'train',frame_cache=tmp_path/'cache');b=cached[0]
        np.testing.assert_array_equal(a['inputs']['observations']['overview'],b['inputs']['observations']['overview'])
        assert a['inputs']['observations']['overview'].shape==(3,3,16,16)
        assert set(a['inputs'])=={'questions','observations','robot_state','remaining_actions','action_mask','action_dt_s','history_mask','constraint_context'}
        assert a['target']==b['target']
        direct.close();cached.close()


    def test_source_refuses_wrong_command_and_reports_reason(self):
        tmp_path=self.root
        import h5py
        from safetyjev.source_episodes import load_source_record
        raw=tmp_path/'raw';source_fixture(raw)
        with h5py.File(raw/'trajectory.hdf5','r+') as f:
            f['commands/applied'][0]=np.zeros(8)
        with self.assertRaisesRegex(ValueError,'Executed command differs'):
            load_source_record(raw)


    def test_source_refuses_invalid_clause_gt(self):
        tmp_path=self.root
        import h5py
        from safetyjev.source_episodes import load_source_record
        raw=tmp_path/'raw';source_fixture(raw)
        with h5py.File(raw/'trajectory.hdf5','r+') as f:
            f['safety/monitor_rejected'][0]=True
        with self.assertRaisesRegex(ValueError,'clause/task monitor disagreement'):
            load_source_record(raw)


    def test_source_does_not_label_unobserved_tail_safe(self):
        tmp_path=self.root
        import h5py
        from maniguard.data.recording.schema import pack_record,unpack_record,read_blob,append_blob
        from safetyjev.source_episodes import load_source_record
        from safetyjev.predictor_judge_data import build_predictor_judge_samples
        raw=tmp_path/'raw';source_fixture(raw)
        with h5py.File(raw/'trajectory.hdf5','r+') as f:
            rows=[unpack_record(read_blob(f['observation_metadata'],i)) for i in range(5)]
            del f['observation_metadata']
            for row in rows:
                row['constraint_records']['upright']['doomed']=False
                append_blob(f,'observation_metadata',pack_record(row))
            f['safety/monitor_rejected'][:]=False
            f['safety/ap_values'][:]=True
        rows=list(build_predictor_judge_samples(load_source_record(raw)))
        assert all(row['target'] is None for row in rows)
