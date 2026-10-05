import json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from predictor_judge_fixture import episode_fixture
from safetyjev.predictor_judge_dataset import build_package,PredictorJudgeWindowDataset,collate_predictor_judge

class PredictorJudgeDatasetTests(unittest.TestCase):
    def test_package_loads_causal_inputs_and_grouped_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dirs=[]
            for i in range(10):
                d=root/'raw'/str(i);d.mkdir(parents=True);e=episode_fixture(violation_step=6 if i%2 else None)
                e['episode_id']=str(i);e['group_id']=f'jar/task_{i:04d}'
                for o in e['observations']:
                    for camera,name in o['images'].items():
                        f=d/name;f.parent.mkdir(exist_ok=True);Image.new('RGB',(8,8),(o['step'],0,0)).save(f)
                (d/'record.json').write_text(json.dumps(e));dirs.append(d)
            output=root/'package';summary=build_package(dirs,output)
            groups={}
            for split in ['train','validation','test']:
                data=PredictorJudgeWindowDataset(output,split)
                for i in range(len(data)):
                    r=data.record(i);groups.setdefault(r['group_id'],set()).add(split)
                item=data[0];batch=collate_predictor_judge([item])
                self.assertEqual(batch['inputs']['remaining_actions'].shape,(1,8,8))
                self.assertEqual(batch['inputs']['observations']['overview'].shape,(1,3,3,8,8))
                self.assertNotIn('oracle',batch['inputs']);self.assertNotIn('target',batch['inputs'])
                self.assertTrue(batch['inputs']['history_mask'][0,-1])
            self.assertTrue(all(len(s)==1 for s in groups.values()))
            self.assertEqual(len(summary['normalization']['state_mean']),16)
            # Poison only validation records: normalization must use train resources only.
            expected=[]
            for eid,res in summary['resources'].items():
                if summary['group_splits'][res['group_id']]=='train':
                    expected.extend(o['robot_state'] for o in json.loads((dirs[int(eid)]/'record.json').read_text())['observations'])
            np.testing.assert_allclose(summary['normalization']['state_mean'],np.mean(expected,axis=0))

    def test_heldout_state_changes_do_not_change_normalization_and_image_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dirs=[]
            for i in range(10):
                d=root/'raw'/str(i);d.mkdir(parents=True);e=episode_fixture(violation_step=None)
                e['episode_id']=str(i);e['group_id']=f'jar/task_{i:04d}'
                for o in e['observations']:
                    for camera,name in o['images'].items():
                        f=d/name;f.parent.mkdir(exist_ok=True);Image.new('RGB',(8,8),(o['step'],0,0)).save(f)
                (d/'record.json').write_text(json.dumps(e));dirs.append(d)
            first=build_package(dirs,root/'one',train_episodes='unsafe_plus_safe')
            for d in dirs:
                e=json.loads((d/'record.json').read_text())
                if first['group_splits'][e['group_id']]!='train':
                    for o in e['observations']:o['robot_state']=[v+1000 for v in o['robot_state']]
                    (d/'record.json').write_text(json.dumps(e))
            second=build_package(dirs,root/'two',train_episodes='unsafe_plus_safe')
            self.assertEqual(first['normalization'],second['normalization'])
            data=PredictorJudgeWindowDataset(root/'two','train');row=data.record(0)
            d=dirs[int(row['episode_id'])];e=json.loads((d/'record.json').read_text())
            Image.new('RGB',(8,8),(255,255,255)).save(d/e['observations'][0]['images']['overview'])
            with self.assertRaisesRegex(ValueError,'Image bytes changed'):data[0]

    def test_curated_membership_preserves_heldout_and_counts_missing_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dirs=[];assignment={}
            for i in range(9):
                d=root/'raw'/str(i);d.mkdir(parents=True)
                e=episode_fixture(violation_step=6 if i<4 else None)
                e.update(episode_id=str(i),group_id=f'jar/task_{i:04d}',result={'status':'completed'})
                assignment[e['group_id']]='test' if i>=6 else 'train'
                for o in e['observations']:
                    for name in o['images'].values():
                        f=d/name;f.parent.mkdir(exist_ok=True);Image.new('RGB',(8,8)).save(f)
                (d/'record.json').write_text(json.dumps(e));dirs.append(d)
            (dirs[8]/'observations/0-overview.png').unlink()
            summary=build_package(dirs,root/'package',group_splits=assignment)
            inventory=[json.loads(s) for s in (root/'package/episode_inventory.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['selected'] and r['split']=='train' for r in inventory),4)
            self.assertEqual(sum(r['selected'] and r['split']=='test' for r in inventory),2)
            self.assertIn('invalid_media',inventory[8]['quality_reasons'])
            report=json.loads((root/'package/composition.json').read_text())
            self.assertEqual(report['episodes']['all']['safety']['unsafe'],4)
            self.assertEqual(report['episodes']['selected']['safety']['safe'],2)
            self.assertEqual(summary['selection']['train_episodes'],'unsafe_only')
            self.assertEqual(summary['selection']['safe_quota'],'none')
            self.assertTrue(all(r['safety']=='unsafe' for r in inventory if r['split']=='train' and r['selected']))
            self.assertEqual(report['events']['selected'],4)
            self.assertEqual(report['windows']['by_split']['test']['negative'],32)
            self.assertTrue((root/'package/DATA_SUMMARY.md').is_file())
            self.assertEqual(summary['group_splits'],assignment)
