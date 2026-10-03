import json,tempfile,unittest,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from predictor_judge_fixture import episode_fixture
from safetyjev.predictor_judge_dataset import build_package,PredictorJudgeWindowDataset

class JudgeCacheTests(unittest.TestCase):
    def package(self,root):
        dirs=[];splits={}
        for i,split in enumerate(['train','validation','test']):
            d=root/'raw'/str(i);d.mkdir(parents=True);e=episode_fixture(violation_step=6)
            e.update(episode_id=str(i),group_id=f'jar/task_{i:04d}')
            splits[e['group_id']]=split
            for o in e['observations']:
                for name in o['images'].values():
                    p=d/name;p.parent.mkdir(exist_ok=True);Image.new('RGB',(8,8),(o['step'],20,30)).save(p)
            (d/'record.json').write_text(json.dumps(e));dirs.append(d)
        build_package(dirs,root/'package',group_splits=splits)
        return root/'package'

    def test_cache_preserves_all_numeric_image_and_label_inputs(self):
        from safetyjev.predictor_judge_cache import build_cache,validate_cache
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=self.package(root);report=build_cache(package,root/'cache')
            self.assertGreater(report['images'],0)
            for split in ['train','validation','test']:
                direct=PredictorJudgeWindowDataset(package,split)
                cached=PredictorJudgeWindowDataset(package,split,frame_cache=root/'cache')
                self.assertEqual(len(direct),len(cached))
                for i in range(len(direct)):
                    a,b=direct[i],cached[i]
                    self.assertEqual(a['target'],b['target'])
                    self.assertEqual(a['sample_id'],b['sample_id'])
                    for key in ['robot_state','remaining_actions','action_mask','history_mask','action_dt_s']:
                        np.testing.assert_array_equal(a['inputs'][key],b['inputs'][key])
                    for camera in ['overview','wrist']:np.testing.assert_array_equal(a['inputs']['observations'][camera],b['inputs']['observations'][camera])
            self.assertEqual(validate_cache(root/'cache',package)['samples'],report['samples'])

    def test_incomplete_and_corrupt_cache_are_rejected(self):
        from safetyjev.predictor_judge_cache import build_cache,JudgeCache,validate_cache
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=self.package(root);cache=root/'cache'
            with self.assertRaisesRegex(ValueError,'budget'):build_cache(package,cache,max_bytes=1)
            with self.assertRaisesRegex(ValueError,'complete'):JudgeCache(cache,package)
            build_cache(package,cache)
            with sqlite3.connect(cache/'frames.sqlite') as db:db.execute("UPDATE windows SET payload='{}'")
            data=PredictorJudgeWindowDataset(package,'train',frame_cache=cache)
            with self.assertRaisesRegex(ValueError,'checksum'):data[0]
            with self.assertRaisesRegex(ValueError,'checksum'):validate_cache(cache,package)

    def test_index_repointing_is_rejected_before_loading(self):
        from safetyjev.predictor_judge_cache import build_cache
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=self.package(root);cache=root/'cache';build_cache(package,cache)
            with sqlite3.connect(cache/'frames.sqlite') as db:db.execute("UPDATE samples SET offset=offset+1 WHERE split='train' AND idx=0")
            with self.assertRaisesRegex(ValueError,'index checksum'):PredictorJudgeWindowDataset(package,'train',frame_cache=cache)
