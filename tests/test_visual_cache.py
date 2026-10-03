import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

class VisualCacheTests(unittest.TestCase):
    def fixture(self, root):
        raw=root/'raw';raw.mkdir()
        pixels=np.zeros((6,16,16,3),dtype=np.uint8)
        for i in range(6):pixels[i]=[i*30,255-i*30,42]
        video=raw/'view.mp4'
        subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','16x16','-framerate','30','-i','pipe:0','-c:v','libx264rgb','-crf','0',str(video)],input=pixels.tobytes(),check=True)
        files=[{'path':'view.mp4','bytes':video.stat().st_size,'sha256':hashlib.sha256(video.read_bytes()).hexdigest()}]
        manifest=raw/'manifest.jsonl';manifest.write_text(json.dumps({'files':files})+'\n')
        rows=[]
        for i in [1,3,5]:
            for q in ['tilted','closed']:
                rows.append({'id':f'{i}-{q}','kind':'noul','options':['no','yes'],'target':[1,0],'split':'train','question':q,
                             'state':{'observation_window':{'resource_id':'raw','videos':{'overview':'view.mp4','wrist':'view.mp4'},'frame_indices':[i],'video_fps':30}},
                             'metadata':{'family':'jar','query_id':q}})
        path=root/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        meta={'file_sha256':{'train':hashlib.sha256(path.read_bytes()).hexdigest()},'resources':{'raw':{'raw_root':'raw','raw_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}}}
        (root/'dataset_metadata.json').write_text(json.dumps(meta))
        return pixels,rows

    def test_cache_deduplicates_frames_and_preserves_samples(self):
        from safetyjev.visual_cache import build_visual_cache,FrameCache,frame_key
        from safetyjev.visual_dataset import APWindowDataset
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pixels,rows=self.fixture(root);cache=root/'cache'
            report=build_visual_cache(root,cache,max_bytes=1024**2)
            self.assertEqual(report['frames'],3)
            self.assertEqual(report['samples'],6)
            reader=FrameCache(cache,root)
            key=frame_key('raw','view.mp4',3,30)
            np.testing.assert_array_equal(reader.read(key),pixels[3])
            ds=APWindowDataset(root,'train',frame_cache=cache)
            self.assertEqual(len(ds),6)
            np.testing.assert_array_equal(ds[2]['inputs']['observations']['overview'][0],pixels[3])
            self.assertEqual(ds[2]['target'].tolist(),[1.,0.])
            self.assertEqual(build_visual_cache(root,cache,max_bytes=1024**2)['frames'],3)
            reader.close()

    def test_budget_interrupt_remains_incomplete_and_can_resume(self):
        from safetyjev.visual_cache import build_visual_cache,FrameCache
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);cache=root/'cache'
            with self.assertRaisesRegex(ValueError,'budget'):
                build_visual_cache(root,cache,max_bytes=1)
            with self.assertRaisesRegex(ValueError,'complete'):
                FrameCache(cache,root)
            self.assertEqual(build_visual_cache(root,cache,max_bytes=1024**2)['frames'],3)

    def test_missing_sample_is_not_silently_dropped(self):
        from safetyjev.visual_cache import build_visual_cache,validate_cache
        from safetyjev.visual_dataset import APWindowDataset
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);cache=root/'cache';build_visual_cache(root,cache)
            with sqlite3.connect(cache/'frames.sqlite') as db:db.execute("DELETE FROM samples WHERE idx=2")
            with self.assertRaisesRegex(ValueError,'count|checksum'):
                APWindowDataset(root,'train',frame_cache=cache)
            with self.assertRaisesRegex(ValueError,'count|checksum'):
                validate_cache(cache,root)

    def test_repointed_index_is_not_admitted_by_loader(self):
        from safetyjev.visual_cache import build_visual_cache
        from safetyjev.visual_dataset import APWindowDataset
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);cache=root/'cache';build_visual_cache(root,cache)
            with sqlite3.connect(cache/'frames.sqlite') as db:
                db.execute("UPDATE samples SET offset=(SELECT offset FROM samples WHERE idx=0) WHERE idx=2")
            with self.assertRaisesRegex(ValueError,'index checksum'):
                APWindowDataset(root,'train',frame_cache=cache)

    def test_loader_profile_uniform_matches_training_and_creates_output(self):
        from tools import profile_visual_loader
        from safetyjev.visual_distributed import GlobalBatchSampler
        from unittest.mock import patch
        import contextlib,io
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            config=root/'config.json';config.write_text(json.dumps({'seed':42,'model':{},'data':{'package':str(root),'batch_size':1,'balance':'uniform'}}))
            output=root/'new-directory/profile.json'
            with patch('sys.argv',['profile','--config',str(config),'--workers','0','--batches','2','--output',str(output)]), \
                 patch.object(profile_visual_loader,'GlobalBatchSampler',wraps=GlobalBatchSampler) as sampler, \
                 contextlib.redirect_stdout(io.StringIO()):
                profile_visual_loader.main()
            self.assertIsNone(sampler.call_args.kwargs['weights'])
            self.assertEqual(json.loads(output.read_text())['samples'],2)

    def test_source_change_and_corrupt_frame_are_rejected(self):
        from safetyjev.visual_cache import build_visual_cache,FrameCache,frame_key
        import sqlite3
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);cache=root/'cache';build_visual_cache(root,cache)
            with sqlite3.connect(cache/'frames.sqlite') as db:db.execute("UPDATE frames SET png=? WHERE frame=3",(b'corrupt',))
            reader=FrameCache(cache,root)
            with self.assertRaisesRegex(ValueError,'checksum'):reader.read(frame_key('raw','view.mp4',3,30))
            reader.close()
            (root/'dataset_metadata.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'identity'):FrameCache(cache,root)

if __name__=='__main__':unittest.main()
