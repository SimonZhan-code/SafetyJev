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

    def two_video_fixture(self, root):
        pixels,rows=self.fixture(root)
        raw=root/'raw';other=raw/'wrist.mp4';other.write_bytes((raw/'view.mp4').read_bytes())
        manifest=raw/'manifest.jsonl'
        record=json.loads(manifest.read_text())
        record['files'].append({'path':other.name,'bytes':other.stat().st_size,'sha256':hashlib.sha256(other.read_bytes()).hexdigest()})
        manifest.write_text(json.dumps(record)+'\n')
        for row in rows:row['state']['observation_window']['videos']['wrist']='wrist.mp4'
        split=root/'train.jsonl';split.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        meta=json.loads((root/'dataset_metadata.json').read_text())
        meta['resources']['raw']['raw_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
        meta['file_sha256']['train']=hashlib.sha256(split.read_bytes()).hexdigest()
        (root/'dataset_metadata.json').write_text(json.dumps(meta))
        return pixels,rows

    def test_parallel_decode_matches_serial_and_runs_concurrently(self):
        import sqlite3,threading
        from unittest.mock import patch
        from safetyjev import visual_cache as vc
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.two_video_fixture(root)
            vc.build_visual_cache(root,root/'serial')
            original=vc._decode_video;barrier=threading.Barrier(2)
            def decode(*args):
                barrier.wait(timeout=5)
                return original(*args)
            with patch.object(vc,'_decode_video',side_effect=decode):
                report=vc.build_visual_cache(root,root/'parallel',workers=2)
            self.assertEqual(report['frames'],6)
            vc.validate_cache(root/'parallel',root)
            with sqlite3.connect(root/'serial/frames.sqlite') as a, sqlite3.connect(root/'parallel/frames.sqlite') as b:
                for query in ['SELECT key,png,sha FROM frames ORDER BY key','SELECT * FROM samples ORDER BY split,idx']:
                    self.assertEqual(a.execute(query).fetchall(),b.execute(query).fetchall())

    def test_parallel_failure_is_incomplete_and_resumable(self):
        from unittest.mock import patch
        from safetyjev import visual_cache as vc
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.two_video_fixture(root);cache=root/'cache'
            original=vc._decode_video
            def fail(path,*args):
                if Path(path).name=='wrist.mp4':raise ValueError('decode failure')
                return original(path,*args)
            with patch.object(vc,'_decode_video',side_effect=fail):
                with self.assertRaisesRegex(ValueError,'decode failure'):vc.build_visual_cache(root,cache,workers=2)
            with self.assertRaisesRegex(ValueError,'complete'):vc.FrameCache(cache,root)
            self.assertEqual(vc.build_visual_cache(root,cache,workers=2)['frames'],6)
            vc.validate_cache(cache,root)
            with self.assertRaisesRegex(ValueError,'workers'):vc.build_visual_cache(root,cache,workers=0)

    def test_parallel_verification_reports_coverage_without_changing_cache(self):
        from safetyjev import visual_cache as vc
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.two_video_fixture(root);cache=root/'cache'
            expected=vc.build_visual_cache(root,cache)
            before=vc.file_hash(cache/'frames.sqlite')
            self.assertEqual(vc.validate_cache(cache,root,workers=2),expected)
            progress=json.loads((cache/'verification.json').read_text())
            self.assertEqual(progress['stage'],'complete')
            self.assertEqual(progress['verified_frames'],6)
            self.assertEqual(progress['total_frames'],6)
            self.assertEqual(before,vc.file_hash(cache/'frames.sqlite'))
            with self.assertRaisesRegex(ValueError,'workers'):vc.validate_cache(cache,root,workers=0)

    def test_parallel_verification_rejects_bad_payloads_even_with_matching_database_hash(self):
        import sqlite3
        from safetyjev import visual_cache as vc
        for mismatch in [True,False]:
            with self.subTest(mismatch=mismatch),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.two_video_fixture(root);cache=root/'cache'
                vc.build_visual_cache(root,cache)
                with sqlite3.connect(cache/'frames.sqlite') as db:
                    db.execute('UPDATE frames SET png=?,sha=? WHERE rowid=6',
                               (b'not an image','incorrect' if mismatch else hashlib.sha256(b'not an image').hexdigest()))
                meta=json.loads((cache/'cache.json').read_text())
                meta['database_sha256']=vc.file_hash(cache/'frames.sqlite')
                (cache/'cache.json').write_text(json.dumps(meta))
                with self.assertRaises((ValueError,OSError)):vc.validate_cache(cache,root,workers=2)
                self.assertEqual(json.loads((cache/'verification.json').read_text())['stage'],'failed')

    def test_writer_waits_for_a_slow_external_reader(self):
        import sqlite3,threading
        from unittest.mock import patch
        from safetyjev import visual_cache as vc
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);cache=root/'cache'
            locked=threading.Event()
            def reader():
                with sqlite3.connect(cache/'frames.sqlite') as db:
                    db.execute('BEGIN');db.execute('SELECT count(*) FROM frames').fetchone()
                    locked.set();threading.Event().wait(6);db.rollback()
            original=vc._decode_video;thread=threading.Thread(target=reader)
            def decode(*args):
                result=original(*args);thread.start()
                self.assertTrue(locked.wait(3));return result
            try:
                with patch.object(vc,'_decode_video',side_effect=decode):report=vc.build_visual_cache(root,cache)
                self.assertEqual(report['frames'],3)
            finally:
                if thread.ident is not None:thread.join(10)

    def test_progress_file_reports_only_committed_frames_and_resumes(self):
        from unittest.mock import patch
        from safetyjev import visual_cache as vc
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.two_video_fixture(root);cache=root/'cache'
            original=vc._decode_video
            def decode(path,*args):
                if Path(path).name=='wrist.mp4':raise ValueError('interruption')
                return original(path,*args)
            with patch.object(vc,'_decode_video',side_effect=decode):
                with self.assertRaisesRegex(ValueError,'interruption'):vc.build_visual_cache(root,cache)
            progress=json.loads((cache/'progress.json').read_text())
            self.assertEqual(progress['stage'],'decoding');self.assertEqual(progress['ready_frames'],3)
            self.assertEqual(progress['total_frames'],6)
            vc.build_visual_cache(root,cache)
            progress=json.loads((cache/'progress.json').read_text())
            self.assertEqual(progress['stage'],'complete');self.assertEqual(progress['ready_frames'],6)

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
