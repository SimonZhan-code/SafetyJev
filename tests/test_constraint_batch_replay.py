import base64
import importlib.util
import tempfile
from pathlib import Path
import unittest

from safetyjev.agentic_predictor import request_digest
spec=importlib.util.spec_from_file_location('batch_replay',Path(__file__).resolve().parents[1]/'scripts/remote/benchmark-constraint-batch.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


class ConstraintBatchReplayTests(unittest.TestCase):
    def test_reset_request_reconstructs_exactly_and_detects_changed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode=Path(tmp)
            for camera in ('overview','wrist'):(episode/(camera+'.png')).write_bytes(camera.encode())
            numeric={'history_mask':[False]*7+[True]}
            inputs={**numeric,'observations':{c:[None]*7+[base64.b64encode(c.encode()).decode()] for c in ('overview','wrist')}}
            questions=[{'id':'tilt','question':'Tilt?'}]
            row={'candidate_id':'one','inputs':numeric,'questions':questions,
                 'history_images':[{'images':{c:c+'.png' for c in ('overview','wrist')}}],
                 'request_sha256':request_digest(questions,inputs)}
            self.assertEqual(bench.saved_request(episode,row)['inputs'],inputs)
            (episode/'overview.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):bench.saved_request(episode,row)

    def test_missing_history_fails_before_inference(self):
        row={'inputs':{'history_mask':[True]*8},'history_images':[]}
        with self.assertRaisesRegex(ValueError,'Missing saved history'):bench.saved_request(Path('/tmp'),row)


if __name__=='__main__':unittest.main()
