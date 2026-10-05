import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock


def fixture(root):
    from safetyjev.model_export import checkpoint_hashes
    run = root/'run'
    model = run/'final/model'
    (model/'adapter').mkdir(parents=True)
    (model/'processor').mkdir()
    (model/'head.pt').write_bytes(b'head')
    (model/'adapter/adapter_model.safetensors').write_bytes(b'weights')
    (model/'adapter/adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'vendor/model'}))
    (model/'adapter/README.md').write_text('PRIVATE host/user')
    (model/'processor/processor_config.json').write_text('{}')
    (model/'processor/tokenizer.json').write_text('{}')
    (model/'model.json').write_text(json.dumps({'method':'native_qwen_visual_noul','history_frames':1,
        'model_id':'vendor/model','revision':'a'*40,'lora_rank':8,'temperature':1.}))
    report = {'training':{'status':'completed','completed_step':1000,'best_step':800,'best_checkpoint':'/private/path'},
              'validation':{'evaluated':12,'nll':.1},'test':None,
              'identity':{'package_sha256':'package-digest','split_hashes':{'test':'test-digest'},
                          'evaluation':{'max_batches':None},'sources':{'visual_train.py':'sha'}}}
    (run/'final/report.json').write_text(json.dumps(report))
    (run/'run.json').write_text(json.dumps({'config':{'seed':42,'data':{'package':'/private/data'}}, 'token':'SECRET'}))
    test = {'task':'classifier','split':'test','expected_samples':20,'evaluated':20,'nll':.2,
            'checkpoint':str(model),'checkpoint_sha256':checkpoint_hashes(model),
            'package_sha256':'package-digest','split_sha256':'test-digest'}
    test_path = root/'test.json'; test_path.write_text(json.dumps(test))
    return run, model, test_path


class ExportTests(unittest.TestCase):
    def test_export_has_only_inference_files_and_sanitized_reports(self):
        from safetyjev.model_export import build_export
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,model,test=fixture(root)
            (run/'training_state.pt').write_bytes(b'optimizer')
            out=root/'export';build_export(run,test,out)
            self.assertTrue((out/'model/head.pt').exists())
            self.assertFalse((out/'model/adapter/README.md').exists())
            self.assertFalse((out/'training_state.pt').exists())
            for path in out.rglob('*.json'):
                self.assertNotIn('/private',path.read_text())
                self.assertNotIn('SECRET',path.read_text())
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertIn('model/head.pt',manifest['files_sha256'])
            self.assertEqual(json.loads((out/'metrics.json').read_text())['test']['evaluated'],20)

    def test_export_refuses_wrong_checkpoint_split_partial_test_and_incomplete_training(self):
        from safetyjev.model_export import build_export
        for field,value in [('checkpoint_sha256',{}),('split','validation'),('evaluated',19),('split_sha256','wrong')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);run,model,test=fixture(root)
                data=json.loads(test.read_text());data[field]=value;test.write_text(json.dumps(data))
                with self.assertRaises(ValueError):build_export(run,test,root/'export')
                self.assertFalse((root/'export').exists())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,_,test=fixture(root)
            p=run/'final/report.json';data=json.loads(p.read_text());data['identity']['evaluation']['max_batches']=1;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'capped'):build_export(run,test,root/'export')

    def test_export_requires_remote_pinned_backbone(self):
        from safetyjev.model_export import build_export,checkpoint_hashes
        for model_id,revision in [('/private/model',None),('vendor/model','main')]:
            with self.subTest(model_id=model_id),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);run,model,test=fixture(root)
                p=model/'model.json';data=json.loads(p.read_text());data.update(model_id=model_id,revision=revision);p.write_text(json.dumps(data))
                data=json.loads(test.read_text());data['checkpoint_sha256']=checkpoint_hashes(model);test.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError,'pinned'):
                    build_export(run,test,root/'export')

    def test_export_removes_local_processor_metadata(self):
        from safetyjev.model_export import build_export,checkpoint_hashes
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,model,test=fixture(root)
            (model/'processor/tokenizer_config.json').write_text(json.dumps({'name_or_path':'/private/cache','model_max_length':2048}))
            data=json.loads(test.read_text());data['checkpoint_sha256']=checkpoint_hashes(model);test.write_text(json.dumps(data))
            build_export(run,test,root/'export')
            saved=json.loads((root/'export/model/processor/tokenizer_config.json').read_text())
            self.assertNotIn('name_or_path',saved)
            self.assertEqual(saved['model_max_length'],2048)

    def test_upload_verifies_bundle_before_creating_new_private_repo(self):
        from safetyjev.model_export import build_export,upload_export
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,model,test=fixture(root)
            out=root/'export';build_export(run,test,out)
            api=MagicMock();upload_export(out,'team/new-model',api=api)
            api.create_repo.assert_called_once_with(repo_id='team/new-model',repo_type='model',private=True,exist_ok=False)
            api.upload_folder.assert_called_once()
            (out/'extra-secret.txt').write_text('SECRET')
            api.reset_mock()
            with self.assertRaises(ValueError):upload_export(out,'team/other',api=api)
            api.create_repo.assert_not_called()
