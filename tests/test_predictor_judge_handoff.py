"""Regression coverage for Judge training, held-out isolation and model delivery."""
import argparse
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from test_model_export import fixture
import test_predictor_judge_cache as cache_tests
from test_predictor_judge_distributed import Model


def judge_export_fixture(root):
    run, model, report = fixture(root)
    path = model/'model.json'
    config = json.loads(path.read_text())
    config.update(method='native_qwen_action_conditioned_noul', history_frames=3,
                  state_dim=16, max_actions=8, state_features=['joint']*16)
    path.write_text(json.dumps(config))
    (model/'predictor_judge.pt').write_bytes(b'projections and normalization')
    from safetyjev.model_export import checkpoint_hashes
    data = json.loads(report.read_text())
    data.update(task='predictor_judge', samples=20, checkpoint_sha256=checkpoint_hashes(model),
                events={'eligible':2,'detected':1,'recall':.5},
                by_constraint={'upright':{'recall':.5}}, threshold=.5)
    data.pop('evaluated')
    report.write_text(json.dumps(data))
    return run, model, report


class JudgeHandoffTests(unittest.TestCase):
    def test_chunk_export_keeps_contract_and_compact_metrics(self):
        from safetyjev.model_export import build_export,checkpoint_hashes
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,model,test=judge_export_fixture(root)
            cfg=json.loads((model/'model.json').read_text())
            cfg.update(method='native_qwen_chunk_noul',input_contract='chunk_start_v2',history_frames=8)
            (model/'model.json').write_text(json.dumps(cfg))
            report=json.loads(test.read_text());report['checkpoint_sha256']=checkpoint_hashes(model)
            report['headline']={'pairs':{'accuracy':.75}}
            report.update(evaluation_seconds=2.,evaluation_time_scope='fixture total wall duration')
            test.write_text(json.dumps(report))
            out=root/'export';build_export(run,test,out)
            card=(out/'README.md').read_text()
            self.assertIn('load_judge',card);self.assertIn('executed',card)
            self.assertNotIn('event-balanced',card)
            self.assertIn('historical robot states',card)
            self.assertEqual(json.loads((out/'metrics.json').read_text())['test']['evaluation_seconds'],2.)
            self.assertEqual(json.loads((out/'metrics.json').read_text())['test']['headline'],report['headline'])

    def test_export_preserves_judge_weights_metrics_and_upload_manifest(self):
        from safetyjev.model_export import build_export, upload_export
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,model,test=judge_export_fixture(root)
            out=root/'export';build_export(run,test,out)
            self.assertEqual((out/'model/predictor_judge.pt').read_bytes(), b'projections and normalization')
            metrics=json.loads((out/'metrics.json').read_text())
            self.assertEqual(metrics['test']['events']['recall'],.5)
            self.assertEqual(metrics['test']['by_constraint']['upright']['recall'],.5)
            self.assertEqual(json.loads((out/'provenance.json').read_text())['task'],'predictor_judge')
            self.assertIn('PredictorJudgeModel.load',(out/'README.md').read_text())
            for p in out.rglob('*.json'):
                self.assertNotIn('/private',p.read_text());self.assertNotIn('SECRET',p.read_text())
            api=MagicMock();upload_export(out,'team/judge',api=api)
            api.create_repo.assert_called_once_with(repo_id='team/judge',repo_type='model',private=True,exist_ok=False)
            (out/'model/predictor_judge.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):upload_export(out,'team/changed',api=api)

    def test_export_refuses_missing_or_changed_numeric_weights_and_wrong_test(self):
        from safetyjev.model_export import build_export, checkpoint_hashes
        for case in ('missing','changed','partial','wrong-task'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);run,model,test=judge_export_fixture(root)
                data=json.loads(test.read_text())
                if case=='missing':
                    (model/'predictor_judge.pt').unlink()
                    data['checkpoint_sha256']=checkpoint_hashes(model)
                elif case=='changed':(model/'predictor_judge.pt').write_bytes(b'changed')
                elif case=='partial':data['samples']=19
                else:data['task']='classifier'
                test.write_text(json.dumps(data))
                with self.assertRaises(ValueError):build_export(run,test,root/'export')
                self.assertFalse((root/'export').exists())

    def test_validation_only_training_tracks_resume_and_never_opens_test_loader(self):
        from safetyjev.predictor_judge_train import run
        from safetyjev.predictor_judge_dataset import PredictorJudgeWindowDataset
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=cache_tests.JudgeCacheTests().package(root)
            # Source-format packages keep derived indices in package/records.
            (package/'records').mkdir(exist_ok=True)
            metadata=json.loads((package/'dataset_metadata.json').read_text())
            for eid,res in metadata['resources'].items():
                res['record_file']='records/'+eid+'.json'
                (package/res['raw_root']/'record.json').rename(package/res['record_file'])
            (package/'dataset_metadata.json').write_text(json.dumps(metadata))
            config={'seed':42,'model':{'dtype':'float32'},
                    'data':{'package':str(package),'batch_size':1,'num_workers':0,'sampling':{'samples_per_epoch':16}},
                    'training':{'max_steps':2,'global_batch_size':2,'lr':.01,'head_lr':.01,'weight_decay':0.,
                                'warmup_steps':0,'clip_grad_norm':1.,'brier_weight':.1,'eval_every':1,'save_every':1},
                    'evaluation':{'max_batches':None,'validation_negative_samples':1,'run_test':False}}
            def dataset(path,split,**kwargs):
                self.assertNotEqual(split,'test','Development training opened the held-out test loader')
                return PredictorJudgeWindowDataset(path,split,**kwargs)
            sdk=MagicMock();sdk.init.return_value.summary={}
            original_import=importlib.import_module
            with patch.dict(os.environ,{'SAFETYJEV_TRACKING':'wandb','WANDB_PROJECT':'judge-test','WANDB_MODE':'offline','RANK':'0','WORLD_SIZE':'1'}), \
                 patch('safetyjev.experiment_tracking.importlib.import_module',
                       side_effect=lambda name,*a,**kw: sdk if name=='wandb' else original_import(name,*a,**kw)), \
                 patch('safetyjev.predictor_judge_train.PredictorJudgeWindowDataset',side_effect=dataset), \
                 patch('jev.predictor_judge_model.PredictorJudgeModel.from_pretrained',side_effect=lambda **kwargs:Model()), \
                 patch('jev.predictor_judge_model.PredictorJudgeModel.load',side_effect=Model.load):
                first=run(config,root/'training',device='cpu',stop_after=1)
                old=json.loads((root/'training/tracking.json').read_text())
                result=run(config,root/'training',device='cpu',resume=first['checkpoint'])
            self.assertIsNone(result['test']);self.assertEqual(result['test_status'],'not_evaluated')
            self.assertFalse((root/'training/evaluations/final-test.jsonl').exists())
            self.assertFalse((root/'training/final/test.jsonl').exists())
            tracking=json.loads((root/'training/tracking.json').read_text())
            self.assertEqual(tracking['id'],old['id']);self.assertEqual(tracking['attempt'],2)
            self.assertIn('test',result['identity']['split_hashes'])
            logs=[call.args[0] for call in sdk.init.return_value.log.call_args_list]
            self.assertTrue(any('train/loss' in row for row in logs))
            self.assertTrue(any('validation/events/eligible' in row for row in logs))
            self.assertTrue(any('final_validation/nll' in row for row in logs))
            self.assertEqual(sdk.init.return_value.summary['training_status'],'completed')

    def test_standalone_judge_evaluation_binds_numeric_weights_and_data(self):
        from safetyjev.evaluate_trained import run
        from safetyjev.predictor_judge_dataset import collate_predictor_judge
        from safetyjev.model_export import checkpoint_hashes,sha256
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);package=cache_tests.JudgeCacheTests().package(root)
            _,checkpoint,_=judge_export_fixture(root)
            model=Model();metadata=json.loads((package/'dataset_metadata.json').read_text())
            model.model_config.update({k:metadata[k] for k in ('history_frames','max_actions','state_features')})
            args=argparse.Namespace(task='predictor_judge',device='cpu',package=str(package),output=str(root/'test.json'),
                                    checkpoint=str(checkpoint),batch_size=2,workers=0,frame_cache=None,split='test',threshold=.5)
            with patch('jev.predictor_judge_model.PredictorJudgeModel.load',return_value=model), \
                 patch('safetyjev.predictor_judge_dataset.PreparedPredictorJudgeCollator',return_value=collate_predictor_judge):
                report=run(args)
            self.assertEqual(report['samples'],report['expected_samples'])
            self.assertEqual(report['checkpoint_sha256'],checkpoint_hashes(checkpoint))
            self.assertIn('predictor_judge.pt',report['checkpoint_sha256'])
            self.assertEqual(report['split_sha256'],sha256(package/'test.jsonl'))

    def test_public_config_preserves_judge_sampling_without_paths(self):
        from safetyjev.experiment_tracking import public_config
        config=public_config({'task':'predictor_judge','data':{'sampling':{'positive_fraction':.5,'samples_per_epoch':128,'token':'SECRET'}},
                              'evaluation':{'run_test':False,'validation_negative_samples':4096}})
        self.assertEqual(config['data']['sampling'],{'positive_fraction':.5,'samples_per_epoch':128})
        self.assertEqual(config['evaluation']['validation_negative_samples'],4096)
        self.assertNotIn('SECRET',json.dumps(config))
