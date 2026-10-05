import unittest
from collections import Counter
from safetyjev.predictor_judge_sampling import JudgeBatchSampler,consumed_draw_report


def row(i,label,event=6,episode='u',split='train',cid='upright',stratum='ordinary'):
    return {'id':str(i),'split':split,'target':[0.,1.] if label else [1.,0.],
            'family':'jar','constraint_id':cid,'episode_id':episode,
            'first_violation_step':event if label else None,'valid_steps':8,
            'episode_safety':'unsafe' if episode.startswith('u') else 'safe',
            'negative_stratum':stratum,'policy':'policy'}

class JudgeSamplingTests(unittest.TestCase):
    def test_reports_available_events_separately_from_resampled_draws(self):
        rows=[row(0,1,episode='u0'),row(1,1,episode='u0'),row(2,1,episode='u1'),row(3,0)]
        sampler=JudgeBatchSampler(rows,2,seed=42,epoch=0,samples_per_epoch=24)
        report=sampler.report()
        self.assertEqual(report['available']['positive_windows'],3)
        self.assertEqual(report['available']['negative_windows'],1)
        self.assertEqual(report['available']['positive_events'],2)
        self.assertEqual(report['positive_draws_per_available_window'],4)
        consumed=consumed_draw_report(rows,batch_size=2,seed=42,epoch=0,batch_offset=2,samples_per_epoch=24)
        self.assertEqual(consumed['available'],report['available'])
        self.assertEqual(consumed['draws'],4)

    def test_validation_retains_all_positives_and_fixed_negative_subset(self):
        from safetyjev import predictor_judge_sampling as s
        self.assertTrue(hasattr(s,'diagnostic_validation_subset'))
        rows=[row(i,i in (2,17,19),split='validation') for i in range(20)]
        indices,report=s.diagnostic_validation_subset(rows,negative_samples=2,seed=42)
        self.assertEqual(len(indices),5)
        self.assertTrue({2,17,19}.issubset(indices))
        self.assertEqual(indices,sorted(set(indices)))
        self.assertEqual(report['population'],{'positive':3,'negative':17})
        self.assertEqual(report['selected'],{'positive':3,'negative':2})
        self.assertEqual((indices,report),s.diagnostic_validation_subset(iter(rows),negative_samples=2,seed=42))
        self.assertFalse(report['full_split'])
        for rank in range(2):
            self.assertEqual(len(set(indices[rank::2])),len(indices[rank::2]))
        self.assertEqual(set(indices[0::2])|set(indices[1::2]),set(indices))

    def test_validation_handles_absent_labels_and_refuses_training_rows(self):
        from safetyjev import predictor_judge_sampling as s
        self.assertTrue(hasattr(s,'diagnostic_validation_subset'))
        for label in (0,1):
            rows=[row(i,label,split='validation') for i in range(3)]
            indices,report=s.diagnostic_validation_subset(rows,negative_samples=5,seed=1)
            self.assertEqual(indices,[0,1,2]);self.assertTrue(report['full_split'])
        with self.assertRaisesRegex(ValueError,'validation'):
            s.diagnostic_validation_subset([row(0,1)],negative_samples=5,seed=1)

    def test_balances_labels_events_and_resume_without_duplicate_claims(self):
        rows=[row(0,1,episode='u0')]+[row(i,1,episode='u1') for i in range(1,101)]
        rows += [row(101,0,episode='u0',stratum='near_event'),row(102,0,episode='s0')]
        sampler=JudgeBatchSampler(rows,8,seed=4,epoch=2,samples_per_epoch=10000)
        order=[i for b in sampler for i in b];labels=Counter(int(rows[i]['target'][1]) for i in order)
        self.assertEqual(labels,{0:5000,1:5000})
        n=sum(rows[i]['episode_id']=='u0' for i in order if rows[i]['target'][1])
        self.assertTrue(2200<n<2800,n)
        resumed=JudgeBatchSampler(rows,8,seed=4,epoch=2,start_batch=13,samples_per_epoch=10000)
        self.assertEqual(list(resumed),list(sampler)[13:])
        report=sampler.report();self.assertEqual(report['draws'],10000)
        self.assertLessEqual(report['unique_samples'],103);self.assertEqual(report['distinct_positive_events'],2)
        self.assertEqual(report['repeated_draws'],10000-report['unique_samples'])
    def test_single_label_and_heldout_handling(self):
        s=JudgeBatchSampler([row(0,0)],1,seed=1,epoch=0)
        self.assertEqual(s.report()['missing_labels'],['positive'])
        with self.assertRaisesRegex(ValueError,'train'):
            JudgeBatchSampler([row(1,1,split='test')],1,seed=1,epoch=0)
        with self.assertRaises(ValueError):JudgeBatchSampler([row(0,1)],1,seed=1,epoch=0,samples_per_epoch=0)
    def test_consumed_report_uses_cursor_not_full_planned_epoch(self):
        rows=[row(0,1),row(1,0)]
        report=consumed_draw_report(rows,batch_size=2,seed=42,epoch=1,batch_offset=1,samples_per_epoch=8)
        self.assertEqual(report['draws'],10)
        self.assertEqual(report['unique_samples'],2)
        self.assertEqual(report['scope'],'consumed through the saved trainer cursor')

    def test_sampling_rows_stream_metadata_without_decoding_media(self):
        import json,tempfile
        from pathlib import Path
        from safetyjev import predictor_judge_sampling as s
        self.assertTrue(hasattr(s,'SamplingRows'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'train.jsonl';data=[row(0,1),row(1,0)]
            path.write_text(''.join(json.dumps(r)+'\n' for r in data))
            class Dataset:
                def __init__(self):self.path=path
                def __len__(self):return 2
                def record(self,i):return data[i]
                def __getitem__(self,i):raise AssertionError('Must not decode images during sampling')
            rows=s.SamplingRows(Dataset())
            self.assertEqual(list(rows),data)
            sampler=JudgeBatchSampler(rows,2,seed=42,epoch=0)
            self.assertEqual(sampler.report()['labels'],{'positive':1,'negative':1})

    def test_rank_shards_and_consumed_counts_follow_global_draws(self):
        rows=[row(0,1),row(1,0),row(2,1,episode='u2')]
        ranks=[JudgeBatchSampler(rows,2,seed=4,epoch=0,samples_per_epoch=7,rank=r,world_size=2) for r in range(2)]
        self.assertEqual(ranks[0].order,ranks[1].order)
        batches=[list(s) for s in ranks]
        combined=[]
        for i in range(len(batches[0])):
            combined.extend(batches[0][i]);combined.extend(batches[1][i])
        self.assertEqual(combined,ranks[0].order)
        self.assertEqual(len(combined),8)
        report=consumed_draw_report(rows,batch_size=2,world_size=2,seed=4,epoch=0,batch_offset=1,samples_per_epoch=7)
        self.assertEqual(report['draws'],4)
