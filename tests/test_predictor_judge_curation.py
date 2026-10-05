import copy
import unittest
from predictor_judge_fixture import episode_fixture
from safetyjev import predictor_judge_curation as c


def record(eid='u', violation=6, family='jar'):
    e=episode_fixture(violation_step=violation)
    e.update(episode_id=eid,group_id=f'{family}/{eid}',result={'status':'completed'})
    return e

class CurationTests(unittest.TestCase):
    def test_unsafe_only_does_not_filter_heldout_or_admit_failed_rollouts(self):
        records=[record('u'),record('s',None),record('only_safe',None,'lid'),
                 record('v',None),record('t'),record('failed')]
        records[-1]['result']['status']='crashed'
        inventory=[c.inventory_episode(e) for e in records]
        original=copy.deepcopy(inventory)
        assignment={r['group_id']:({'v':'validation','t':'test'}.get(r['episode_id'],'train')) for r in inventory}
        selected=c.select_episodes(inventory,assignment)
        self.assertEqual({r['episode_id'] for r in selected if r['selected']},{'u','v','t'})
        self.assertEqual(inventory,original)

    def test_complete_safe_unsafe_and_step_zero_are_distinct(self):
        safe=c.inventory_episode(record('s',None));unsafe=c.inventory_episode(record())
        self.assertEqual(safe['safety'],'safe');self.assertTrue(safe['quality_ok']);self.assertTrue(safe['active'])
        self.assertEqual(unsafe['safety'],'unsafe');self.assertEqual(unsafe['events'],[{'constraint_id':'upright','step':6}])
        zero=c.inventory_episode(record('z',0));self.assertEqual(zero['safety'],'unsafe')
        self.assertIn('initially_violated',zero['quality_reasons'])
    def test_incomplete_and_invalid_oracle_never_become_safe(self):
        for edit in ('status','missing','invalid','missing_constraint','nonfinite'):
            e=record('s',None)
            if edit=='status':e['result']['status']='crashed'
            if edit=='missing':e['oracle'].pop(5)
            if edit=='invalid':e['oracle'][5]['valid']=False
            if edit=='missing_constraint':e['oracle'][5]['violated']={}
            if edit=='nonfinite':e['observations'][5]['robot_state'][0]=float('nan')
            info=c.inventory_episode(e)
            self.assertFalse(info['quality_ok'],edit);self.assertEqual(info['safety'],'unknown',edit)
        e=record();e['result']['status']='crashed';info=c.inventory_episode(e)
        self.assertEqual(info['safety'],'unsafe');self.assertFalse(info['quality_ok'])
    def test_selection_is_deterministic_train_only_and_keeps_scarce_families(self):
        inventory=[c.inventory_episode(record(f'u{i}')) for i in range(8)]
        inventory += [c.inventory_episode(record(f's{i}',None)) for i in range(5)]
        inventory += [c.inventory_episode(record('other',None,'lid'))]
        inventory += [c.inventory_episode(record('heldout',None))]
        assignments={r['group_id']:('test' if r['episode_id']=='heldout' else 'train') for r in inventory}
        chosen=c.select_episodes(inventory,assignments,seed=42,train_episodes='unsafe_plus_safe')
        self.assertEqual(chosen,c.select_episodes(list(reversed(inventory)),assignments,seed=42,train_episodes='unsafe_plus_safe'))
        jar=[r for r in chosen if r['selected'] and r['split']=='train' and r['family']=='jar']
        self.assertEqual(sum(r['safety']=='unsafe' for r in jar),8)
        self.assertEqual(sum(r['safety']=='safe' for r in jar),2)
        self.assertTrue(next(r for r in chosen if r['episode_id']=='other')['selected'])
        self.assertTrue(next(r for r in chosen if r['episode_id']=='heldout')['selected'])
        e=record('idle',None)
        for o in e['observations']:o['robot_state']=[0.]*16
        info=c.inventory_episode(e);self.assertFalse(info['active'])
        self.assertFalse(c.select_episodes([info],{info['group_id']:'train'},seed=1)[0]['selected'])
        self.assertTrue(c.select_episodes([info],{info['group_id']:'test'},seed=1)[0]['selected'])
    def test_frozen_assignments_must_cover_every_group(self):
        with self.assertRaisesRegex(ValueError,'split'):
            c.select_episodes([c.inventory_episode(record())],{},seed=42)

    def test_checkpoint_revision_is_preserved_in_inventory_and_reports(self):
        from safetyjev.predictor_judge_curation import composition_report
        a=record('a');b=record('b')
        a['provenance']={'checkpoint':{'repo':'org/model','revision':'abc','step':10}}
        b['provenance']={'checkpoint':{'repo':'org/model','revision':'def','step':20}}
        rows=[c.inventory_episode(e) for e in (a,b)]
        self.assertNotEqual(rows[0]['checkpoint_id'],rows[1]['checkpoint_id'])
        chosen=c.select_episodes(rows,{r['group_id']:'train' for r in rows})
        report=composition_report(chosen,{})
        self.assertEqual(len(report['episodes']['by_checkpoint_id']),2)

    def test_event_counts_are_separate_from_correlated_window_counts(self):
        rows=[c.inventory_episode(record('a')),c.inventory_episode(record('b',None))]
        selected=c.select_episodes(rows,{r['group_id']:'train' for r in rows})
        report=c.composition_report(selected,{})
        self.assertIn('by_constraint',report['events'])
        self.assertEqual(report['events']['by_constraint']['jar/upright'],{'all':1,'selected':1})
        self.assertEqual(report['events']['by_split']['train'],{'all':1,'selected':1})
