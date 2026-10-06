import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('partitioned_plan',ROOT/'scripts/remote/partitioned_plan.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.assignment=json.loads((ROOT/'configs/two-node-assignments.json').read_text())
        self.resources=json.loads((ROOT/'docs/results/2026-10-06-id-ood-queue/domain-sweep-resources.json').read_text())

    def test_exact_coverage_and_no_overlap(self):
        owners=plan.validate_assignment(self.assignment,self.resources)
        self.assertEqual(len(owners),1000)
        self.assertEqual(sum(w=='node-a' for w in owners.values()),574)
        self.assertEqual(sum(w=='node-b' for w in owners.values()),426)
        self.assertEqual(owners['jar/task_0000/base'],'node-b')
        self.assertEqual(owners['clutter/task_0000/base'],'node-a')

    def test_reject_overlap_omission_and_queue_repetition(self):
        changed=copy.deepcopy(self.assignment)
        changed['workers']['node-b']['cases'].append(changed['workers']['node-a']['cases'][0])
        with self.assertRaisesRegex(ValueError,'Overlapping'):plan.validate_assignment(changed,self.resources)
        changed=copy.deepcopy(self.assignment);changed['workers']['node-b']['cases'].pop()
        with self.assertRaisesRegex(ValueError,'exact benchmark'):plan.validate_assignment(changed,self.resources)
        changed=copy.deepcopy(self.assignment);changed['workers']['node-b']['phases'].append(changed['workers']['node-b']['phases'][0])
        with self.assertRaisesRegex(ValueError,'repeated'):plan.validate_assignment(changed,self.resources)

    def test_cannot_execute_other_workers_cases(self):
        selected={'jar':{'scenes':['task_0000/base']}}
        with self.assertRaisesRegex(ValueError,'does not own'):plan.authorize_selection(self.assignment,self.resources,'node-a',selected)
        self.assertEqual(len(plan.authorize_selection(self.assignment,self.resources,'node-b',selected)),64)


if __name__=='__main__':unittest.main()
