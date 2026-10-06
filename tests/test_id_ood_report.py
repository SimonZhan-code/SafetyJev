import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).parents[1]/'scripts/reporting/build-id-ood-report.py'
spec=importlib.util.spec_from_file_location('id_ood_report',path)
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


class DomainReportTests(unittest.TestCase):
    def test_pending_and_failed_cases_never_count_as_safe_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resources={'families':{f:{'scenes_by_level':{l:['task_0000/'+l] for l in report.LEVELS}} for f in report.FAMILIES}}
            cases=root/'base/lid/cases';cases.mkdir(parents=True)
            (cases/'task_0000-base.json').write_text(json.dumps({'status':'completed','episode_id':'e','result':{'success':True,'ltl_violated':True,'steps':12}}))
            failed=root/'domain/target/lid/cases';failed.mkdir(parents=True)
            (failed/'task_0000-target.json').write_text(json.dumps({'scene':'task_0000/target','status':'failed','error':'fixture'}))
            data=report.collect(root/'base',root/'domain',resources)
            self.assertFalse(data['complete']);self.assertEqual(data['totals']['planned'],30)
            self.assertEqual(data['totals']['completed'],1);self.assertEqual(data['totals']['failed'],1)
            row=next(r for r in data['rows'] if r['family']=='lid' and r['level']=='base')
            self.assertEqual(row['task_successes'],1);self.assertEqual(row['raw_safe_successes'],0)
            self.assertEqual(sum(r['raw_safe_successes'] for r in data['rows']),0)

    def test_single_class_rows_do_not_create_balanced_accuracy_comparison(self):
        def row(level,balanced):
            return {'family':'jar','level':level,'classification':{'calibrated':{'by_question':{'jar_tilted':{'balanced_accuracy':balanced,'positive':0,'violation_recall':None}}}}}
        self.assertEqual(report.findings({'rows':[row('base',.8),row('target',None)]}),[])


if __name__=='__main__':unittest.main()
