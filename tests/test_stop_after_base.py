import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

p=Path(__file__).parents[1]/'scripts/remote/stop-after-base.py'
spec=importlib.util.spec_from_file_location('base_stop',p)
stop=importlib.util.module_from_spec(spec);spec.loader.exec_module(stop)


class StopAfterBaseTests(unittest.TestCase):
    def test_requires_all_200_terminal_cases_and_both_finished_workers(self):
        a={'status':'finished','planned':174};b={'status':'finished','planned':26}
        totals={'planned':200,'completed':200,'failed':0,'running':0,'pending':0}
        self.assertTrue(stop.all_base_terminal(a,b,totals))
        self.assertFalse(stop.all_base_terminal(a,{**b,'status':'running'},totals))
        self.assertFalse(stop.all_base_terminal(a,b,{**totals,'completed':199,'running':1}))
        self.assertFalse(stop.all_base_terminal(a,b,{**totals,'planned':1000,'pending':800}))
        self.assertFalse(stop.all_base_terminal({**a,'planned':173},b,totals))

    def test_refuses_wrong_instance_without_issuing_api_call(self):
        with patch.dict('os.environ',{'CONTAINER_ID':'other','CONTAINER_API_KEY':'test-only'},clear=True),patch.object(stop.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'identity'):stop.own_stop('node-a',Path('/unused'))
            run.assert_not_called()

    def test_stop_uses_only_own_credential_and_redacts_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            status=Path(tmp)/'status.json'
            with patch.dict('os.environ',{'CONTAINER_ID':stop.IDS['node-b'],'CONTAINER_API_KEY':'local-test-only'},clear=True),patch.object(stop.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='stopping instance 54533396.')) as run:
                stop.own_stop('node-b',status)
            self.assertEqual(run.call_args.args[0][:4],['vastai','stop','instance','54533396'])
            data=json.loads(status.read_text());self.assertEqual(data['status'],'stop_request_accepted')
            self.assertNotIn('local-test-only',status.read_text())


if __name__=='__main__':unittest.main()
