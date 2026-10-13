import copy
import unittest

from safetyjev.codex_dispatch_audit import (DENIED_PROBES, DIRECT_PROBES, ROBOT_NAMES,
    WRAPPERS, assess, rejected_output, restricted_catalog)
from safetyjev.codex_tools import TOOL_NAMES


class DispatchAuditTests(unittest.TestCase):
    def evidence(self):
        return [list(WRAPPERS), {'names':list(ROBOT_NAMES), 'keys':list(ROBOT_NAMES),
            'globals':{key:'undefined' for key in ('process','require','fetch','Deno','WebSocket','XMLHttpRequest')},
            'denied':{key:True for key in DENIED_PROBES}},
            [{'tool':name,'rejected':True} for name in DIRECT_PROBES], list(TOOL_NAMES), True]

    def test_complete_evidence_and_bypasses(self):
        self.assertEqual(assess(*self.evidence())['status'], 'passed')
        for kind in ('nested', 'direct', 'missing_probe', 'missing_direct', 'globals', 'canary', 'robot'):
            evidence = self.evidence()
            if kind == 'nested': evidence[1]['names'].append('exec_command')
            elif kind == 'direct': evidence[2][0]['rejected'] = False
            elif kind == 'missing_probe': evidence[1]['denied'].pop('node_fs')
            elif kind == 'missing_direct': evidence[2].pop()
            elif kind == 'globals': evidence[1]['globals']['fetch'] = 'function'
            elif kind == 'canary': evidence[4] = False
            elif kind == 'robot': evidence[3].pop()
            with self.subTest(kind=kind):
                self.assertEqual(assess(*evidence)['status'], 'blocked_tool_isolation')

    def test_empty_resource_success_is_not_rejection(self):
        for value in ('{"resources":[]}', '{"resourceTemplates":[]}', '{"text":"error"}', None, []):
            self.assertFalse(rejected_output(value))
        for value in ('unsupported call: exec_command', 'patch rejected: read-only',
                      'resources/read failed: Method not found'):
            self.assertTrue(rejected_output(value))

    def test_catalog_change_only_removes_optional_tools(self):
        source = {'models':[{'slug':'gpt-6-astra','tool_mode':'code_mode_only',
            'base_instructions':'stock instructions', 'experimental_supported_tools':['clock','send_user_message_async']},
            {'slug':'other','experimental_supported_tools':['clock']}]}
        original = copy.deepcopy(source)
        result = restricted_catalog(source)
        expected = copy.deepcopy(source); expected['models'][0]['experimental_supported_tools'] = []
        self.assertEqual(result, expected)
        self.assertEqual(source, original)
        with self.assertRaises(ValueError): restricted_catalog({'models':[]})


if __name__ == '__main__': unittest.main()
