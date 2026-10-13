import copy
import unittest

from safetyjev.codex_dispatch_audit import (DENIED_PROBES, DIRECT_PROBES, ROBOT_NAMES,
    WRAPPERS, EMPTY_LISTINGS, assess, rejected_output, restricted_catalog, validate_report)
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

    def test_approved_exception_is_empty_listing_only(self):
        import json
        evidence=self.evidence()
        for row in evidence[2]:
            if row['tool'] in EMPTY_LISTINGS:
                row.update(rejected=False,result=json.dumps(EMPTY_LISTINGS[row['tool']]))
        self.assertEqual(assess(*evidence)['status'],'blocked_tool_isolation')
        self.assertEqual(assess(*evidence,allow_empty_resources=True)['status'],'passed')
        for payload in ('{"resources":[{"uri":"private"}]}','{"resources":[],"secret":1}',
                        '{"resourceTemplates":[]}','null','not json'):
            changed=copy.deepcopy(evidence)
            next(r for r in changed[2] if r['tool']=='list_mcp_resources')['result']=payload
            self.assertEqual(assess(*changed,allow_empty_resources=True)['status'],'blocked_tool_isolation')
        changed=copy.deepcopy(evidence)
        next(r for r in changed[2] if r['tool']=='read_mcp_resource').update(rejected=False,result='{}')
        self.assertEqual(assess(*changed,allow_empty_resources=True)['status'],'blocked_tool_isolation')

    def test_dispatch_attestation_requires_complete_audit(self):
        from safetyjev.codex_worker import validate_attestation, SDK_VERSION, DISABLED_FEATURES
        e=self.evidence()
        report=assess(*e,allow_empty_resources=True)
        report.update(allow_empty_resources=True,sdk_version=SDK_VERSION,runtime_version=SDK_VERSION,
            model='gpt-6-astra',native_inventory=e[0],nested_audit=e[1],direct_attempts=e[2],
            robot_stub_calls=e[3],mcp_servers=[{'name':'robot','tools':list(TOOL_NAMES),
                'resource_count':0,'resource_template_count':0}])
        value={'sdk_version':SDK_VERSION,'runtime_version':SDK_VERSION,'model':'gpt-6-astra',
            'mcp_tools':list(TOOL_NAMES),'native_inventory':list(WRAPPERS),'isolation':report,
            'disabled_features':[k for k in DISABLED_FEATURES if k not in ('code_mode','code_mode_host')]}
        validate_attestation(value)
        for key in ('nested','resources','status','policy'):
            changed=copy.deepcopy(value)
            if key=='nested':changed['isolation']['nested_audit']['names'].append('exec_command')
            elif key=='resources':changed['isolation']['mcp_servers'][0]['resource_count']=1
            elif key=='status':changed['isolation']['status']='blocked_tool_isolation'
            else:changed['isolation']['allow_empty_resources']=False
            with self.assertRaises(ValueError):validate_attestation(changed)

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
