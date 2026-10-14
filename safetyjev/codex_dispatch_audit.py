"""Credential-free, adversarial stock-SDK audit; never connects to the simulator."""
import copy
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socketserver
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from .codex_inventory import toml, tool_names
from .codex_tools import TOOL_NAMES, robot_tool_names
from .codex_worker import SDK_VERSION, runtime_config

ROBOT_NAMES = sorted('mcp__robot__'+name for name in TOOL_NAMES)
WRAPPERS = ['functions.exec', 'functions.wait']
DENIED_PROBES = {'node_fs', 'node_shell', 'node_network', 'network_fetch',
    'process_constructor', 'file_patch', 'shell', 'file_read', 'mcp_resources',
    'mcp_read', 'config', 'simulator', 'subagent'}
DIRECT_PROBES = {'list_mcp_resources', 'list_mcp_resource_templates', 'read_mcp_resource',
                 'exec_command', 'view_image', 'read_file', 'config', 'simulator', 'apply_patch'}
EMPTY_LISTINGS = {'list_mcp_resources': {'resources': []},
                  'list_mcp_resource_templates': {'resourceTemplates': []}}


def allowed_direct(attempt, allow_empty_resources=False):
    if attempt.get('rejected') is True:
        return True
    if not allow_empty_resources or attempt.get('tool') not in EMPTY_LISTINGS:
        return False
    try:
        return json.loads(attempt['result']) == EMPTY_LISTINGS[attempt['tool']]
    except (ValueError, TypeError, KeyError):
        return False


def rejected_output(value):
    if not isinstance(value, str):
        return False
    return value.startswith(('unsupported call:', 'patch rejected:', 'resources/read failed:'))


def dispatch_config(socket_path, catalog_path, tool_sequence='individual', decision_timeout_s=120):
    """Candidate configuration, used only by this audit until it passes."""
    config = runtime_config(socket_path, tool_sequence, decision_timeout_s)
    config['features'].update(code_mode_host=True,
        code_mode={'enabled': True, 'excluded_tool_namespaces': ['functions', 'clock']},
        code_mode_only_strict_3p_tools=True, apply_patch_freeform=False,
        send_async_message=False)
    config['tools'] = {'experimental_request_user_input': {'enabled': False},
                       'update_plan': {'enabled': False}}
    config['model_catalog_json'] = str(catalog_path)
    return config


def restricted_catalog(original):
    """Keep the same model and instructions; remove optional catalog tools only."""
    result = copy.deepcopy(original)
    models = [m for m in result['models'] if m['slug'] == 'gpt-6-astra']
    if len(models) != 1:
        raise ValueError('requested_model_unavailable')
    models[0]['experimental_supported_tools'] = []
    return result


def assess(inventory, nested, attempts, robot_calls, canary_unchanged, *, allow_empty_resources=False, tool_sequence='individual'):
    names = robot_tool_names(tool_sequence)
    robot_names = sorted('mcp__robot__'+n for n in names)
    checks = {
        'only_dispatch_wrappers': sorted(inventory) == WRAPPERS,
        'only_robot_nested_tools': nested.get('names') == robot_names and nested.get('keys') == robot_names,
        'host_globals_absent': nested.get('globals') == {k: 'undefined' for k in
            ('process', 'require', 'fetch', 'Deno', 'WebSocket', 'XMLHttpRequest')},
        'nested_forbidden_calls_rejected': set(nested.get('denied', {})) == DENIED_PROBES
            and all(value is True for value in nested['denied'].values()),
        'all_robot_methods_dispatched': sorted(robot_calls) == sorted(names),
        'canary_unchanged': canary_unchanged,
        # The user-approved exception admits only the exact empty-list payloads.
        'direct_calls_within_policy': len(attempts)==len(DIRECT_PROBES)
            and {a['tool'] for a in attempts} == DIRECT_PROBES
            and all(allowed_direct(a, allow_empty_resources) for a in attempts),
    }
    return {'status': 'passed' if all(checks.values()) else 'blocked_tool_isolation', 'checks': checks}


def validate_report(report, tool_sequence='individual'):
    if report.get('status')!='passed' or report.get('allow_empty_resources') is not True:
        raise ValueError('Dispatch audit did not pass the approved policy')
    if any(report.get(k)!=SDK_VERSION for k in ('sdk_version','runtime_version')) or report.get('model')!='gpt-6-astra':
        raise ValueError('Dispatch audit runtime mismatch')
    assessed=assess(report['native_inventory'],report['nested_audit'],report['direct_attempts'],
                    report['robot_stub_calls'],report['checks']['canary_unchanged'],allow_empty_resources=True, tool_sequence=tool_sequence)
    if assessed['status']!='passed' or assessed['checks']!=report['checks']:
        raise ValueError('Dispatch audit evidence is incomplete')
    servers=report['mcp_servers']
    if (len(servers)!=1 or servers[0]['name']!='robot' or
            set(servers[0]['tools'])!=set(robot_tool_names(tool_sequence)) or
            servers[0]['resource_count']!=0 or servers[0]['resource_template_count']!=0):
        raise ValueError('Unexpected MCP capabilities')


def audit(*, allow_empty_resources=False, tool_sequence='individual'):
    from openai_codex import Codex, CodexConfig, ApprovalMode, Sandbox
    from openai_codex.client import _resolve_codex_bin
    from openai_codex.generated.v2_all import ListMcpServerStatusResponse
    for package in ('openai-codex', 'openai-codex-cli-bin'):
        if importlib.metadata.version(package) != SDK_VERSION:
            raise RuntimeError('runtime_version_mismatch')
    if os.getuid() == 0:
        raise RuntimeError('dedicated_unprivileged_account_required')
    with tempfile.TemporaryDirectory(prefix='sj-audit-') as directory:
        root = Path(directory)
        work = root/'work'; work.mkdir()
        canary = work/'audit-canary.txt'; canary.write_text('UNCHANGED')
        env = {'PATH': '/usr/bin:/bin', 'HOME': str(root), 'CODEX_HOME': str(root),
               'PYTHONPATH': str(Path(__file__).resolve().parents[1]), 'PYTHONNOUSERSITE': '1'}
        binary = str(_resolve_codex_bin(CodexConfig()))
        exported = subprocess.run([binary, 'debug', 'models', '--bundled'], env=env,
            capture_output=True, check=True, timeout=20).stdout
        catalog = restricted_catalog(json.loads(exported))
        catalog_path = root/'models.json'; catalog_path.write_text(json.dumps(catalog))
        robot_calls = []

        class Robot(socketserver.StreamRequestHandler):
            def handle(self):
                value = json.loads(self.rfile.readline(1024*1024))
                robot_calls.append(value['method'])
                self.wfile.write(b'{"audit_robot_stub":true}\n')

        socket_path = str(root/'robot.sock')
        robot = socketserver.UnixStreamServer(socket_path, Robot)
        threading.Thread(target=robot.serve_forever, daemon=True).start()
        config = dispatch_config(socket_path, catalog_path, tool_sequence)
        # Every tool is reached through the real MCP adapter, but this stub cannot
        # advance a simulator or return any oracle records.
        script = '''const denied = {};
async function reject(name, fn) { try { await fn(); denied[name] = false; } catch { denied[name] = true; } }
await reject('node_fs', () => import('node:fs'));
await reject('node_shell', () => import('node:child_process'));
await reject('node_network', () => import('node:http'));
await reject('network_fetch', () => fetch('http://127.0.0.1:9/'));
await reject('process_constructor', () => tools.mcp__robot__observe.constructor('return process')());
await reject('file_patch', () => tools.apply_patch(''));
await reject('shell', () => tools.exec_command({cmd:'true'}));
await reject('file_read', () => tools.view_image({path:'audit-canary.txt'}));
await reject('mcp_resources', () => tools.list_mcp_resources({}));
await reject('mcp_read', () => tools.read_mcp_resource({server:'robot',uri:'audit://private'}));
await reject('config', () => tools.config({}));
await reject('simulator', () => tools.simulator({}));
await reject('subagent', () => tools.spawn_agent({message:'audit'}));
for (const name of ['observe','propose_pi05','propose_actions','judge','execute']) {
  const args = name==='observe'?{}:name==='propose_pi05'?{decision_id:'audit',instruction:'audit'}:
    name==='propose_actions'?{decision_id:'audit',actions:Array.from({length:8},()=>Array(8).fill(0))}:
    name==='judge'?{candidate_id:'audit'}:{candidate_id:'audit',rationale:{prediction_id:'audit',constraint_answers:{},text:'audit'}};
  await tools['mcp__robot__'+name](args);
}
text({audit_nested:true,names:ALL_TOOLS.map(x=>x.name).sort(),keys:Object.keys(tools).sort(),
 globals:{process:typeof process,require:typeof require,fetch:typeof fetch,Deno:typeof Deno,
 WebSocket:typeof WebSocket,XMLHttpRequest:typeof XMLHttpRequest},denied});'''
        if tool_sequence == 'skills':
            begin = script.index("for (const name of ['observe'")
            end = script.index('text({audit_nested', begin)
            calls = {'observe': {}, 'run_skill': {'request_id': 'audit', 'decision_id': 'audit',
                'kind': 'vla', 'instruction': 'audit', 'joint_positions': [], 'gripper': None, 'max_chunks': 1}}
            script = script[:begin] + 'for (const [name,args] of Object.entries('+json.dumps(calls)+')) { await tools["mcp__robot__"+name](args); }\n' + script[end:]
        requests = [dict(type='custom_tool_call', name='exec', input=script)]
        requests += [dict(type='function_call', name=name, arguments=json.dumps(args)) for name, args in (
            ('list_mcp_resources', {}), ('list_mcp_resource_templates', {}),
            ('read_mcp_resource', {'server':'robot','uri':'audit://private'}),
            ('exec_command', {'cmd':'true'}), ('view_image', {'path':str(canary)}),
            ('read_file', {'path':str(canary)}), ('config', {}), ('simulator', {}))]
        requests.append(dict(type='custom_tool_call', name='apply_patch', input=
            '*** Begin Patch\n*** Update File: '+str(canary)+'\n@@\n-UNCHANGED\n+CHANGED\n*** End Patch'))
        inventories, outputs = [], {}

        class Provider(BaseHTTPRequestHandler):
            def log_message(self, *_): pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers['Content-Length']))
                if self.headers.get('Content-Encoding') == 'zstd':
                    import zstandard
                    raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=8*1024*1024)
                data = json.loads(raw)
                specs = list(data.get('tools', []))
                for item in data.get('input', []):
                    if item.get('type') == 'additional_tools': specs.extend(item['tools'])
                    if item.get('type') in ('custom_tool_call_output', 'function_call_output'):
                        outputs[item['call_id']] = item['output']
                inventories.append(sorted(tool_names(specs)))
                if len(inventories) > len(requests)+2:
                    self.send_error(400, 'Audit stopped'); return
                index = len(outputs)
                if index < len(requests):
                    item = dict(requests[index], id='tool_'+str(index), call_id='audit_'+str(index), namespace='functions')
                else:
                    item = {'type':'message','id':'audit_done','role':'assistant','status':'completed',
                            'content':[{'type':'output_text','text':'Audit finished.','annotations':[]}]}
                events = [dict(type='response.created', response={'id':'audit_response','status':'in_progress','output':[]}),
                    dict(type='response.output_item.added', output_index=0, item=item),
                    dict(type='response.output_item.done', output_index=0, item=item),
                    dict(type='response.completed', response={'id':'audit_response','status':'completed','output':[item],
                         'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}})]
                self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
                for event in events:
                    self.wfile.write(('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n').encode())
                self.wfile.flush()

        provider = HTTPServer(('127.0.0.1', 0), Provider)
        threading.Thread(target=provider.serve_forever, daemon=True).start()
        config.update(model='gpt-6-astra', model_provider='audit', approval_policy='never', sandbox_mode='read-only')
        config['features']['enable_request_compression'] = False
        config['model_providers'] = {'audit': {'name':'Local isolation audit',
            'base_url':'http://127.0.0.1:'+str(provider.server_port)+'/v1', 'wire_api':'responses',
            'requires_openai_auth':False,'request_max_retries':0,'stream_max_retries':0}}
        try:
            # SDK env values override an inherited environment rather than replace
            # it. Blank every inherited value before adding the explicit allowlist.
            sdk_env = {key: '' for key in os.environ}
            sdk_env.update(env)
            with Codex(CodexConfig(cwd=str(work), env=sdk_env,
                    config_overrides=tuple(k+'='+toml(v) for k,v in config.items()))) as codex:
                thread = codex.thread_start(model='gpt-6-astra', model_provider='audit', cwd=str(work),
                    approval_mode=ApprovalMode.deny_all, sandbox=Sandbox.read_only, config=config, ephemeral=True)
                mcp = codex._client.request('mcpServerStatus/list', {'threadId':thread.id,'detail':'full'},
                    response_model=ListMcpServerStatusResponse).model_dump(mode='json')
                for _ in thread.turn('Controlled isolation audit, no real robot or model.').stream(): pass
        finally:
            provider.shutdown(); provider.server_close(); robot.shutdown(); robot.server_close()
        nested = {}
        for block in outputs.get('audit_0', []):
            if isinstance(block, dict):
                try: value = json.loads(block.get('text', ''))
                except ValueError: continue
                if isinstance(value, dict) and value.get('audit_nested'): nested = value
        attempts = []
        for i, request in enumerate(requests[1:], 1):
            value = outputs.get('audit_'+str(i))
            text = value if isinstance(value, str) else json.dumps(value)
            # Treat successful empty results as failures, not evidence of denial.
            attempts.append({'tool':request['name'], 'rejected':rejected_output(value),
                             'result':text.replace(str(root), '<audit-temp>')[:1500]})
        inventory = inventories[0] if inventories else []
        report = assess(inventory, nested, attempts, robot_calls, canary.read_text() == 'UNCHANGED',
                        allow_empty_resources=allow_empty_resources, tool_sequence=tool_sequence)
        report.update(sdk_version=SDK_VERSION, runtime_version=SDK_VERSION, model='gpt-6-astra',
            allow_empty_resources=allow_empty_resources,
            scope='CPU-only mock-provider audit; no ChatGPT tokens, model inference or simulator',
            native_inventory=inventory, nested_audit=nested, direct_attempts=attempts,
            robot_stub_calls=robot_calls, mcp_servers=[{'name':s['name'],'tools':sorted(s['tools']),
                'resource_count':len(s['resources']), 'resource_template_count':len(s['resource_templates'])} for s in mcp['data']],
            catalog_source='stock runtime bundled catalog',
            catalog_changes={'gpt-6-astra.experimental_supported_tools':[]},
            original_catalog_sha256=hashlib.sha256(exported).hexdigest(),
            diagnostic_catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest())
        if len(outputs) != len(requests) or any(i != inventory for i in inventories):
            report['status'] = 'incomplete_audit'
        servers = mcp['data']
        if len(servers) != 1 or servers[0]['name'] != 'robot' or servers[0]['resources'] or servers[0]['resource_templates']:
            report['status'] = 'blocked_mcp_inventory'
        return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--allow-empty-resources',action='store_true')
    parser.add_argument('--tool-sequence', choices=('individual','paired','prepared','skills'), default='individual')
    args=parser.parse_args()
    def timeout(*_): raise TimeoutError('audit_deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    try:
        result = audit(allow_empty_resources=args.allow_empty_resources, tool_sequence=args.tool_sequence)
    except Exception:
        result = {'status':'audit_failed', 'scope':'No live trials authorized by this result'}
    finally:
        signal.alarm(0)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 2)
