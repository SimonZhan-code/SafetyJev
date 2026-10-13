"""Inspect the actual pinned CLI request tools against a local non-model endpoint."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def toml(value):
    if isinstance(value, dict):
        return '{'+', '.join(json.dumps(k)+' = '+toml(v) for k,v in value.items())+'}'
    if isinstance(value, list):
        return '['+', '.join(toml(v) for v in value)+']'
    return json.dumps(value)


def tool_names(tools, prefix=''):
    names = []
    for tool in tools:
        if tool.get('type') == 'namespace':
            names.extend(tool_names(tool.get('tools', []), tool['name']+'.'))
        else:
            name = tool.get('name') or tool.get('function', {}).get('name')
            names.append(prefix+name if name else '__unsupported_tool_type__'+str(tool.get('type')))
    return [n.replace('mcp__robot.', 'mcp__robot__') for n in names]


def probe_inventory(config, model):
    from openai_codex.client import CodexConfig, _resolve_codex_bin
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size < 8*1024*1024:
                    raise ValueError()
                raw = self.rfile.read(size)
                if self.headers.get('Content-Encoding') == 'zstd':
                    import zstandard
                    raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=8*1024*1024)
                data = json.loads(raw)
                tools = list(data.get('tools', []))
                for item in data.get('input', []):
                    if isinstance(item, dict) and item.get('type') == 'additional_tools':
                        tools.extend(item['tools'])
                captured.append(tool_names(tools))
            except Exception:
                captured.append(['__invalid_request__'])
            # No model is called and no candidate tools execute during this probe.
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"inventory probe complete","type":"invalid_request_error"}}')

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='codex-inventory-') as temp:
            root = Path(temp)
            (root/'work').mkdir()
            config = copy.deepcopy(config)
            config.update(model=model, model_provider='inventory', approval_policy='never',
                          sandbox_mode='read-only')
            config['features']['enable_request_compression'] = False
            config['model_providers'] = {'inventory': {'name': 'Local inventory only',
                'base_url': 'http://127.0.0.1:'+str(server.server_port)+'/v1',
                'wire_api': 'responses', 'requires_openai_auth': False,
                'request_max_retries': 0, 'stream_max_retries': 0}}
            (root/'config.toml').write_text('\n'.join(k+' = '+toml(v) for k,v in config.items()))
            env = {'PATH': '/usr/bin:/bin', 'HOME': str(root), 'CODEX_HOME': str(root),
                   'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
            subprocess.run([str(_resolve_codex_bin(CodexConfig())), 'exec', '--skip-git-repo-check',
                '-C', str(root/'work'), 'Inventory probe; do not call any tools.'],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=35)
        if not captured or any(names != captured[0] for names in captured):
            raise RuntimeError('tool_inventory_unavailable')
        return sorted(captured[0])
    finally:
        server.shutdown()
        server.server_close()
