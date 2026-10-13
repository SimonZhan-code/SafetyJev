"""Private Unix transport. Simulator operations run only on its owning thread."""
import json
import os
from pathlib import Path
import queue
import socket
import socketserver
import struct
import threading

MAX_MESSAGE = 8 * 1024 * 1024


def rpc(path, method, arguments=None):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(150)
        sock.connect(path)
        sock.sendall(json.dumps({'method': method, 'arguments': {} if arguments is None else arguments}, allow_nan=False).encode()+b'\n')
        with sock.makefile('rb') as stream:
            data = stream.readline(MAX_MESSAGE+1)
        if not data.endswith(b'\n') or len(data) > MAX_MESSAGE:
            raise RuntimeError('invalid_bridge_response')
        return json.loads(data)


class Bridge:
    def __init__(self, path, uid, gid):
        self.pending = queue.Queue(maxsize=16)
        self.closed = False
        self.path = Path(path)
        owner = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    if hasattr(socket, 'SO_PEERCRED'):
                        _, peer_uid, _ = struct.unpack('3i', self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if peer_uid != uid:
                            return
                    self.request.settimeout(150)
                    raw = self.rfile.readline(MAX_MESSAGE+1)
                    if not raw.endswith(b'\n') or len(raw) > MAX_MESSAGE:
                        return
                    value = json.loads(raw)
                    if not isinstance(value, dict) or set(value) != {'method', 'arguments'}:
                        return
                    if owner.closed:
                        result = {'status': 'session_closed'}
                    else:
                        reply = queue.Queue(maxsize=1)
                        owner.pending.put((value, reply), timeout=1)
                        result = reply.get(timeout=145)
                    self.wfile.write(json.dumps(result, allow_nan=False).encode()+b'\n')
                except Exception:
                    # Transport errors intentionally reveal no filesystem or request details.
                    return

        class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.server = Server(str(path), Handler)
        os.chown(path, 0, gid)
        os.chmod(path, 0o660)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.closed = True
        while True:
            try:
                _, reply = self.pending.get_nowait()
                reply.put({'status': 'session_closed'})
            except queue.Empty:
                break

    def shutdown(self):
        self.close()
        self.server.shutdown()
        self.server.server_close()
        self.path.unlink(missing_ok=True)
