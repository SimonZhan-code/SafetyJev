"""Optional RPent Toolkit facade over the restricted skill gateway.

Interface reference: RLinf/RPent a3340f46b00755dae419e1350a6740829534064c.
No RPent planner, simulator, CommonTools, filesystem tools or persistent memory
are instantiated. Install RPent separately to use this optional facade. The
production Codex path continues to use the audited SDK/MCP adapter directly.
"""
import base64
import json
import threading
from typing import Literal

from .codex_bridge import rpc
from .codex_tools import call, inventory

RPENT_REFERENCE = 'a3340f46b00755dae419e1350a6740829534064c'


def make_toolkit(socket_path):
    from pydantic import BaseModel, ConfigDict, create_model
    from rpent.tools.base import Tool, ToolResult
    from rpent.tools.toolkit import Toolkit

    class Empty(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)

    skill_args = create_model('SafetyJevSkillArgs',
        __config__=ConfigDict(extra='forbid', strict=True, allow_inf_nan=False),
        request_id=(str, ...), decision_id=(str, ...),
        kind=(Literal['vla', 'hold_joints', 'joint_target'], ...), instruction=(str, ...),
        joint_positions=(list[float], ...), gripper=(Literal[-1, 1] | None, ...), max_chunks=(int, ...))

    class SafetyJevToolkit(Toolkit):
        def _register_common_tools(self):
            pass  # RPent defaults include file tools and memory; do not register them.

        def __init__(self):
            super().__init__(dashboard_events=None, memory=None)
            self._lock = threading.RLock()
            self._closed = False
            for item in inventory('skills'):
                name = item['name']
                self.add_tool(Tool(name=name, description=item['description'],
                    args_schema=Empty if name == 'observe' else skill_args,
                    handler=lambda _name=name, **args: self.execute_tool(_name, args),
                    _input_schema=item['inputSchema'], readonly=name == 'observe'))

        def execute_tool(self, name, input_dict):
            # Override RPent's implicit environment capture / exception tracebacks.
            with self._lock:
                if self._closed:
                    return ToolResult(data={'status': 'session_closed'})
                if name not in ('observe', 'run_skill'):
                    return ToolResult(data={'error': 'unknown_tool'})
                try:
                    self._tools[name].args_schema.model_validate(input_dict, strict=True)
                    result = call(socket_path, name, input_dict, 'skills')
                    data, images = {}, []
                    for block in result['content']:
                        if block['type'] == 'image':
                            images.append(base64.b64decode(block['data'], validate=True))
                        elif block['type'] == 'text' and block['text'].startswith('{'):
                            data = json.loads(block['text'])
                    return ToolResult(data=data, images=images)
                except Exception:
                    return ToolResult(data={'error': 'robot_tool_unavailable'})

        def solved(self):
            return False  # Private evaluator alone determines benchmark outcomes.

        def get_env_state(self, **_):
            return self.execute_tool('observe', {})

        def cancel_active_and_wait(self):
            self._closed = True
            rpc(socket_path, '_cancel')
            with self._lock:
                pass

        def resume_calls(self):
            raise RuntimeError('Start a fresh episode after cancellation')

        def close(self):
            self.cancel_active_and_wait()

    return SafetyJevToolkit()
