"""Optional RPent facade protocol tests; upstream simulator/planner never loads."""
from dataclasses import dataclass, field
import json
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from safetyjev.rpent_adapter import make_toolkit


@dataclass
class Result:
    data: dict=field(default_factory=dict)
    images: list=field(default_factory=list)


class NativeTool:
    def __init__(self, **kwargs):self.__dict__.update(kwargs)


class BaseToolkit:
    def __init__(self, **kwargs):
        self._tools={};self._register_common_tools()
    def _register_common_tools(self):raise AssertionError('Common file/memory tools must not load')
    def add_tool(self,tool):self._tools[tool.name]=tool
    def list_tools(self):return tuple(self._tools.values())


class RPentFacadeTests(unittest.TestCase):
    def setUp(self):
        modules={name:ModuleType(name) for name in ('rpent','rpent.tools','rpent.tools.base','rpent.tools.toolkit')}
        modules['rpent.tools.base'].Tool=NativeTool;modules['rpent.tools.base'].ToolResult=Result
        modules['rpent.tools.toolkit'].Toolkit=BaseToolkit
        self.patch=patch.dict(sys.modules,modules);self.patch.start();self.addCleanup(self.patch.stop)
        self.toolkit=make_toolkit('/not-a-real-socket')

    def test_only_robot_interface_no_common_tools_or_outcomes(self):
        self.assertEqual([t.name for t in self.toolkit.list_tools()],['observe','run_skill'])
        self.assertFalse(self.toolkit.solved())
        with patch('safetyjev.rpent_adapter.call') as call:
            for name in ('read_text_file','list_dir','finish','execute','teleport','config'):
                self.assertEqual(self.toolkit.execute_tool(name,{}).data,{'error':'unknown_tool'})
            call.assert_not_called()

    def test_binary_images_and_errors_without_host_exceptions(self):
        content=[{'type':'text','text':json.dumps({'task':'Move jar'})},
                 {'type':'text','text':'overview camera'}, {'type':'image','data':'AA=='}]
        with patch('safetyjev.rpent_adapter.call',return_value={'content':content}) as call:
            r=self.toolkit.execute_tool('observe',{});self.assertEqual(r.data,{'task':'Move jar'});self.assertEqual(r.images,[b'\0'])
            call.assert_called_once_with('/not-a-real-socket','observe',{},'skills')
        with patch('safetyjev.rpent_adapter.call',side_effect=RuntimeError('PRIVATE_CANARY')):
            self.assertEqual(self.toolkit.execute_tool('observe',{}).data,{'error':'robot_tool_unavailable'})

    def test_cancellation_closes_without_simulator_or_memory_access(self):
        with patch('safetyjev.rpent_adapter.rpc') as rpc:
            self.toolkit.close();rpc.assert_called_once_with('/not-a-real-socket','_cancel')
        self.assertEqual(self.toolkit.execute_tool('observe',{}).data,{'status':'session_closed'})
        with self.assertRaisesRegex(RuntimeError,'fresh episode'):self.toolkit.resume_calls()
