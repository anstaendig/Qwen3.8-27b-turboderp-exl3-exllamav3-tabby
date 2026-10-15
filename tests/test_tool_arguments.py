"""CPU-only Qwen schema conversion and real stream collector parity."""
import ast
import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

TABBY = Path(__file__).resolve().parents[1] / 'tabbyAPI'


@pytest.fixture
def parser(monkeypatch):
    # Avoid package initializers that load GPU backends; execute the real parser modules.
    for prefix in ('common', 'endpoints', 'endpoints.OAI', 'endpoints.OAI.types',
                   'endpoints.OAI.utils', 'endpoints.OAI.utils.toolcall_formats'):
        module = types.ModuleType(prefix)
        module.__path__ = [str(TABBY / prefix.replace('.', '/'))]
        monkeypatch.setitem(sys.modules, prefix, module)
    logger = SimpleNamespace(debug=lambda *a, **kw: None, info=lambda *a, **kw: None,
                             error=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, 'common.logger', SimpleNamespace(xlogger=logger))
    # Track imports so monkeypatch restores the complete module namespace after each test.
    before = set(sys.modules)
    tooltypes = importlib.import_module('endpoints.OAI.types.tools')
    tools = importlib.import_module('endpoints.OAI.utils.tools')
    stream = importlib.import_module('endpoints.OAI.utils.stream_parser')
    tree = ast.parse((TABBY / 'endpoints/OAI/utils/chat_completion.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in ('_parse_tool_calls', '_chat_stream_collector')]
    ns = dict(vars(stream), asyncio=asyncio, xlogger=logger, parse_toolcalls=tools.parse_toolcalls,
              get_toolcall_tags=tools.get_toolcall_tags,
              _resolve_reasoning_budget=lambda *a: (None, ''), model=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                            type_ignores=[])), 'collector', 'exec'), ns)
    yield tooltypes.ToolSpec, tools.parse_toolcalls, ns
    for name in set(sys.modules) - before:
        if name.startswith(('endpoints.', 'common.')):
            sys.modules.pop(name, None)


CASES = [
    ('boolean', 'False', False), ('boolean', 'false', False), ('boolean', 'TRUE', True),
    ('boolean', 'maybe', 'maybe'), ('boolean', '0', '0'), ('boolean', 'null', 'null'),
    ('integer', '10', 10), ('integer', '10.0', 10), ('integer', '-3', -3),
    ('integer', '10.5', '10.5'), ('integer', '1.00000000000000001', '1.00000000000000001'),
    ('integer', '1e100000', '1e100000'),
    ('integer', '1e9999999999999999999999999', '1e9999999999999999999999999'),
    ('number', '10', 10), ('number', '10.5', 10.5), ('number', '-3', -3),
    ('number', 'true', 'true'), ('number', '1e999', '1e999'),
    ('string', '123', '123'), ('string', 'true', 'true'), ('string', '  true  ', '  true  '),
    (['string'], '123', '123'), (['boolean', 'null'], 'None', None),
    (['null', 'boolean'], 'FALSE', False), ('null', 'none', None), ('null', '10', '10'),
    ('array', "['a', 'b']", ['a', 'b']), ('object', "{'k': True}", {'k': True}),
    ('array', '[true, null]', [True, None]), ('array', '{}', '{}'), ('object', '[]', '[]'),
    ('array', '[1, (2, 3)]', '[1, (2, 3)]'), ('object', "{1: 'x'}", "{1: 'x'}"),
    ('object', "{'x': __import__('os')}", "{'x': __import__('os')}"),
    (None, 'True', True), (None, 'False', False), (None, 'None', None),
    (None, '123', 123), ('unknown', '123', '123'),
]


@pytest.mark.parametrize('declared,value,expected', CASES)
def test_schema_conversion(parser, declared, value, expected):
    ToolSpec, parse, _ = parser
    prop = {} if declared is None else {'type': declared}
    tools = [ToolSpec(type='function', function={'name': 'f', 'parameters': {'properties': {'p': prop}}})]
    text = f'<function=f><parameter=p>\n{value}\n</parameter></function>'
    for fmt in ('qwen3_coder', 'qwen3_5', 'step3_5'):
        result = json.loads(parse(text, fmt, tools=tools)[0].function.arguments)
        assert result == {'p': expected}
        assert type(result['p']) is type(expected)


def test_unknown_function_parameter_and_other_format(parser):
    ToolSpec, parse, _ = parser
    tools = [ToolSpec(type='function', function={'name': 'f', 'parameters': {'properties': {'p': {'type': 'string'}}}})]
    for name, key in [('f', 'unknown'), ('other', 'p')]:
        text = f'<function={name}><parameter={key}>True</parameter></function>'
        assert json.loads(parse(text, 'qwen3_coder', tools)[0].function.arguments) == {key: True}
    text = '<function=f><parameter=p>true</parameter></function>'
    assert json.loads(parse(text, 'qwen3_coder', [t.model_dump() for t in tools])[0].function.arguments) == {'p': 'true'}
    text = '<tool_call>{"name":"f","arguments":{"p":true}}</tool_call>'
    assert json.loads(parse(text, 'hermes', tools)[0].function.arguments) == {'p': True}


@pytest.mark.parametrize('step', [1, 7, 10000])
def test_real_collector_stream_nonstream_parity(parser, step):
    ToolSpec, _, ns = parser
    properties = {f'p{i}': ({} if t is None else {'type': t}) for i, (t, _, _) in enumerate(CASES)}
    tools = [ToolSpec(type='function', function={'name': 'f', 'parameters': {'properties': properties}})]
    text = '<tool_call><function=f>' + ''.join(
        f'<parameter=p{i}>\n{v}\n</parameter>' for i, (_, v, _) in enumerate(CASES)
    ) + '</function></tool_call>'
    async def generate(*a, **kw):
        for i in range(0, len(text), step):
            yield {'text': text[i:i+step]}
        yield {'text': '', 'finish_reason': 'stop'}
    container = SimpleNamespace(harmony=False, muse_glimmer=False, tool_format='qwen3_coder',
                                reasoning=False, reasoning_start_token=None, reasoning_end_token=None,
                                tool_calls_in_reasoning=False, stream_generate=generate)
    ns['model'] = SimpleNamespace(container=container)
    params = SimpleNamespace(tools=tools, tool_choice='auto', json_schema=None)
    async def run():
        queue = asyncio.Queue()
        result = await ns['_chat_stream_collector'](0, None, 'test', '', params, False, streaming_mode=False)
        assert isinstance(result, dict), result
        await ns['_chat_stream_collector'](0, queue, 'test', '', params, False)
        chunks = []
        while not queue.empty():
            item = queue.get_nowait()
            assert isinstance(item, dict), item
            if item.get('delta_tool_calls'):
                chunks.extend(item['delta_tool_calls'])
        return result['tool_calls'], chunks
    direct, streamed = asyncio.run(run())
    expected = {f'p{i}': v for i, (_, _, v) in enumerate(CASES)}
    for calls in (direct, streamed):
        assert len(calls) == 1
        assert calls[0]['function']['name'] == 'f'
        assert json.loads(calls[0]['function']['arguments']) == expected
    assert direct[0]['function'] == streamed[0]['function']
