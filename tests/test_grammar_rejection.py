"""Execute upstream grammar construction with failing CPU filter stubs."""
import ast
import importlib.util
import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

TABBY = Path(__file__).resolve().parents[1] / "tabbyAPI"


@pytest.mark.parametrize("kind", ["json_schema", "regex", "grammar"])
@pytest.mark.parametrize("backend_ready", [True, False])
def test_rejected_constraint_never_becomes_an_empty_filter_list(kind, backend_ready):
    spec = importlib.util.spec_from_file_location("_grammar_errors", TABBY / "common/errors.py")
    errors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(errors)
    tree = ast.parse((TABBY / "backends/exllamav3/grammar.py").read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    def failing_filter(*args, **kwargs):
        raise RuntimeError("compile failed")
    namespace = dict(json=json, traceback=traceback, List=List, Optional=Optional,
                     Filter=object, Tokenizer=object, LLGuidanceFilter=failing_filter,
                     GrammarParseError=errors.GrammarParseError,
                     xlogger=SimpleNamespace(error=lambda *a: None, debug=lambda *a: None))
    exec(compile(tree, "grammar", "exec"), namespace)
    namespace["_llguidance_ready"] = lambda: backend_ready
    handler = namespace["ExLlamaV3Grammar"]()
    call, constraint = {"json_schema": (handler.add_json_schema_filter, {"type": "invalid"}),
                        "regex": (handler.add_regex_filter, "["),
                        "grammar": (handler.add_grammar_filter, "invalid")}[kind]
    with pytest.raises(errors.GrammarParseError if backend_ready else RuntimeError):
        call(constraint, object())
    assert handler.filters == []
