"""Real request parsing and locked-template rendering, with GPU imports excluded."""
import ast
import hashlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
TABBY = ROOT / "tabbyAPI"


@pytest.fixture
def resolve(monkeypatch):
    for name, path in [("common", "common"), ("endpoints", "endpoints"),
                       ("endpoints.OAI", "endpoints/OAI"),
                       ("endpoints.OAI.types", "endpoints/OAI/types")]:
        module = types.ModuleType(name)
        module.__path__ = [str(TABBY / path)]
        monkeypatch.setitem(sys.modules, name, module)
    sampler = types.ModuleType("common.sampling")
    sampler.BaseSamplerRequest = BaseModel
    sampler.get_default_sampler_value = lambda key, fallback=None: fallback
    monkeypatch.setitem(sys.modules, "common.sampling", sampler)
    for leaf in ["common", "tools", "chat_completion"]:
        name = f"endpoints.OAI.types.{leaf}"
        spec = importlib.util.spec_from_file_location(name, TABBY / f"endpoints/OAI/types/{leaf}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    request_type = module.ChatCompletionRequest
    tree = ast.parse((TABBY / "endpoints/OAI/utils/chat_completion.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_template_vars")
    namespace = {"ChatCompletionRequest": request_type}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "resolver", "exec"), namespace)

    def call(payload=None, *, default=None, force=None, compat=True):
        request = request_type(messages=[{"role": "user", "content": "Hello"}], **(payload or {}))
        container = SimpleNamespace(template_vars_default=default or {}, template_vars_force=force or {},
                                    qwen38_thinking_compat=compat)
        return namespace["resolve_template_vars"](request, container)
    return call


def render(variables):
    source = (ROOT / "tests/fixtures/qwen38-chat-template.jinja").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
    def reject(message):
        raise ValueError(message)
    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = reject
    return env.from_string(source).render(messages=[{"role": "user", "content": "Hello"}],
                                          add_generation_prompt=True, **variables)


@pytest.mark.parametrize("surface", ["flat", "reasoning", "template_vars", "chat_template_kwargs"])
@pytest.mark.parametrize("effort", ["high", "none", "low", "medium", "xhigh"])
def test_each_request_surface_renders_locked_template(resolve, surface, effort):
    payload = ({"reasoning_effort": effort} if surface == "flat" else
               {"reasoning": {"effort": effort}} if surface == "reasoning" else
               {surface: {"reasoning_effort": effort}})
    result = resolve(payload)
    if effort == "none":
        assert result == {"enable_thinking": False}
        assert render(result).endswith("<think>\n\n</think>\n\n")
    else:
        assert result["reasoning_effort"] == ("xhigh" if effort == "high" else effort)
        assert render(result).endswith("<think>\n")


def test_existing_precedence_and_explicit_toggle_win(resolve):
    result = resolve({"reasoning": {"effort": "low"}, "reasoning_effort": "high",
                      "template_vars": {"reasoning_effort": "medium", "custom": 7}},
                     default={"reasoning_effort": "xhigh"})
    assert result == {"reasoning_effort": "medium", "custom": 7}
    assert resolve({"reasoning_effort": "none", "enable_thinking": True}) == {"enable_thinking": True}
    assert resolve({"reasoning_effort": "none"}, force={"reasoning_effort": "high"}) == {"reasoning_effort": "xhigh"}
    assert resolve({"reasoning_effort": "high"}, force={"enable_thinking": False}) == {"reasoning_effort": "xhigh", "enable_thinking": False}
    assert resolve({"reasoning_effort": "none"}, default={"enable_thinking": True}) == {"enable_thinking": True}


def test_defaults_opt_in_unknown_effort_and_input_immutability(resolve):
    assert render(resolve()) == render({"reasoning_effort": "xhigh"})
    payload = {"template_vars": {"reasoning_effort": "high"}}
    assert resolve(payload)["reasoning_effort"] == "xhigh"
    assert payload["template_vars"]["reasoning_effort"] == "high"
    assert resolve(payload, compat=False)["reasoning_effort"] == "high"
    with pytest.raises(ValueError, match="Unexpected reasoning effort"):
        render(resolve({"reasoning_effort": "invalid"}))
