"""CPU-only HTTP check of the patched TabbyAPI setup_app registration."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families


TABBY = Path(__file__).resolve().parents[1] / "tabbyAPI"


def load_source(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def app_modules(monkeypatch):
    """Stub GPU/import-time services; execute real auth, metrics and setup_app."""
    def stub(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    common = stub("common")
    common.__path__ = [str(TABBY / "common")]
    endpoints = stub("endpoints")
    endpoints.__path__ = [str(TABBY / "endpoints")]

    class Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    logger = Logger()
    stub("loguru", logger=logger)
    stub("uvicorn")
    stub("aiofiles")
    stub("ruamel")
    stub("ruamel.yaml", YAML=lambda **_kwargs: None)
    common.logger = stub("common.logger", xlogger=logger, UVICORN_LOG_CONFIG={})
    common.utils = stub("common.utils", coalesce=lambda *args: next((v for v in args if v is not None), None))
    common.signals = stub("common.signals", SERVER_SERVING=False)
    common.model = stub("common.model", container=None)
    common.networking = stub("common.networking", get_global_depends=lambda: [])
    network = types.SimpleNamespace(allowed_origins=[], api_servers=["oai"], access_log=False)
    common.tabby_config = stub("common.tabby_config", config=types.SimpleNamespace(network=network))

    async def context_length_exception_handler(_request, _exc):
        raise AssertionError("unrelated exception handler was called")

    common.errors = stub("common.errors", ContextLengthHTTPException=type("ContextLengthHTTPException", (Exception,), {}),
                         context_length_exception_handler=context_length_exception_handler)

    oai_router = types.SimpleNamespace(api_name="OAI", setup=lambda: APIRouter())
    kobold_router = types.SimpleNamespace(api_name="Kobold", setup=lambda: APIRouter())
    stub("endpoints.OAI", router=oai_router)
    stub("endpoints.Kobold", router=kobold_router)
    core = stub("endpoints.core")
    core.__path__ = [str(TABBY / "endpoints/core")]
    stub("endpoints.core.router", router=APIRouter())

    auth = load_source("common.auth", TABBY / "common/auth.py", monkeypatch)
    auth.AUTH_KEYS = auth.AuthKeys(api_key="test-api-key", admin_key="test-admin-key")
    auth.DISABLE_AUTH = False
    common.auth = auth
    common.metrics = load_source("common.metrics", TABBY / "common/metrics.py", monkeypatch)
    server = load_source("endpoints.server", TABBY / "endpoints/server.py", monkeypatch)
    return server, auth


def test_real_setup_app_metrics_route_auth_and_exposition(app_modules, monkeypatch):
    server, auth = app_modules
    calls = []
    real_exposition = server.exposition

    def counted_exposition():
        calls.append(True)
        return real_exposition()

    monkeypatch.setattr(server, "exposition", counted_exposition)
    app = server.setup_app()
    client = TestClient(app)

    denied = client.get("/metrics")
    assert denied.status_code == 401
    assert "tabbyapi_" not in denied.text
    assert calls == []

    invalid = client.get("/metrics", headers={"x-api-key": "wrong"})
    assert invalid.status_code == 401
    assert calls == []

    response = client.get("/metrics", headers={"x-api-key": "test-api-key"})
    assert response.status_code == 200
    assert response.headers["content-type"] == server.CONTENT_TYPE_LATEST
    parsed = {family.name: family for family in text_string_to_metric_families(response.text)}
    assert parsed["tabbyapi_metrics_schema_info"].type == "gauge"
    assert parsed["tabbyapi_requests_active"].type == "gauge"
    assert calls == [True]

    assert client.get("/v1/metrics", headers={"x-api-key": "test-api-key"}).status_code == 404
    assert calls == [True]
    assert not any(getattr(route, "path", None) == "/metrics" for route in app.routes
                   if getattr(route, "include_in_schema", True))
    assert auth.DISABLE_AUTH is False
