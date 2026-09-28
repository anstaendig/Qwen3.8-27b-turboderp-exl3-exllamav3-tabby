"""CPU-only acceptance tests for the public Prometheus job metrics contract."""

import ast
import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import List

import pytest
from prometheus_client.parser import text_string_to_metric_families


METRICS_PATH = Path(__file__).resolve().parents[1] / "tabbyAPI/common/metrics.py"
BACKEND_PATH = Path(__file__).resolve().parents[1] / "tabbyAPI/backends/exllamav3/model.py"


@pytest.fixture
def metrics(monkeypatch):
    """Each case gets a fresh process-like registry and a fake model holder."""
    common = types.ModuleType("common")
    model = types.ModuleType("common.model")
    model.container = None
    common.model = model
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.model", model)
    name = "_test_metrics_" + str(id(model))
    spec = importlib.util.spec_from_file_location(name, METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, model


def families(module):
    return {family.name: family for family in text_string_to_metric_families(module.exposition().decode())}


def value(module, family, *, labels=None, suffix=""):
    labels = labels or {}
    sample_name = family + suffix
    parsed = families(module)
    metric = parsed.get(family) or parsed.get(family.removesuffix("_total"))
    assert metric is not None, family
    samples = [s.value for s in metric.samples
               if s.name == sample_name and s.labels == labels]
    assert len(samples) == 1, (sample_name, labels, samples)
    assert math.isfinite(samples[0]) and samples[0] >= 0
    return samples[0]


def test_idle_exposition_has_fixed_families_and_zero_children(metrics):
    module, _ = metrics
    parsed = families(module)
    assert parsed["tabbyapi_metrics_schema_info"].type == "gauge"
    assert parsed["tabbyapi_requests_active"].type == "gauge"
    assert parsed["tabbyapi_requests_active"].samples
    assert parsed["tabbyapi_requests_active"].samples[0].name == "tabbyapi_requests_active"
    assert parsed["tabbyapi_requests"].type == "counter"
    assert parsed["tabbyapi_prompt_tokens"].type == "counter"
    assert parsed["tabbyapi_engine_time_to_first_token_seconds"].type == "histogram"
    assert parsed["tabbyapi_engine_phase_duration_seconds"].type == "histogram"
    assert value(module, "tabbyapi_metrics_schema_info", labels={"version": "1"}) == 1
    assert value(module, "tabbyapi_model_loaded") == 0
    for state in ("queued", "prefill", "decode"):
        assert value(module, "tabbyapi_requests_active", labels={"state": state}) == 0
    for outcome in ("completed", "cancelled", "error"):
        assert value(module, "tabbyapi_requests_total", labels={"outcome": outcome}) == 0
    for source in ("cached", "computed"):
        assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": source}) == 0
    for phase in ("queue", "prefill", "decode", "total"):
        assert value(module, "tabbyapi_engine_phase_duration_seconds", labels={"phase": phase}, suffix="_count") == 0
    assert value(module, "tabbyapi_engine_time_to_first_token_seconds", suffix="_count") == 0
    assert "tabbyapi_mtp_draft_tokens" not in parsed
    assert "tabbyapi_kv_cache_used_tokens" not in parsed


def test_scraping_is_read_only_and_has_no_private_labels(metrics):
    module, _ = metrics
    job = module.JobMetrics(10)
    job.started()
    job.prefill(8)
    job.output(3)
    first = module.exposition()
    assert module.exposition() == first
    assert module.exposition() == first
    allowed = {"version", "state", "outcome", "source", "phase", "le"}
    for family in families(module).values():
        for sample in family.samples:
            assert set(sample.labels) <= allowed
            assert math.isfinite(sample.value) and sample.value >= 0


def test_cache_is_one_current_snapshot_and_absent_on_failure(metrics):
    module, model = metrics
    class Generator:
        stats = {"max_tokens": 100, "used_tokens": 20, "cached_tokens": 8}

        def get_cache_stats(self):
            if isinstance(self.stats, Exception):
                raise self.stats
            return self.stats

    generator = Generator()
    model.container = types.SimpleNamespace(
        loaded=True, model=object(), generator=types.SimpleNamespace(generator=generator)
    )
    assert value(module, "tabbyapi_model_loaded") == 1
    assert value(module, "tabbyapi_kv_cache_capacity_tokens") == 100
    assert value(module, "tabbyapi_kv_cache_used_tokens") == 20
    assert value(module, "tabbyapi_kv_cache_reusable_tokens") == 8
    generator.stats = {"max_tokens": 100, "used_tokens": 0, "cached_tokens": 0}
    assert value(module, "tabbyapi_kv_cache_used_tokens") == 0
    generator.stats = RuntimeError("secret cache path")
    exposed = module.exposition()
    assert b"tabbyapi_kv_cache_used_tokens" not in exposed
    assert b"secret cache path" not in exposed
    generator.stats = {"max_tokens": 100, "used_tokens": 20, "cached_tokens": 8}
    model.container.model = None
    assert value(module, "tabbyapi_model_loaded") == 0
    for family in (
        "tabbyapi_kv_cache_capacity_tokens",
        "tabbyapi_kv_cache_used_tokens",
        "tabbyapi_kv_cache_reusable_tokens",
    ):
        assert family not in families(module)
    model.container.model = object()
    model.container.loaded = False
    assert value(module, "tabbyapi_model_loaded") == 0
    assert "tabbyapi_kv_cache_capacity_tokens" not in families(module)


def test_state_output_and_repeated_prefill_do_not_invent_prompt_split(metrics):
    module, _ = metrics
    job = module.JobMetrics(20)
    assert value(module, "tabbyapi_requests_active", labels={"state": "queued"}) == 1
    job.started()
    assert value(module, "tabbyapi_requests_active", labels={"state": "prefill"}) == 1
    for progress in (10, 10, 8, 20, 20):
        job.prefill(progress)
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "cached"}) == 0
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "computed"}) == 0
    job.first_output()
    job.output(2)
    job.output(1)
    assert value(module, "tabbyapi_requests_active", labels={"state": "decode"}) == 1
    assert value(module, "tabbyapi_output_tokens_total") == 3
    assert value(module, "tabbyapi_engine_time_to_first_token_seconds", suffix="_count") == 1
    job.finish("cancelled")
    job.finish("cancelled")
    assert value(module, "tabbyapi_requests_total", labels={"outcome": "cancelled"}) == 1
    assert value(module, "tabbyapi_requests_active", labels={"state": "decode"}) == 0
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "cached"}) == 0
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "computed"}) == 0


RESULT = {"prompt_tokens": 20, "cached_tokens": 5, "time_enqueued": 0.123456,
          "time_prefill": 1.234567, "time_generate": 2.345678,
          "accepted_draft_tokens": 3, "rejected_draft_tokens": 4}


def test_eos_records_exact_prompt_split_timing_and_mtp_once(metrics):
    module, _ = metrics
    module.enable_mtp()
    job = module.JobMetrics(20)
    job.started()
    job.prefill(10)
    job.first_output()
    job.output(2)
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "cached"}) == 0
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "computed"}) == 0
    job.completed(RESULT, mtp_enabled=True)
    job.completed(RESULT, mtp_enabled=True)
    job.finish("completed")
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "cached"}) == 5
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "computed"}) == 15
    assert value(module, "tabbyapi_engine_phase_duration_seconds", labels={"phase": "total"}, suffix="_sum") == pytest.approx(3.703701)
    for phase in ("queue", "prefill", "decode", "total"):
        assert value(module, "tabbyapi_engine_phase_duration_seconds", labels={"phase": phase}, suffix="_count") == 1
    assert value(module, "tabbyapi_mtp_draft_tokens_total", labels={"outcome": "accepted"}) == 3
    assert value(module, "tabbyapi_mtp_draft_tokens_total", labels={"outcome": "rejected"}) == 4
    assert value(module, "tabbyapi_requests_total", labels={"outcome": "completed"}) == 1


def test_cancel_and_error_do_not_invent_prompt_split_without_eos(metrics):
    module, _ = metrics
    cancelled = module.JobMetrics(20)
    cancelled.started()
    cancelled.prefill(10)
    cancelled.finish("cancelled")
    errored = module.JobMetrics(20)
    errored.finish("error")
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "computed"}) == 0
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "cached"}) == 0
    assert value(module, "tabbyapi_requests_total", labels={"outcome": "cancelled"}) == 1
    assert value(module, "tabbyapi_requests_total", labels={"outcome": "error"}) == 1
    assert value(module, "tabbyapi_engine_phase_duration_seconds", labels={"phase": "total"}, suffix="_count") == 0
    for state in ("queued", "prefill", "decode"):
        assert value(module, "tabbyapi_requests_active", labels={"state": state}) == 0


def test_first_text_without_source_ids_has_ttft_but_no_guessed_output(metrics):
    module, _ = metrics
    job = module.JobMetrics(4)
    job.started()
    job.first_output()
    job.first_output()
    assert value(module, "tabbyapi_requests_active", labels={"state": "decode"}) == 1
    assert value(module, "tabbyapi_engine_time_to_first_token_seconds", suffix="_count") == 1
    assert value(module, "tabbyapi_output_tokens_total") == 0
    job.output(2)
    job.output(1)
    assert value(module, "tabbyapi_output_tokens_total") == 3
    job.finish("cancelled")


def test_coalesced_mixed_source_ids_count_only_known_spans(metrics):
    """Execute the real backend helper without importing GPU dependencies."""
    module, _ = metrics
    tree = ast.parse(BACKEND_PATH.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_observe_output_metrics")
    code = compile(ast.Module(body=[helper], type_ignores=[]), str(BACKEND_PATH), "exec")
    namespace = {
        "List": List,
        "unwrap": lambda value, default: default if value is None else value,
        "observe": module.observe,
        "torch": types.SimpleNamespace(Tensor=type("FakeTensor", (), {})),
    }
    exec(code, namespace)

    job = module.JobMetrics(10)
    source_spans = [
        {"text": "known", "token_ids": [11, 12]},
        {"text": "unknown", "token_ids": None},
        {"text": "also known", "token_ids": [13]},
        {"text": "", "token_ids": [99]},
    ]
    namespace["_observe_output_metrics"](job, source_spans)
    assert value(module, "tabbyapi_output_tokens_total") == 3
    assert value(module, "tabbyapi_metrics_observation_errors_total") == 1
    job.finish("cancelled")


def test_invalid_eos_observations_do_not_publish_partial_timing_or_draft(metrics):
    module, _ = metrics
    module.enable_mtp()
    job = module.JobMetrics(20)
    result = {**RESULT, "time_prefill": math.nan, "rejected_draft_tokens": None}
    job.completed(result, mtp_enabled=True)
    job.finish("completed")
    for phase in ("queue", "prefill", "decode", "total"):
        assert value(module, "tabbyapi_engine_phase_duration_seconds", labels={"phase": phase}, suffix="_count") == 0
    for outcome in ("accepted", "rejected"):
        assert value(module, "tabbyapi_mtp_draft_tokens_total", labels={"outcome": outcome}) == 0
    assert value(module, "tabbyapi_metrics_observation_errors_total") > 0


def test_invalid_eos_prompt_pair_is_omitted_as_a_whole(metrics):
    module, _ = metrics
    job = module.JobMetrics(20)
    job.started()
    job.prefill(20)
    job.completed({**RESULT, "cached_tokens": 21})
    job.finish("completed")
    for source in ("cached", "computed"):
        assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": source}) == 0
    assert value(module, "tabbyapi_metrics_observation_errors_total") > 0
    assert value(module, "tabbyapi_requests_total", labels={"outcome": "completed"}) == 1


def test_prefill_progress_past_prompt_is_clamped_without_invalidating_eos(metrics):
    module, _ = metrics
    job = module.JobMetrics(20)
    job.started()
    job.prefill(24)
    assert job.progress == 20
    assert value(module, "tabbyapi_metrics_observation_errors_total") == 0
    job.completed({**RESULT, "prompt_tokens": 20, "cached_tokens": 0})
    job.finish("completed")
    assert value(module, "tabbyapi_prompt_tokens_total", labels={"source": "computed"}) == 20
    assert value(module, "tabbyapi_metrics_observation_errors_total") == 0
