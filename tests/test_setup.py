"""CPU-only contract tests for the native deployment setup wizard."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from setup import wizard


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "setup").mkdir()
    (tmp_path / "setup/models.json").write_bytes((ROOT / "setup/models.json").read_bytes())
    (tmp_path / "config.yml").write_bytes((ROOT / "config.yml").read_bytes())
    (tmp_path / "model.lock.json").write_bytes((ROOT / "model.lock.json").read_bytes())
    return tmp_path


def _snapshot(root):
    return (root / "config.yml").read_bytes(), (root / "model.lock.json").read_bytes()


def _unexpected_runner(*args, **kwargs):
    pytest.fail("No installation command should run")


def test_catalog_has_distinct_immutable_quantization_variants():
    catalog = wizard.load_catalog(ROOT / "setup/models.json")
    assert catalog["source"] == "turboderp/Qwen3.8-27B-exl3"
    models = catalog["models"]
    branches = {model["branch"] for model in models}
    assert len(branches) == len(models)
    assert len({model["revision"] for model in models}) == len(models)
    assert {"SC_5.00bpw_H6", "SC_5.00bpw_H6_V6", "5.00bpw"} <= branches
    assert {"SC_6.00bpw_H6", "SC_6.00bpw_H6_V6"} <= branches
    for model in models:
        assert len(model["revision"]) == 40
        assert all(c in "0123456789abcdef" for c in model["revision"])
        assert model["label"].strip()
        assert model["directory"].strip()


@pytest.mark.parametrize("cache", ["FP16", "Q8", "Q6", "Q4"])
def test_cache_aliases(cache):
    assert wizard.parse_cache(cache) == cache


@pytest.mark.parametrize("k", range(2, 9))
@pytest.mark.parametrize("v", range(2, 9))
def test_all_native_numeric_cache_pairs(k, v):
    assert wizard.parse_cache(f"{k},{v}") == f"{k},{v}"


@pytest.mark.parametrize("cache", ["1,4", "4,9", "8", "Q2", "Q5", "garbage", "4,4,4"])
def test_reject_unsupported_cache_modes(cache):
    with pytest.raises(ValueError):
        wizard.parse_cache(cache)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("127.0.0.1", ("127.0.0.1", None)),
        ("192.0.2.10:8080", ("192.0.2.10", 8080)),
        ("localhost", ("localhost", None)),
        ("my-host.example:5050", ("my-host.example", 5050)),
        ("::1", ("::1", None)),
        ("[::1]:8080", ("::1", 8080)),
        ("http://127.0.0.1:8080/v1", ("127.0.0.1", 8080)),
        ("http://[2001:db8::1]:8080/v1", ("2001:db8::1", 8080)),
        ("http://my-host.example:8080", ("my-host.example", 8080)),
    ],
)
def test_endpoint_parsing_yields_native_bind_host(value, expected):
    assert wizard.parse_endpoint(value) == expected


def test_explicit_port_overrides_url_port():
    assert wizard.parse_endpoint("http://localhost:8080/v1", 9000) == ("localhost", 9000)


@pytest.mark.parametrize("value", ["http://", "http://a:bad", "http://a:8080/other", "ftp://a:8080", "user@host", "a b", "host:0", "host:65536", "http://a:80?x=1"])
def test_invalid_or_ambiguous_endpoints_fail(value):
    with pytest.raises(ValueError):
        wizard.parse_endpoint(value)


@pytest.mark.parametrize("port", [0, 65536, -1])
def test_invalid_explicit_ports_fail(port):
    with pytest.raises(ValueError):
        wizard.parse_endpoint("localhost", port)


@pytest.mark.parametrize("tokens,mode", [(0, "disabled"), (3, "mtp")])
def test_render_config_preserves_unrelated_settings_and_sets_fixed_mtp(tokens, mode):
    original = (ROOT / "config.yml").read_text().replace(
        "  draft_num_tokens: 3\n", "  draft_num_tokens: 3\n  dynamic_draft: true\n"
    )
    original += "\n# Operator override\nlogging:\n  log_requests: true\n"
    configured = wizard.render_config(
        original,
        model_dir="/tmp/selected-models",
        model_name="selected-quant",
        cache_mode="6,4",
        mtp_tokens=tokens,
        host="::1",
        port=9000,
    )
    before = yaml.safe_load(original)
    after = yaml.safe_load(configured)
    assert after["network"]["host"] == "::1"
    assert after["network"]["port"] == 9000
    assert after["model"]["model_dir"] == "/tmp/selected-models"
    assert after["model"]["model_name"] == "selected-quant"
    assert after["model"]["cache_mode"] == "6,4"
    assert after["draft_model"]["draft_mode"] == mode
    assert after["draft_model"]["draft_num_tokens"] == tokens
    assert after["draft_model"].get("dynamic_draft") is False
    for section, key in (
        ("network", "disable_auth"),
        ("network", "allowed_origins"),
        ("model", "max_seq_len"),
        ("model", "tool_format"),
        ("model", "vision"),
        ("draft_model", "draft_cache_mode"),
    ):
        assert after[section][key] == before[section][key]
    assert after["logging"] == {"log_requests": True}
    assert "# Operator override" in configured


@pytest.mark.parametrize("cache", ["FP16", "Q8", "Q6", "Q4", "2,8", "8,2"])
@pytest.mark.parametrize("tokens", [0, 3])
def test_rendered_variants_validate_against_pinned_native_schema(cache, tokens):
    spec = importlib.util.spec_from_file_location(
        "pinned_tabby_config_models", ROOT / "tabbyAPI/common/config_models.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = wizard.render_config(
        (ROOT / "config.yml").read_text(),
        model_dir="/tmp/selected-models",
        model_name="selected-quant",
        cache_mode=cache,
        mtp_tokens=tokens,
        host="127.0.0.1",
        port=8080,
    )
    parsed = module.TabbyConfigModel.model_validate(yaml.safe_load(rendered))
    assert parsed.model.cache_mode == cache
    assert parsed.draft_model.draft_mode == ("disabled" if tokens == 0 else "mtp")


def test_lock_uses_immutable_selected_revision_and_drops_other_quant_hashes():
    prior = json.loads((ROOT / "model.lock.json").read_text())
    entry = next(
        model
        for model in wizard.load_catalog(ROOT / "setup/models.json")["models"]
        if model["branch"] == "SC_6.00bpw_H6_V6"
    )
    changed = wizard.make_lock(prior, entry, "turboderp/Qwen3.8-27B-exl3")
    assert changed["revision"] == entry["revision"]
    assert changed["source"] == "turboderp/Qwen3.8-27B-exl3"
    assert changed["revision"] != prior["revision"]
    assert not changed.get("weight_shards_sha256")


def test_lock_preserves_verified_hashes_for_same_revision():
    prior = json.loads((ROOT / "model.lock.json").read_text())
    entry = next(
        model
        for model in wizard.load_catalog(ROOT / "setup/models.json")["models"]
        if model["revision"] == prior["revision"]
    )
    changed = wizard.make_lock(prior, entry, prior["source"])
    assert changed["weight_shards_sha256"] == prior["weight_shards_sha256"]


def test_dry_run_previews_without_any_mutation_or_install(sandbox, tmp_path):
    before = _snapshot(sandbox)
    models_dir = tmp_path / "models"
    result = wizard.main(
        ["--model", "SC_6.00bpw_H6_V6", "--cache", "4,8", "--mtp", "0", "--url", "http://localhost:9000/v1", "--models-dir", str(models_dir), "--dry-run", "--yes"],
        root=sandbox,
        runner=_unexpected_runner,
    )
    assert result == 0
    assert _snapshot(sandbox) == before
    assert not (sandbox / ".state").exists()
    assert not models_dir.exists()


def test_configure_only_writes_config_lock_and_backup_without_install(sandbox, tmp_path):
    before = _snapshot(sandbox)
    result = wizard.main(
        ["--model", "SC_6.00bpw_H6_V6", "--cache", "Q6", "--mtp", "0", "--url", "http://[::1]:9000/v1", "--models-dir", str(tmp_path / "models"), "--configure-only", "--yes"],
        root=sandbox,
        runner=_unexpected_runner,
    )
    assert result == 0
    config = yaml.safe_load((sandbox / "config.yml").read_text())
    lock = json.loads((sandbox / "model.lock.json").read_text())
    assert config["network"]["host"] == "::1"
    assert config["network"]["port"] == 9000
    assert config["model"]["cache_mode"] == "Q6"
    assert config["draft_model"]["draft_mode"] == "disabled"
    assert lock["revision"] == "60d005a257b39ecb25e4ba23c1dd29df877d5c69"
    assert not lock.get("weight_shards_sha256")
    backups = list((sandbox / ".state/setup").iterdir())
    assert len(backups) == 1
    assert _snapshot(backups[0]) == before
    assert not (tmp_path / "models").exists()


@pytest.mark.parametrize("bad", ["ftp://localhost:8080", "http://localhost:0/v1", "http://localhost:8080/other"])
def test_invalid_url_leaves_files_untouched(sandbox, bad):
    before = _snapshot(sandbox)
    result = wizard.main(["--url", bad, "--configure-only", "--yes"], root=sandbox, runner=_unexpected_runner)
    assert result == 1
    assert _snapshot(sandbox) == before
    assert not (sandbox / ".state").exists()


def test_interactive_cancellation_leaves_files_untouched(sandbox, monkeypatch):
    before = _snapshot(sandbox)
    monkeypatch.setattr("builtins.input", lambda _: "")
    result = wizard.main(["--configure-only"], root=sandbox, runner=_unexpected_runner)
    assert result == 0
    assert _snapshot(sandbox) == before
    assert not (sandbox / ".state").exists()


def test_same_revision_reuses_current_model_directory(sandbox):
    original = (sandbox / "config.yml").read_text()
    installed = sandbox / "installed-models"
    (installed / "qwen3-8-27b-exl3-sc-5-00bpw-h6").mkdir(parents=True)
    (sandbox / "config.yml").write_text(original.replace("model_dir: models", f"model_dir: {installed}"))
    before = yaml.safe_load((sandbox / "config.yml").read_text())
    result = wizard.main(["--configure-only", "--yes"], root=sandbox, runner=_unexpected_runner)
    assert result == 0
    after = yaml.safe_load((sandbox / "config.yml").read_text())
    assert (after["model"]["model_dir"], after["model"]["model_name"]) == (
        before["model"]["model_dir"], before["model"]["model_name"]
    )


def test_relative_model_dir_resolves_from_tabbyapi_directory(sandbox):
    installed = sandbox / "tabbyAPI/models/qwen3-8-27b-exl3-sc-5-00bpw-h6"
    installed.mkdir(parents=True)
    result = wizard.main(["--configure-only", "--yes"], root=sandbox, runner=_unexpected_runner)
    assert result == 0
    after = yaml.safe_load((sandbox / "config.yml").read_text())
    assert after["model"]["model_dir"] == str((sandbox / "tabbyAPI/models").resolve())
    assert after["model"]["model_name"] == "qwen3-8-27b-exl3-sc-5-00bpw-h6"


def test_fresh_clone_does_not_keep_foreign_model_path(sandbox):
    result = wizard.main(["--configure-only", "--yes"], root=sandbox, runner=_unexpected_runner)
    assert result == 0
    after = yaml.safe_load((sandbox / "config.yml").read_text())
    assert after["model"]["model_dir"] == str(Path.home() / "models")
    assert after["model"]["model_name"] == "qwen3-8-27b-exl3-sc-5-00bpw-h6"


def test_interactive_models_directory_answer_is_used(sandbox, tmp_path, monkeypatch):
    custom = tmp_path / "chosen-models"
    responses = iter(["", "", "", "", "", str(custom), "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    result = wizard.main(["--configure-only"], root=sandbox, runner=_unexpected_runner)
    assert result == 0
    after = yaml.safe_load((sandbox / "config.yml").read_text())
    assert after["model"]["model_dir"] == str(custom)


@pytest.mark.parametrize("directory", [".", ".."])
def test_catalog_rejects_directory_traversal(sandbox, directory):
    catalog_path = sandbox / "setup/models.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["models"][0]["directory"] = directory
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError):
        wizard.load_catalog(catalog_path)


def test_full_install_uses_pinned_commands_and_never_starts_server(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wizard.platform, "machine", lambda: "x86_64")
    commands = []

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return SimpleNamespace(stdout=wizard.UPSTREAM + "\n" if argv[:3] == ["git", "rev-parse", "HEAD"] else "")

    selected = next(m for m in wizard.load_catalog(ROOT / "setup/models.json")["models"] if m["branch"] == "SC_6.00bpw_H6_V6")
    result = wizard.main(
        ["--model", selected["branch"], "--models-dir", str(tmp_path / "models"), "--yes"],
        root=sandbox,
        runner=runner,
    )
    assert result == 0
    argv = [command for command, _ in commands]
    assert ["git", "submodule", "update", "--init", "--checkout"] in argv
    assert [str(sandbox / "scripts/apply-metrics")] in argv
    assert [str(sandbox / "scripts/apply-ban-eos")] in argv
    assert ["uv", "python", "install", "3.12.13"] in argv
    assert any(command[:3] == ["uv", "pip", "install"] and f"{sandbox / 'tabbyAPI'}[cu13]" in command for command in argv)
    downloads = [command for command in argv if "download" in command]
    assert len(downloads) == 1
    assert ["--revision", selected["revision"]] == downloads[0][downloads[0].index("--revision"):downloads[0].index("--revision") + 2]
    assert not any("main.py" in command or "start.py" in command for command in argv)


def test_install_error_keeps_original_config_and_lock(sandbox, monkeypatch):
    monkeypatch.setattr(wizard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wizard.platform, "machine", lambda: "x86_64")
    before = _snapshot(sandbox)

    def failing_runner(argv, **kwargs):
        raise OSError("mock installer failure")

    result = wizard.main(["--yes"], root=sandbox, runner=failing_runner)
    assert result == 1
    assert _snapshot(sandbox) == before


def test_nonempty_model_dir_with_unknown_revision_blocks_install(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wizard.platform, "machine", lambda: "x86_64")
    before = _snapshot(sandbox)
    entry = next(m for m in wizard.load_catalog(ROOT / "setup/models.json")["models"] if m["branch"] == "SC_6.00bpw_H6_V6")
    target = tmp_path / "models" / entry["directory"]
    target.mkdir(parents=True)
    (target / "partial-file").write_text("unknown revision")
    result = wizard.main(["--model", entry["branch"], "--models-dir", str(tmp_path / "models"), "--yes"], root=sandbox, runner=_unexpected_runner)
    assert result == 1
    assert _snapshot(sandbox) == before


def test_models_directory_regular_file_fails_before_install_commands(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wizard.platform, "machine", lambda: "x86_64")
    before = _snapshot(sandbox)
    regular_file = tmp_path / "models"
    regular_file.write_text("not a directory")
    result = wizard.main(
        ["--models-dir", str(regular_file), "--yes"],
        root=sandbox,
        runner=_unexpected_runner,
    )
    assert result == 1
    assert _snapshot(sandbox) == before
    assert regular_file.read_text() == "not a directory"


def test_interrupted_download_can_resume_same_revision(sandbox, tmp_path, monkeypatch):
    monkeypatch.setattr(wizard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(wizard.platform, "machine", lambda: "x86_64")
    before = _snapshot(sandbox)
    entry = next(m for m in wizard.load_catalog(ROOT / "setup/models.json")["models"] if m["branch"] == "SC_6.00bpw_H6_V6")
    target = tmp_path / "models" / entry["directory"]
    argv = ["--model", entry["branch"], "--models-dir", str(tmp_path / "models"), "--yes"]
    commands = []

    def runner(command, **kwargs):
        commands.append(command)
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=wizard.UPSTREAM + "\n")
        if "download" in command and len([c for c in commands if "download" in c]) == 1:
            (target / "partial-shard").write_text("partial")
            raise OSError("mock interrupted download")
        return SimpleNamespace(stdout="")

    assert wizard.main(argv, root=sandbox, runner=runner) == 1
    assert _snapshot(sandbox) == before
    assert json.loads((target / ".setup-model.json").read_text()) == {
        "source": "turboderp/Qwen3.8-27B-exl3",
        "revision": entry["revision"],
        "complete": False,
    }
    assert wizard.main(argv, root=sandbox, runner=runner) == 0
    assert len([c for c in commands if "download" in c]) == 2
    assert json.loads((target / ".setup-model.json").read_text())["complete"] is True


@pytest.mark.parametrize("failure", [OSError("mock replace failure"), KeyboardInterrupt()])
def test_second_file_write_failure_restores_both_original_files(sandbox, monkeypatch, failure):
    before = _snapshot(sandbox)
    original_replace = wizard.os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original_replace(source, destination)

    monkeypatch.setattr(wizard.os, "replace", fail_second_replace)
    args = ["--model", "SC_6.00bpw_H6_V6", "--configure-only", "--yes"]
    assert wizard.main(args, root=sandbox, runner=_unexpected_runner) == (
        130 if isinstance(failure, KeyboardInterrupt) else 1
    )
    assert calls == 2
    assert _snapshot(sandbox) == before
    assert not list(sandbox.glob(".config.yml.*"))
    assert not list(sandbox.glob(".model.lock.json.*"))
