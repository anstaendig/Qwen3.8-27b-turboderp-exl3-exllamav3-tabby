#!/usr/bin/env python3
"""Configure and optionally install the pinned native TabbyAPI deployment.

Only the known, scalar native fields are edited in config.yml. Unsupported YAML
shapes fail closed; the rest of the file is preserved byte for byte.
"""

from __future__ import annotations

import argparse
import ast
import ipaddress
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
REVISION = re.compile(r"^[0-9a-f]{40}$")
BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DIRECTORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HOST_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
KEY = re.compile(r"^  ([A-Za-z_][A-Za-z0-9_]*):(?:[ \t]+(.*?))?[ \t]*$")
SECTION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):[ \t]*$")
UPSTREAM = "f07131cd8fe34e449fe87cdd3a066b52b96d3cac"


def load_catalog(path: Path) -> dict:
    catalog = json.loads(Path(path).read_text())
    if not isinstance(catalog, dict) or not isinstance(catalog.get("source"), str):
        raise ValueError("Invalid model catalog source")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", catalog["source"]):
        raise ValueError("Invalid model catalog source")
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Model catalog is empty")
    branches, directories = set(), set()
    for entry in models:
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in ("branch", "revision", "label", "directory")):
            raise ValueError("Invalid model catalog entry")
        if not BRANCH.fullmatch(entry["branch"]) or not REVISION.fullmatch(entry["revision"]):
            raise ValueError("Invalid model branch or immutable revision")
        if entry["directory"] in (".", "..") or not DIRECTORY.fullmatch(entry["directory"]) or not entry["label"].strip():
            raise ValueError("Invalid model directory or label")
        if entry["branch"] in branches or entry["directory"] in directories:
            raise ValueError("Duplicate model branch or directory")
        branches.add(entry["branch"])
        directories.add(entry["directory"])
    return catalog


def parse_cache(value: str) -> str:
    value = value.strip().upper()
    if value in {"FP16", "Q8", "Q6", "Q4"}:
        return value
    match = re.fullmatch(r"([2-8])\s*,\s*([2-8])", value)
    if match:
        return f"{match[1]},{match[2]}"
    raise ValueError("KV cache must be FP16, Q8, Q6, Q4, or K,V bits from 2 to 8")


def parse_port(value: object) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError("Port must be an integer from 1 to 65535")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("Port must be an integer from 1 to 65535")
    return port


def parse_mtp(value: object) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError("MTP tokens must be a nonnegative integer")
    return int(value)


def validate_host(value: str) -> str:
    host = value.strip()
    if not host or any(c.isspace() for c in host) or "%" in host:
        raise ValueError("Invalid bind host")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    if len(host) > 253 or not all(HOST_LABEL.fullmatch(label) for label in host.split(".")):
        raise ValueError("Bind host must be an IPv4 address, IPv6 address, or hostname")
    return host


def parse_endpoint(value: str, port_override: object = None) -> tuple[str, int | None]:
    if not value:
        raise ValueError("Bind address is empty")
    if any(c.isspace() for c in value):
        raise ValueError("Bind address cannot contain whitespace")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme.lower() != "http" or not parsed.netloc:
            raise ValueError("Only plain HTTP URLs are supported; native TLS is not configured")
        if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
            raise ValueError("URL credentials are not supported")
        if "?" in value or "#" in value or parsed.path not in ("", "/", "/v1", "/v1/"):
            raise ValueError("URL path must be /v1, with no query or fragment")
        try:
            host, url_port = parsed.hostname, parsed.port
        except ValueError as exc:
            raise ValueError("Invalid URL port or host") from exc
        if host is None:
            raise ValueError("URL host is empty")
        return validate_host(host), parse_port(port_override) if port_override is not None else (parse_port(url_port) if url_port is not None else None)
    if any(c in value for c in "/?#@"):
        raise ValueError("Use a plain host or an HTTP URL ending in /v1")
    bracketed = re.fullmatch(r"\[([^]]+)\](?::([0-9]+))?", value)
    if bracketed:
        try:
            address = ipaddress.IPv6Address(bracketed[1])
        except ValueError as exc:
            raise ValueError("Invalid bracketed IPv6 address") from exc
        embedded = parse_port(bracketed[2]) if bracketed[2] is not None else None
        return str(address), parse_port(port_override) if port_override is not None else embedded
    if value.count(":") == 1 and not value.startswith("["):
        host, raw_port = value.rsplit(":", 1)
        return validate_host(host), parse_port(port_override) if port_override is not None else parse_port(raw_port)
    return validate_host(value), parse_port(port_override) if port_override is not None else None


def _locations(text: str) -> tuple[list[str], dict[str, dict[str, int]]]:
    lines = text.splitlines(keepends=True)
    sections: dict[str, dict[str, int]] = {}
    current = None
    for index, line in enumerate(lines):
        content = line.rstrip("\r\n")
        if not content or content.lstrip().startswith("#"):
            continue
        section = SECTION.fullmatch(content)
        if section:
            current = section[1]
            if current in sections:
                raise ValueError(f"Duplicate YAML section: {current}")
            sections[current] = {}
            continue
        if not content[0].isspace():
            current = None
        if current in {"network", "model", "draft_model"}:
            key = KEY.fullmatch(content)
            if key:
                if key[1] in sections[current]:
                    raise ValueError(f"Duplicate YAML key: {current}.{key[1]}")
                sections[current][key[1]] = index
            elif content.startswith("  ") and not content.startswith("    "):
                raise ValueError(f"Unsupported YAML structure in {current}")
    return lines, sections


def _scalar(text: str, section: str, key: str) -> str:
    lines, locations = _locations(text)
    try:
        raw = KEY.fullmatch(lines[locations[section][key]].rstrip("\r\n"))[2]
    except KeyError as exc:
        raise ValueError(f"Missing native config field: {section}.{key}") from exc
    if raw is None or not raw or raw[0] in "&*|>!{[":
        raise ValueError(f"Unsupported native config value: {section}.{key}")
    if raw[0] in "\"'":
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid native config value: {section}.{key}") from exc
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    return raw.strip()


def config_defaults(text: str) -> dict:
    result = {}
    for section, key, name in (("network", "host", "host"), ("network", "port", "port"), ("model", "model_dir", "model_dir"), ("model", "model_name", "model_name"), ("model", "cache_mode", "cache_mode"), ("draft_model", "draft_num_tokens", "mtp_tokens")):
        result[name] = _scalar(text, section, key)
    result["port"] = parse_port(result["port"])
    result["mtp_tokens"] = parse_mtp(result["mtp_tokens"])
    if _scalar(text, "draft_model", "draft_mode") == "disabled":
        result["mtp_tokens"] = 0
    return result


def render_config(text: str, *, model_dir: str, model_name: str, cache_mode: str, mtp_tokens: int, host: str, port: int) -> str:
    host = validate_host(host)
    port = parse_port(port)
    cache_mode = parse_cache(cache_mode)
    mtp_tokens = parse_mtp(mtp_tokens)
    if not Path(model_dir).is_absolute() or model_name in ("", ".", "..") or "/" in model_name or "\\" in model_name:
        raise ValueError("Model directory must be absolute and model name must be a directory name")
    replacements = {
        "network": {"host": json.dumps(host), "port": str(port)},
        "model": {"model_dir": json.dumps(model_dir), "model_name": json.dumps(model_name), "cache_mode": json.dumps(cache_mode)},
        "draft_model": {"draft_mode": json.dumps("mtp" if mtp_tokens else "disabled"), "draft_num_tokens": str(mtp_tokens)},
    }
    lines, locations = _locations(text)
    if lines and lines[0].startswith("# Native TabbyAPI configuration for the pinned Qwen3.8"):
        lines[0] = "# Native TabbyAPI configuration for the selected Qwen3.8 EXL3 model.\n"
    # A previous dynamic draft setting must not override a selected fixed count.
    if "dynamic_draft" in locations.get("draft_model", {}):
        replacements["draft_model"]["dynamic_draft"] = "false"
    for section, values in replacements.items():
        for key, value in values.items():
            _scalar(text, section, key)
            index = locations[section][key]
            ending = "\r\n" if lines[index].endswith("\r\n") else "\n" if lines[index].endswith("\n") else ""
            lines[index] = f"  {key}: {value}{ending}"
    if "dynamic_draft" not in locations.get("draft_model", {}):
        index = locations["draft_model"]["draft_num_tokens"]
        lines.insert(index + 1, "  dynamic_draft: false\n")
    return "".join(lines)


def make_lock(existing: dict, entry: dict, source: str) -> dict:
    same = existing.get("source") == source and existing.get("revision") == entry["revision"]
    if same:
        result = dict(existing)
    else:
        result = {}
    result.update({"model_id": entry["directory"], "source": source, "revision": entry["revision"]})
    if not same:
        result["quantization"] = entry["label"]
        result["tokenizer_source"] = source
        result["tokenizer_revision"] = entry["revision"]
    return result


def _ask(prompt: str, default: str) -> str:
    response = input(f"{prompt} [{default}]: ").strip()
    return response or default


def _run(runner, argv: list[str], *, cwd: Path | None = None, capture_output: bool = False):
    print("+", " ".join(argv))
    return runner(argv, cwd=cwd, check=True, capture_output=capture_output, text=capture_output)


def install(root: Path, *, source: str, entry: dict, model_dir: Path, existing_lock: dict, runner=subprocess.run) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Full install requires Linux x86_64 for the pinned CUDA 13 wheels")
    target = model_dir / entry["directory"]
    marker = target / ".setup-model.json"
    if target.exists() and any(target.iterdir()):
        if marker.exists():
            record = json.loads(marker.read_text())
            if record.get("source") != source or record.get("revision") != entry["revision"]:
                raise ValueError(f"Model directory belongs to another revision: {target}")
        elif existing_lock.get("source") == source and existing_lock.get("revision") == entry["revision"] and existing_lock.get("model_id") == entry["directory"]:
            pass
        else:
            raise ValueError(f"Nonempty model directory has no matching revision record: {target}")
    # Validate the destination before any dependency or submodule command runs.
    target.mkdir(parents=True, exist_ok=True)
    submodule = root / "tabbyAPI"
    if not (submodule / ".git").exists():
        _run(runner, ["git", "submodule", "update", "--init", "--checkout"], cwd=root)
    head = _run(runner, ["git", "rev-parse", "HEAD"], cwd=submodule, capture_output=True).stdout.strip()
    if head != UPSTREAM:
        raise ValueError(f"TabbyAPI must be pinned at {UPSTREAM}; found {head}")
    _run(runner, [str(root / "scripts" / "apply-metrics")], cwd=root)
    _run(runner, [str(root / "scripts" / "apply-ban-eos")], cwd=root)
    _run(runner, ["uv", "python", "install", "3.12.13"], cwd=root)
    venv = submodule / "venv"
    python = venv / "bin" / "python"
    if venv.exists():
        if not python.exists():
            raise ValueError(f"Existing environment has no Python interpreter: {venv}")
        version = _run(runner, [str(python), "--version"], cwd=root, capture_output=True).stdout.strip()
        if version != "Python 3.12.13":
            raise ValueError(f"Existing environment has incompatible Python: {version}")
    else:
        _run(runner, ["uv", "venv", str(venv), "-p", "3.12.13"], cwd=root)
    _run(runner, ["uv", "pip", "install", "--python", str(python), "-e", f"{submodule}[cu13]"], cwd=root)
    marker.write_text(json.dumps({"source": source, "revision": entry["revision"], "complete": False}, indent=2) + "\n")
    _run(runner, ["uv", "tool", "run", "--from", "huggingface_hub==1.2.3", "hf", "download", source, "--revision", entry["revision"], "--local-dir", str(target)], cwd=root)
    marker.write_text(json.dumps({"source": source, "revision": entry["revision"], "complete": True}, indent=2) + "\n")


def _write_transaction(root: Path, config: str, lock: dict) -> Path:
    config_path, lock_path = root / "config.yml", root / "model.lock.json"
    backup = root / ".state" / "setup" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, backup / "config.yml")
    shutil.copy2(lock_path, backup / "model.lock.json")
    pending = []
    try:
        for destination, content in ((config_path, config), (lock_path, json.dumps(lock, indent=2) + "\n")):
            with tempfile.NamedTemporaryFile("w", dir=root, prefix=f".{destination.name}.", delete=False) as file:
                file.write(content)
                pending.append((Path(file.name), destination))
            os.chmod(pending[-1][0], destination.stat().st_mode)
        for source, destination in pending:
            os.replace(source, destination)
    except BaseException:
        shutil.copy2(backup / "config.yml", config_path)
        shutil.copy2(backup / "model.lock.json", lock_path)
        raise
    finally:
        for source, _ in pending:
            source.unlink(missing_ok=True)
    return backup


def main(argv: list[str] | None = None, *, root: Path | None = None, runner=subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Catalog branch name")
    parser.add_argument("--cache", help="FP16/Q8/Q6/Q4 or K,V bits from 2 to 8")
    parser.add_argument("--mtp", help="Fixed draft tokens; 0 disables MTP")
    parser.add_argument("--url", help="Plain bind host or http://host[:port][/v1]")
    parser.add_argument("--port", help="Bind port, overrides URL port")
    parser.add_argument("--models-dir", help="Absolute directory containing model directories")
    parser.add_argument("--configure-only", action="store_true", help="Write config and lock without installing")
    parser.add_argument("--dry-run", action="store_true", help="Preview only; no writes or downloads")
    parser.add_argument("--yes", action="store_true", help="Apply without an interactive confirmation")
    args = parser.parse_args(argv)
    root = Path(root or ROOT)
    try:
        catalog = load_catalog(root / "setup" / "models.json")
        current = (root / "config.yml").read_text()
        existing_lock = json.loads((root / "model.lock.json").read_text())
        defaults = config_defaults(current)
        entries = catalog["models"]
        default_entry = next((item for item in entries if item["revision"] == existing_lock.get("revision")), entries[0])
        if args.model is None and not args.yes:
            print("Available model branches:")
            for index, item in enumerate(entries, 1):
                print(f"  {index:2}. {item['label']} ({item['branch']})")
            choice = _ask("Model number or branch", str(entries.index(default_entry) + 1))
            entry = entries[int(choice) - 1] if choice.isdecimal() and 1 <= int(choice) <= len(entries) else next((item for item in entries if item["branch"] == choice), None)
        else:
            entry = next((item for item in entries if item["branch"] == (args.model or default_entry["branch"])), None)
        if entry is None:
            raise ValueError("Unknown model branch; choose a listed branch or number")
        cache = parse_cache(args.cache or (defaults["cache_mode"] if args.yes else _ask("KV cache (FP16/Q8/Q6/Q4 or K,V)", defaults["cache_mode"])))
        mtp = parse_mtp(args.mtp if args.mtp is not None else (defaults["mtp_tokens"] if args.yes else _ask("MTP draft tokens (0 disables)", str(defaults["mtp_tokens"]))))
        address = args.url or (defaults["host"] if args.yes else _ask("Bind host or HTTP URL", defaults["host"]))
        host, embedded_port = parse_endpoint(address, args.port)
        port_default = embedded_port or defaults["port"]
        port = port_default if args.yes or args.port is not None else parse_port(_ask("Bind port", str(port_default)))
        previous_dir = Path(defaults["model_dir"]).expanduser()
        if not previous_dir.is_absolute():
            # TabbyAPI resolves relative model paths from its own directory.
            previous_dir = root / "tabbyAPI" / previous_dir
        reuse_previous = (entry["revision"] == existing_lock.get("revision") and catalog["source"] == existing_lock.get("source") and previous_dir.is_dir() and (previous_dir / defaults["model_name"]).is_dir())
        directory_default = str(previous_dir if reuse_previous else Path.home() / "models")
        directory_answer = args.models_dir or (directory_default if args.yes else _ask("Models directory", directory_default))
        model_dir = Path(directory_answer).expanduser().resolve()
        model_name = entry["directory"]
        if reuse_previous and args.models_dir is None and Path(directory_answer).expanduser().resolve() == previous_dir.resolve():
            # Preserve an installed directory for this exact revision.
            model_name = defaults["model_name"]
        selected = dict(entry, directory=model_name)
        rendered = render_config(current, model_dir=str(model_dir), model_name=model_name, cache_mode=cache, mtp_tokens=mtp, host=host, port=port)
        lock = make_lock(existing_lock, selected, catalog["source"])
        mode = "preview" if args.dry_run else "configure only" if args.configure_only else "full install"
        print(f"\nSetup summary ({mode})")
        print(f"Model: {entry['label']} [{entry['branch']}] at {entry['revision']}")
        print(f"Source: {catalog['source']}")
        print(f"Directory: {model_dir / model_name}")
        display_host = f"[{host}]" if ":" in host else host
        print(f"KV cache: {cache}; MTP tokens: {mtp}; base URL: http://{display_host}:{port}/v1")
        if not args.dry_run and not args.configure_only:
            print("Stop any active service using this environment before full setup; setup will not restart it.")
        if host in ("0.0.0.0", "::"):
            print("The wildcard address binds every network interface; use a specific host to limit exposure.")
        if args.dry_run:
            print("\nProposed config.yml:\n" + rendered)
            print("Proposed model.lock.json:\n" + json.dumps(lock, indent=2))
            print("Preview complete; no files changed.")
            return 0
        if not args.yes and input("Apply this setup? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Cancelled; no files changed.")
            return 0
        if not args.configure_only:
            install(root, source=catalog["source"], entry=selected, model_dir=model_dir, existing_lock=existing_lock, runner=runner)
        backup = _write_transaction(root, rendered, lock)
        print(f"Wrote config.yml and model.lock.json; backup: {backup}")
        print("Setup complete. The server was not started.")
        return 0
    except (ValueError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"setup: {exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nSetup interrupted; rerun to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
