# Local deployment update, 2026-11-15

TabbyAPI is now pinned at
[`47281587fafbd69e50aaf77c7626f390811126b6`](https://github.com/theroyallab/tabbyAPI/tree/47281587fafbd69e50aaf77c7626f390811126b6)
(upstream `main`, 2026-11-14). Its CUDA 13 extra selects
[ExLlamaV3 1.6.7](https://github.com/turboderp-org/exllamav3/releases/tag/v1.6.7),
the latest release, with PyTorch 2.11.0+cu130 unchanged. The update spans 68
upstream commits, including ExLlamaV3 1.6.2 to 1.6.7, a status page and UI
additions, `/v1/decisions`, streamed `tool_calls` argument deltas,
`finish_reason: repetition` for detected loops, and an opt-in upstream `/metrics`.

Before the update, the submodule's edits were compared byte-for-byte against a
clean `a3fed43` checkout plus the four maintained patches. They matched, so no
unrelated local edit was discarded.

## Are the patches still needed?

| Patch | Upstream status at 4728158 | Decision |
| --- | --- | --- |
| `tabbyapi-metrics.patch` | Upstream added an opt-in exporter at `/metrics` with different names (`tabbyapi:` prefix) and semantics: no outcome or state labels, TTFT from rounded phase times, cancellations counted as requests, gauges reset on each scrape. | **Kept and rebased.** Module renamed to `common/prometheus_metrics.py` to avoid upstream's `common/metrics.py`. The route is registered ahead of the core router, so the version-1 contract owns `/metrics`. |
| `tabbyapi-ban-eos.patch` | Unchanged. `ban_eos_token` is still in `UNSUPPORTED_PARAMS`, and EOS IDs are still added to stop conditions unconditionally. | **Kept and rebased** (import context only). |
| `tabbyapi-thinking.patch` | Unchanged. Upstream passes `reasoning_effort` to the template as is; nothing maps `high` → `xhigh` or `none` → `enable_thinking: false`. | **Kept and rebased.** Normalization still runs after the full merge, now including upstream's `PRESERVE_REASONING_VARS`. |
| `tabbyapi-tool-arguments.patch` | Partly covered. Upstream now keeps `string` parameters as strings and streams argument deltas. A declared boolean `False` still becomes `"False"`. Lists of types are ignored, `None` isn't recognized, and string values lose all surrounding whitespace. | **Kept, rewritten** on upstream's `param_types` / delta-streamer design. |

## Tool-argument rewrite

Behavior is unchanged from the previous patch for both parse paths, and it now also
holds for the new streamed-delta path:

- `param_types_from_tools` keeps lists of types instead of mapping them to `None`.
- `coerce_param_value` applies the schema rules: any-case booleans, `null`/`None`,
  integers without fractions, finite numbers, and JSON-compatible objects/arrays via
  JSON then `ast.literal_eval`. String-only values are returned verbatim.
- The Qwen parser and the delta emitter remove only the template's boundary
  newlines. A string value streams from its first final character; one trailing
  newline is held back until later text makes it interior.
- For untyped values, the emitter holds `True`, `None`, `NaN` and similar prefixes
  (in any case) until they can no longer be converted, so streamed fragments stay
  byte-identical to the end-of-stream parse. Untyped `NaN`/`Infinity`, which
  `json.loads` accepts but are not JSON, stay as text.
- Two upstream test expectations in the submodule change accordingly:
  `"\n123\n"` declared `string` stays verbatim, and `True`/`tRu` are buffered.

## Gates

| Gate | Evidence and status |
| --- | --- |
| Source/config | All pins moved (`scripts/apply-*`, `setup/wizard.py`, `tests/test_patch_stack.py`, `Dockerfile`, notices). Root `config.yml` validates against the patched schema with no unknown keys; no config change needed. |
| Reproducible patches | All four public scripts apply to a clean pinned checkout and are idempotent on rerun. The result is byte-identical to the working rebase. `test_patch_stack.py` passes. |
| CPU contracts | `.state/metrics-test-venv/bin/python -m pytest tests -q`: **210 passed**, one existing Starlette/httpx deprecation warning. |
| Metrics | **14 focused tests pass**, including route shadowing of upstream's exporter. ExLlamaV3 1.6.7 source was checked for the EOS timing/prompt/draft fields and the three cache-snapshot keys. |
| Upstream suite on patched tree | CPU-only run excluding four torch-importing files: **395 passed, 1 failed**. The failure, `test_port_probe::test_wildcard_listener_blocks_a_specific_bind`, also fails on clean upstream on macOS and is unrelated. |
| Lock | Regenerated with the documented `uv pip compile ... --upgrade`. Notable changes: ExLlamaV3 1.6.7, fla 0.6.0, transformers 5.23.0, fastapi 0.163.0, pillow 13.0.0; torch 2.11.0 and prometheus-client 0.26.0 unchanged. The hash-required dry run for x86_64-manylinux_2_39 resolves **91 packages**. This is not installation or import proof. |
| Packaging | `docker compose config --quiet` and `docker build --check .` pass with no warnings. No image layers executed. |
| Full image build | **Pending** on Linux amd64. |
| Host installation/cutover | **Pending**. No remote access or host changes. Inspect the effective service before an authorized cutover. |
| Model load/inference | **Pending**. No weights downloaded and no inference endpoint called. Live checks still needed: `/metrics` scrape, `ban_eos_token`, thinking efforts, and the `unlock: false` tool check in both streaming and non-streaming modes. |

Historical host acceptance records keep their original scope. No local gate
promotes this update to host or runtime acceptance.
