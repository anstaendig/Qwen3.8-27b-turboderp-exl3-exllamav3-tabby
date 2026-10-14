# Local deployment update, 2026-10-14

TabbyAPI is now pinned at
[`a3fed43efae3b18ec6181618d6a4a66153a9d14e`](https://github.com/theroyallab/tabbyAPI/tree/a3fed43efae3b18ec6181618d6a4a66153a9d14e),
whose CUDA 13 extra selects ExLlamaV3 1.6.1 with PyTorch 2.11.0+cu130.
It includes [the upstream grammar rejection fix](https://github.com/theroyallab/tabbyAPI/commit/b58614126fd46e909ba92101e9435c64a7df9835).
Malformed constraints raise a client error instead of silently being skipped;
broken grammar-backend dependencies retain server-error classification.

The previous submodule edits were compared byte-for-byte against a clean
checkout plus the two maintained patches before updating. All matched. Receipts
are retained locally under `.state/backend-update/`. No unrelated source edit
was discarded. The rebased metrics patch preserves all public metric families,
labels, bucket boundaries, and event semantics. The EOS patch has updated
context and removes the new upstream warning that incorrectly calls
`ban_eos_token` unsupported. The final thinking patch adds opt-in compatibility
and its native schema field; root config enables it. The model revision, shard
hashes, sampling defaults, cache dimensions, and draft settings are unchanged.

The Python dependency closure is hash-locked, including the HF CLI; the wizard
syncs that lock and uses its installed CLI instead of a separately resolved
tool environment. Docker pins both base images by digest, fetches immutable
upstream source, applies the same patch stack, and installs only at build time.
Compose persists config, API keys, and compilation caches; models are an
external read-only mount. See [packaging](packaging.md) and
[thinking controls](thinking-controls.md).

| Gate | Evidence and status |
| --- | --- |
| Source/config | Pin and revision guards aligned; root config validates against patched native schema. Root/submodule `git diff --check` pass. |
| Reproducible patches | CPU test clones local clean pinned upstream, invokes all three public scripts twice, compares patched files to working source, and reverses to a clean tree. Pass. |
| CPU contracts | `.state/metrics-test-venv/bin/python -m pytest tests -q`: **163 passed**, one existing Starlette/httpx deprecation warning. No GPU dependency installed. |
| Metrics | **14 focused tests pass** for event accounting, setup-failure cleanup, scrape semantics and authenticated HTTP exposition with unrelated/GPU imports stubbed. |
| Thinking | **22 tests pass**, using real request parsing with sampler imports stubbed and the immutable 8,952-byte template fixture. All four request surfaces render mapped/canonical efforts; precedence and invalid effort behavior checked. |
| Grammar rejection | **6 tests pass** executing actual upstream grammar-construction source with a failing filter stub. Live LLGuidance compilation remains pending. |
| Lock | `uv pip install --dry-run --python-version 3.12.13 --python-platform x86_64-manylinux_2_39 --target .state/backend-update/lock-target --require-hashes --only-binary :all: -r requirements/cu13-python312.lock` resolves **91 packages**. This is a resolution dry run, not installation or import proof. |
| Packaging | `docker compose config --quiet` and `docker build --check .` pass; build check reports no warnings. Image digests checked against their registry manifests. No image layers executed. |
| Full image build | **Pending** on Linux amd64. |
| Host installation/cutover | **Pending**; no remote access or host changes performed. Inspect effective service/config before any authorized cutover. |
| Model load/inference | **Pending**; no weights downloaded or inference endpoint called. |
| Persistence under runtime | **Pending**; mounts and paths are configured but a running server has not tested key/cache reuse. |

Historical host acceptance records retain their original scope. No local gate
promotes this update to host/runtime acceptance.
