# Qwen3.8 27B EXL3 on TabbyAPI, with Prometheus metrics

A reproducible, self-hosted deployment of
[Qwen3.8 27B](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) (EXL3 quantisation by
turboderp) on [TabbyAPI](https://github.com/theroyallab/tabbyAPI) with the
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) backend. It gives you an
OpenAI-compatible API on a single NVIDIA GPU, plus a Prometheus `/metrics` endpoint
for monitoring.

## What it is for

- **Run a local OpenAI-compatible LLM server** for chat, coding agents, tool calling,
  and vision, using any client that speaks the OpenAI API.
- **Pin everything.** TabbyAPI, the Python/CUDA wheels, and the model weights are all
  pinned to exact revisions, so an install today matches one next month.
- **Observe the server.** A small patch adds `GET /metrics` in Prometheus format:
  active/completed requests, prompt and output tokens, time to first token,
  prefill/decode timing, KV cache usage, and MTP (speculative decoding) acceptance.

The default configuration targets a 32 GB GPU (developed on an RTX 5090): the 5.00 bpw
H6 quant, 262,144-token context, an `8,4` quantised KV cache, vision enabled, and
in-checkpoint MTP drafting with three draft tokens. Smaller cards can pick a
lower-bpw quant and/or a shorter context.

## How it works

```
this repository
├── tabbyAPI/            git submodule, pinned to upstream commit f07131c
├── patches/             source patches applied to that exact commit
│   ├── tabbyapi-metrics.patch   adds GET /metrics (prometheus-client 0.26.0)
│   └── tabbyapi-ban-eos.patch   honours ban_eos_token for implicit EOS stop IDs
├── config.yml           native TabbyAPI config (TabbyAPI reads it directly)
├── model.lock.json      model source, immutable revision, and shard SHA-256s
├── setup/               setup wizard + catalog of all published quant branches
├── scripts/             setup, apply-metrics, apply-ban-eos
├── systemd/             example systemd user unit
├── docs/                /metrics contract and design research
└── tests/               CPU-only tests (no GPU, model, or network needed)
```

1. The `tabbyAPI/` submodule is upstream TabbyAPI, unmodified in Git.
2. `scripts/apply-metrics` and `scripts/apply-ban-eos` check that the submodule is at
   the pinned commit and apply the patches exactly once. They refuse to run over
   unrelated local edits. ExLlamaV3 itself is not modified.
3. Dependencies come from TabbyAPI's own `cu13` extra (PyTorch 2.11.0 + CUDA 13.0 and
   the official ExLlamaV3 v1.5.1 wheel), installed into `tabbyAPI/venv`.
4. Model weights are downloaded from Hugging Face at an immutable commit, never a moving
   branch head.
5. TabbyAPI is started with its normal `main.py`, pointed at the root `config.yml`.

The **EOS patch** fixes one behaviour: with `ban_eos_token: true` in a request, the
pinned backend still stopped on the model's implicit EOS stop IDs. With the patch it
drops those but keeps any stop strings or tokens the request supplies. Requests that
don't set `ban_eos_token` behave exactly as before.

## Requirements

- Linux x86_64 with an NVIDIA GPU and a driver that supports CUDA 13 (driver ≥ 580).
  The default config needs about 32 GB of VRAM.
- Git and [`uv`](https://docs.astral.sh/uv/). The wizard downloads Python 3.12.13 and
  the Hugging Face CLI through `uv`, so a system Python 3.10+ is enough to start it.
- Disk space for the model: roughly 5–22 GB depending on the quant.

## Installation

### 1. Clone

```sh
git clone --recurse-submodules https://github.com/anstaendig/Qwen3.8-27b-turboderp-exl3-exllamav3-tabby.git
cd Qwen3.8-27b-turboderp-exl3-exllamav3-tabby
```

If you cloned without `--recurse-submodules`, run `git submodule update --init --checkout`.

### 2a. Setup wizard (recommended)

```sh
./scripts/setup --dry-run   # preview only: shows the config it would write, changes nothing
./scripts/setup             # configure, install dependencies, download the model
```

The wizard asks for:

| Prompt | Meaning |
| --- | --- |
| Model | One of the 25 published quant branches in [`setup/models.json`](setup/models.json) (1.4–6.0 bpw, plain and self-calibrated). Each maps to an immutable revision. |
| KV cache | `FP16`, `Q8`, `Q6`, `Q4`, or separate K,V bits from 2 to 8, e.g. `8,4`. |
| MTP draft tokens | `0` turns speculative drafting off; a positive number sets a fixed draft length. |
| Bind host / port | The address the server **listens on**. Default is `127.0.0.1:8080` (this machine only). |
| Models directory | An absolute path; default `~/models`. Each quant gets its own subdirectory. |

It then patches the submodule, creates `tabbyAPI/venv`, installs the pinned
dependencies, downloads the selected model revision, and writes `config.yml` and
`model.lock.json`. The previous versions of both files are backed up under `.state/setup/`
(ignored by Git). It does **not** start the server or install a service. Everything
else in `config.yml` (context length, vision, reasoning, tool format, auth) is left as is.

Non-interactive example:

```sh
./scripts/setup --yes \
  --model SC_5.00bpw_H6 --cache 8,4 --mtp 3 \
  --url 127.0.0.1 --port 8080 --models-dir "$HOME/models"
```

`--configure-only` writes `config.yml` and `model.lock.json` without installing or
downloading anything. The wizard does not check whether a choice fits in VRAM; a large
quant with the full 262k context may need `max_seq_len`/`cache_size` lowered in
`config.yml`.

### 2b. Manual installation

These steps reproduce the default configuration (5.00 bpw H6) by hand:

```sh
git submodule update --init --checkout
./scripts/apply-metrics
./scripts/apply-ban-eos

cd tabbyAPI
uv python install 3.12.13
uv venv venv -p 3.12.13
uv pip install --python venv/bin/python -e '.[cu13]'

# config.yml uses model_dir "models", which resolves to tabbyAPI/models
uvx --from 'huggingface_hub==1.2.3' hf download turboderp/Qwen3.8-27B-exl3 \
  --revision dfe2f4fd71bf9a1994051d4d47e6460fa96dd723 \
  --local-dir models/qwen3-8-27b-exl3-sc-5-00bpw-h6
```

Compare the SHA-256 of each downloaded `*.safetensors` shard with
`weight_shards_sha256` in [`model.lock.json`](model.lock.json) (`sha256sum models/qwen3-8-27b-exl3-sc-5-00bpw-h6/*.safetensors`).

Notes:

- Apply both patches **before** installing dependencies, so `prometheus-client` is included.
- Use the uv-managed Python rather than a distro Python: Triton compiles a small C
  helper at model warm-up and needs the Python development headers.
- Use a fresh virtual environment. Don't upgrade an existing CUDA 12 environment in place.
- TabbyAPI's own `start.sh`/`start.py` also install and update packages. This project
  starts `main.py` directly so that installation and startup stay separate.

## Running

```sh
cd tabbyAPI
venv/bin/python main.py --config ../config.yml
```

The first start loads the model (this takes a while). The OpenAI-compatible base URL
is `http://127.0.0.1:8080/v1`, and the model ID is the value of `model_name` in `config.yml`.

### API keys

Authentication is **on** by default. On first start TabbyAPI generates
`tabbyAPI/api_tokens.yml` holding an `api_key` (for inference) and an `admin_key` (for
model load/unload). The file is ignored by Git; keep it private. Send the key as
`Authorization: Bearer <key>` or `x-api-key: <key>`:

```sh
KEY=$(awk '/^api_key:/ {print $2}' tabbyAPI/api_tokens.yml)
curl -fsS http://127.0.0.1:8080/health
curl -fsS -H "Authorization: Bearer $KEY" http://127.0.0.1:8080/v1/model
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "qwen3-8-27b-exl3-sc-5-00bpw-h6", "messages": [{"role": "user", "content": "Hello"}]}'
```

For any OpenAI SDK, set `base_url` to `http://127.0.0.1:8080/v1` and `api_key` to that key.

### Run as a service (optional)

[`systemd/exl3-tabbyapi.service`](systemd/exl3-tabbyapi.service) is a systemd user unit.
It expects the checkout at `~/Qwen3.8-27b-turboderp-exl3-exllamav3-tabby`; edit
`WorkingDirectory` and `ExecStart` if yours lives elsewhere.

```sh
mkdir -p ~/.config/systemd/user
cp systemd/exl3-tabbyapi.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now exl3-tabbyapi.service
loginctl enable-linger "$USER"      # optional: start at boot without logging in
journalctl --user -u exl3-tabbyapi.service -f
```

Stop the service before re-running the setup wizard, since it changes the environment
the service uses.

## Prometheus metrics

The metrics patch adds `GET /metrics` on the same host and port as the API, outside
`/v1`. When authentication is enabled it requires the API key, like every other route:

```sh
curl -fsS -H "Authorization: Bearer $KEY" http://127.0.0.1:8080/metrics
```

Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: tabbyapi
    static_configs:
      - targets: ["127.0.0.1:8080"]
    authorization:
      credentials_file: /path/to/tabby-api-key   # file containing only the api_key
```

Exported families (all prefixed `tabbyapi_`):

| Metric | What it measures |
| --- | --- |
| `model_loaded` | Whether a model is loaded |
| `requests_active` / `requests_total` | Jobs by state (queued, prefill, decode) and by outcome |
| `prompt_tokens_total` | Prompt tokens, split into computed and cache-reused |
| `output_tokens_total` | Generated tokens |
| `engine_time_to_first_token_seconds` | Time to first token (histogram) |
| `engine_phase_duration_seconds` | Queue, prefill, decode, and total durations (histogram) |
| `kv_cache_capacity_tokens` / `_used_tokens` / `_reusable_tokens` | Live KV cache occupancy |
| `mtp_draft_tokens_total` | MTP draft tokens, accepted vs rejected |
| `metrics_observation_errors_total`, `metrics_schema_info` | Self-diagnostics and schema version |

Labels are bounded: there are no request IDs, prompt text, IP addresses, keys, or
paths. Counters reset only when the process restarts. The full contract (units,
labels, and when a value is absent versus zero) is in
[`docs/metrics-contract.md`](docs/metrics-contract.md). The design research comparing
vLLM, SGLang, and llama.cpp metrics is in [`docs/research/`](docs/research/).

## Security

- **Defaults:** the server binds to `127.0.0.1` with API-key authentication on, and
  `disable_fetch_requests: true` stops the server from fetching remote URLs (such as
  image links) on a client's behalf.
- **Remote access:** there is no TLS. To reach the server from another machine, keep
  auth on and put it behind a VPN, SSH tunnel, or TLS reverse proxy rather than binding
  `0.0.0.0` on an untrusted network. If you set `disable_auth: true`, anyone who can
  reach the port can run inference and read `/metrics`.
- **Secrets:** `api_tokens.yml` holds your keys and is ignored by Git. Delete it and
  restart to rotate them.
- **Supply chain:** the submodule, the patch base, the wheels, and the model revision
  are pinned. The setup wizard only accepts catalog entries with 40-character commit
  hashes. Transitive Python dependencies follow TabbyAPI's constraints and are not fully
  locked.

## Development and tests

The tests need no GPU, no model, and no network. They apply the patches to the pinned
submodule and exercise the metrics code and the wizard with stubs:

```sh
git submodule update --init --checkout
./scripts/apply-metrics && ./scripts/apply-ban-eos
uv venv .venv -p 3.12
uv pip install --python .venv/bin/python pytest pydantic PyYAML==6.0.2 \
  prometheus-client==0.26.0 fastapi httpx loguru
.venv/bin/python -m pytest tests -q
```

To go back to a clean submodule, run `git -C tabbyAPI checkout -- . && git -C tabbyAPI clean -fd`.
Patch changes must keep applying to the pinned commit; regenerate them with
`git -C tabbyAPI diff` from a clean checkout plus your edits.

## Limitations

- Linux x86_64 and CUDA 13 only for a full install. On other platforms the wizard can
  only preview or configure.
- The wizard doesn't estimate whether a model/cache/context combination fits in VRAM.
- Not every quant/cache/MTP combination in the catalog has been load-tested.
- `/metrics` covers one server process. For multi-process setups, scrape each process.

## License

The code, configuration, and docs in this repository are [MIT-licensed](LICENSE).
The files in [`patches/`](patches/) modify AGPL-3.0 TabbyAPI and are distributed under
AGPL-3.0. TabbyAPI, ExLlamaV3, the Prometheus client, and the model weights keep their
own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This project is not affiliated with TabbyAPI, ExLlamaV3, turboderp, or Qwen.
