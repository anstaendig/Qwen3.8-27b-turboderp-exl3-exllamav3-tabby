# Dependency locking and Docker packaging

Supported target: Linux x86_64, glibc ≥ 2.31, CPython 3.12.13, CUDA 13.
The lock was resolved for Ubuntu 24.04's glibc 2.39. LLGuidance's CPython 3.12
wheel needs glibc 2.31; do not use the lock for macOS, ARM, or another Python ABI.

`requirements/cu13-python312.lock` contains exact versions/direct wheel URLs and
SHA-256 hashes for the full runtime dependency closure, including the HF CLI.
Both the native setup wizard and Docker use `uv pip sync --require-hashes
--only-binary :all:`. There is no editable project install or independent,
unlocked `uvx` tool environment. TabbyAPI runs directly from patched source.
Embeddings extras are outside this deployment and are not installed.

To intentionally refresh the lock with uv 0.11.14, first apply all four patches:

```sh
./scripts/apply-metrics
./scripts/apply-ban-eos
./scripts/apply-thinking
./scripts/apply-tool-arguments
uv pip compile tabbyAPI/pyproject.toml --extra cu13 \
  -c requirements/constraints.txt --python-version 3.12.13 \
  --python-platform x86_64-manylinux_2_39 --generate-hashes \
  --only-binary :all: --no-emit-package tabbyapi --upgrade \
  -o requirements/cu13-python312.lock
```

The Transformers lower bound prevents the resolver from satisfying FLA with an
obsolete release when HF dependency constraints change. Review the complete
lock diff when refreshing. Native installation requires an explicit setup run;
startup does not install anything. Installation replaces the environment's
packages to match the lock, so stop its service before running setup.

## Container state

The Dockerfile pins `nvidia/cuda:13.2.1-runtime-ubuntu24.04` and `uv:0.11.14` by
registry digest. It fetches only the immutable TabbyAPI commit, applies the
patches in order, installs Python 3.12.13 with development headers and the locked
wheels during build, then starts upstream `main.py`. Compilation tools remain
available for Triton/runtime JIT; these compile kernels, not install packages.
The build context whitelist excludes local environments, Git state, config,
API tokens, model files, and caches.

Compose persists:

| State | Location | Behavior |
| --- | --- | --- |
| Config | Host `./config.yml` → `/state/config.yml` | Read-only; survives rebuild/recreation on the host. |
| Model weights | Host `MODELS_DIR` → `/app/tabbyAPI/models` | Read-only; must already contain the configured `model_name`. |
| API/admin keys | Named `auth` volume → `/state/auth` | TabbyAPI's working-directory `api_tokens.yml` is a symlink into it. First start generates keys once. |
| Torch/Triton/CUDA caches | Named `compilation-cache` volume → `/cache` | Explicit cache environment paths; survive container recreation. |

Compose binds only `127.0.0.1:8080` on the host. Inside the container, CLI arguments
override host to `0.0.0.0`, port to `8080`, and model parent directory to
`/app/tabbyAPI/models`, so wizard-written host paths cannot break that mount.
All other settings still come from the mounted config. Model loading can also
read a checkpoint's `tabby_config.yml`; inspect any inline overrides before a
host cutover, as upstream supports those taking precedence over root config.

```sh
MODELS_DIR=/absolute/path/to/models docker compose build
MODELS_DIR=/absolute/path/to/models docker compose up -d
docker compose logs -f tabbyapi
```

To retain an existing server identity, copy its private `api_tokens.yml` into
the auth volume **before** first startup. Do not bake keys into the image. A
new empty volume creates a new identity. Keep that volume private and back it
up according to your normal secret-handling process. `docker compose down`
retains volumes; `down -v` deletes keys and compilation caches. Check that
your installed NVIDIA driver/container toolkit accepts the pinned CUDA image
before launching. This work does not change the GPU host.

## Verification boundaries

Local checks cover clean patch application/idempotence/reversal, request
parsing and immutable template rendering, metric event/HTTP behavior with
CPU stubs, hash-enforced dependency resolution in dry-run mode, Compose
validation, and Dockerfile build checks. `docker build --check` executes no
installation layers and does not produce an image.

A full Linux amd64 image build, host package installation, GPU model load,
inference, structured-output runtime rejection, and persistent-volume behavior
under a running server are separate pending gates. The pinned image does not
freeze apt repositories, the uv Python download artifact, compiler output,
host driver, or model files. The model has its own immutable lock; Python
dependencies and container base images have the pins described above.
