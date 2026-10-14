# CUDA and uv are pinned by registry digest; Linux amd64 is the supported target.
FROM ghcr.io/astral-sh/uv:0.11.14@sha256:1025398289b62de8269e70c45b91ffa37c373f38118d7da036fb8bb8efc85d97 AS uv
FROM nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:4c17c2e7474ca1d470428b25e87e95756f51e7da3b663859894a03bc5a7ebdae
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates git && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.12.13 && uv venv /opt/venv -p 3.12.13
WORKDIR /app
COPY scripts/apply-metrics scripts/apply-ban-eos scripts/apply-thinking /app/scripts/
COPY patches/ /app/patches/
RUN git init tabbyAPI && git -C tabbyAPI fetch --depth=1 \
    https://github.com/theroyallab/tabbyAPI.git a3fed43efae3b18ec6181618d6a4a66153a9d14e \
    && git -C tabbyAPI checkout --detach FETCH_HEAD \
    && scripts/apply-metrics && scripts/apply-ban-eos && scripts/apply-thinking
COPY requirements/cu13-python312.lock /app/requirements/cu13-python312.lock
RUN uv pip sync --python /opt/venv/bin/python --require-hashes --only-binary :all: \
    /app/requirements/cu13-python312.lock && uv cache clean \
    && mkdir -p /state/auth /cache/torch /cache/triton /cache/cuda \
    && ln -s /state/auth/api_tokens.yml /app/tabbyAPI/api_tokens.yml
ENV PATH=/opt/venv/bin:$PATH \
    TORCH_EXTENSIONS_DIR=/cache/torch TRITON_CACHE_DIR=/cache/triton CUDA_CACHE_PATH=/cache/cuda
WORKDIR /app/tabbyAPI
EXPOSE 8080
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "/state/config.yml", "--host", "0.0.0.0", "--port", "8080", "--model-dir", "/app/tabbyAPI/models"]
