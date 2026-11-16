# AutoGluon Tabular runtime for DataQuorum's AutoML benchmark (week 10).
# CPU only. Build:  docker build -t dataquorum/autogluon:1.6.3 docker/autogluon
#
# Run it like the sandbox: no network, read-only root, data in/out through /io.
#   docker run --rm --network none --read-only --tmpfs /tmp:size=512m \
#     --memory 3g --cpus 2 --pids-limit 256 --cap-drop ALL \
#     --security-opt no-new-privileges -v <in>:/io/in:ro -v <out>:/io/out \
#     dataquorum/autogluon:1.6.3 python /io/in/script.py

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# libgomp1: OpenMP runtime needed by LightGBM and XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# CPU-only PyTorch first, so AutoGluon does not pull the multi-GB CUDA build.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# Model families used by AutoGluon's standard presets. Foundation models
# (tabpfn, mitra, tabicl) are left out: multi-GB weights, GPU-oriented.
RUN pip install "autogluon.tabular[lightgbm,catboost,xgboost,fastai]==1.6.3" pyarrow

# Non-root user. HOME and caches point at /tmp because the root filesystem is read-only at run time.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin ag
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache
USER ag
WORKDIR /tmp
