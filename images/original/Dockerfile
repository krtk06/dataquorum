# The original business-science/ai-data-science-team (commit 4ffeb7f), for the parity eval.
# It runs LLM-written code with only an import blocklist as a sandbox (bug log #2-#4), so we
# run the whole project inside this container instead of on the host.
#
# Build context is the local clone:
#   docker build -t dataquorum/original:4ffeb7f -f images/original/Dockerfile .original/ai-data-science-team

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# libgomp1: OpenMP runtime for xgboost, which the original imports.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/original
COPY . /opt/original
# Unpinned upstream requirements; pandas < 3 because the original breaks on pandas 3, and
# IPython because the original imports it without declaring it (bug log #14b).
RUN echo "pandas<3" > /tmp/constraints.txt \
    && pip install -c /tmp/constraints.txt . ipython \
    && rm /tmp/constraints.txt

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin original
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache
USER original
WORKDIR /tmp
