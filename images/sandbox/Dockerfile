# Sandbox image for running LLM-generated pandas / Plotly code (spec S5).
# Small on purpose: only what generated code may import.
#
#   docker build -t dataquorum/sandbox:0.2 images/sandbox
#
# Run flags live in src/dataquorum/sandbox/docker_executor.py (no network, read-only root,
# non-root user, all capabilities dropped, memory/CPU/PID limits, timeout).

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache
USER sandbox
WORKDIR /tmp
