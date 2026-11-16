# H2O + FLAML runtime for the week-10 AutoML benchmark (AutoGluon uses images/autogluon).
# CPU only, runs offline. Not built yet:
#   docker build -t dataquorum/automl:0.1 images/automl

FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# H2O needs a Java runtime; libgomp1 for LightGBM/XGBoost. Base is Debian trixie.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin automl
ENV HOME=/tmp XDG_CACHE_HOME=/tmp/.cache
USER automl
WORKDIR /tmp
