"""Fixtures for tests that need Docker and the sandbox image.

Tests marked `docker` are skipped (not failed) when Docker is unreachable or the image has
not been built, so the normal test jobs stay green. The CI `sandbox` job builds the image
and runs them with `pytest -m docker`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import docker
import pandas as pd
import pytest

from dataquorum.sandbox.docker_executor import LABEL, DockerExecutor

IMAGE = "dataquorum/sandbox:0.4"


def _unavailable_reason() -> str | None:
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception as exc:
        return f"Docker or {IMAGE} not available: {type(exc).__name__}"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    reason = _unavailable_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client() -> docker.DockerClient:
    return docker.from_env()


@pytest.fixture
def executor(client: docker.DockerClient) -> DockerExecutor:
    return DockerExecutor(IMAGE, timeout_seconds=20, client=client)


@pytest.fixture
def input_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_parquet(path, index=False)
    return path


@pytest.fixture(autouse=True)
def no_leftover_containers(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if "docker" not in request.keywords or _unavailable_reason():
        return
    left = docker.from_env().containers.list(all=True, filters={"label": LABEL})
    assert left == [], f"sandbox containers left behind: {[c.short_id for c in left]}"
