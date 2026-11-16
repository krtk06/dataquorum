"""DockerExecutor behaviour on normal and badly-behaved (but not malicious) code."""

from __future__ import annotations

import json
import time
from pathlib import Path

import docker
import pandas as pd
import pytest

from dataquorum.sandbox.docker_executor import DockerExecutor, SandboxUnavailableError

pytestmark = pytest.mark.docker

DOUBLE = "def transform(df):\n    return df.assign(a=df['a'] * 2)\n"


def test_dataframe_round_trip(executor: DockerExecutor, input_path: Path, tmp_path: Path) -> None:
    result = executor.run(DOUBLE, input_path, tmp_path)
    assert result.ok, result.error
    assert result.output_path is not None
    assert pd.read_parquet(result.output_path)["a"].tolist() == [2, 4, 6]
    assert not (tmp_path / "in" / "output.parquet").exists()  # nothing written to the host mount


def test_figure_output(executor: DockerExecutor, input_path: Path, tmp_path: Path) -> None:
    code = (
        "import plotly.express as px\n"
        "def make_figure(df):\n"
        "    return px.bar(df, x='b', y='a', title='t')\n"
    )
    result = executor.run(code, input_path, tmp_path, kind="figure")
    assert result.ok, result.error
    assert result.output_path is not None
    figure = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert figure["data"][0]["type"] == "bar"


@pytest.mark.parametrize(
    ("code", "error_part"),
    [
        ("def transform(df):\n    return df['missing']\n", "KeyError"),
        ("def transform(df)\n    return df\n", "SyntaxError"),
        ("def something_else(df):\n    return df\n", "does not define transform"),
        ("def transform(df):\n    return 42\n", "not a DataFrame"),
        ("import sys\ndef transform(df):\n    sys.exit(3)\n", "SystemExit"),
        ("import os\ndef transform(df):\n    os._exit(5)\n", "exit 5"),
    ],
)
def test_bad_code_is_reported_not_raised(
    code: str, error_part: str, executor: DockerExecutor, input_path: Path, tmp_path: Path
) -> None:
    result = executor.run(code, input_path, tmp_path)
    assert not result.ok
    assert result.error is not None and error_part in result.error


def test_infinite_loop_times_out_and_is_cleaned_up(
    client: docker.DockerClient, input_path: Path, tmp_path: Path
) -> None:
    fast = DockerExecutor("dataquorum/sandbox:0.2", timeout_seconds=4, client=client)
    start = time.perf_counter()
    result = fast.run("def transform(df):\n    while True:\n        pass\n", input_path, tmp_path)
    assert not result.ok and result.error == "Timed out after 4 s."
    assert time.perf_counter() - start < 15


def test_stdout_flood_does_not_break_the_run(
    executor: DockerExecutor, input_path: Path, tmp_path: Path
) -> None:
    code = (
        "import sys\n"
        "def transform(df):\n"
        "    for _ in range(200):\n"
        "        sys.stdout.write('x' * 100_000 + '\\n')\n"
        "    return df\n"
    )
    result = executor.run(code, input_path, tmp_path)
    assert result.ok, result.error  # 20 MB of output; logs are capped at 2 MB


def test_output_size_is_capped(
    client: docker.DockerClient, input_path: Path, tmp_path: Path
) -> None:
    small = DockerExecutor("dataquorum/sandbox:0.2", max_output_mb=1, client=client)
    code = (
        "import numpy as np, pandas as pd\n"
        "def transform(df):\n"
        "    return pd.DataFrame({'x': np.random.default_rng(0).random(400_000)})\n"
    )
    result = small.run(code, input_path, tmp_path)
    assert not result.ok and result.error == "Output is larger than 1 MB."


def test_missing_image_raises_unavailable(
    client: docker.DockerClient, input_path: Path, tmp_path: Path
) -> None:
    missing = DockerExecutor("dataquorum/does-not-exist:0", client=client)
    with pytest.raises(SandboxUnavailableError, match="docker build"):
        missing.run(DOUBLE, input_path, tmp_path)


def test_folder_input_is_readable_but_not_writable(
    executor: DockerExecutor, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text("x,y\n1,2\n3,4\n", encoding="utf-8")
    code = (
        "import os\nimport pandas as pd\n"
        "def load(data_dir):\n"
        "    df = pd.read_csv(os.path.join(data_dir, 'a.csv'))\n"
        "    try:\n"
        "        open(os.path.join(data_dir, 'evil.txt'), 'w').write('x')\n"
        "        df['wrote'] = True\n"
        "    except OSError:\n"
        "        df['wrote'] = False\n"
        "    return df\n"
    )
    result = executor.run(code, data, tmp_path / "work", input_kind="files")
    assert result.ok, result.error
    assert result.output_path is not None
    out = pd.read_parquet(result.output_path)
    assert out["x"].tolist() == [1, 3]
    assert not out["wrote"].any()
    assert sorted(p.name for p in data.iterdir()) == ["a.csv"]  # host folder unchanged
