"""Run generated code in a locked-down Docker container (spec S5), via the docker SDK.

Isolation (all verified by tests/sandbox/test_escapes.py):
- no network (`network_mode="none"`), no IPC namespace sharing
- read-only root filesystem; input mounted read-only at /io/in
- no writable host mount at all: `/tmp` is a size-limited tmpfs (`noexec`, `nosuid`), and
  the result is read out of it with `exec cat` while the container is still alive
- non-root user 10001, all Linux capabilities dropped, no-new-privileges
- memory (no swap), CPU, PID and open-file limits; capped container logs
- wall-clock timeout; the container is always force-removed

Generated-code problems (crash, timeout, out of memory, output too large) come back as
`ExecutionResult(ok=False, ...)` so the engine can retry. Sandbox infrastructure problems
(Docker not running, image missing) raise `SandboxUnavailableError`: retrying with the
LLM cannot fix those.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.types import LogConfig, Ulimit

from dataquorum.sandbox.executor import (
    OUTPUT_FILES,
    ExecutionResult,
    InputKind,
    OutputKind,
    function_name,
)

LABEL = "dataquorum.sandbox"
DONE = "__DQ_DONE__"
USER = "10001:10001"

# Runs inside the container: python runner.py <function> <output path> <dataframe|files>
RUNNER = """
import json, os, sys, traceback
import pandas as pd

func_name, out_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
meta = {"ok": False, "error": None}
try:
    namespace = {}
    with open("/io/in/code.py", encoding="utf-8") as f:
        exec(f.read(), namespace)
    if func_name not in namespace:
        raise NameError(f"The code does not define {func_name}(df).")
    if mode == "files":
        out = namespace[func_name]("/io/data")
    else:
        out = namespace[func_name](pd.read_parquet("/io/in/input.parquet").copy())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.endswith(".parquet"):
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{func_name}() returned {type(out).__name__}, not a DataFrame.")
        out.to_parquet(out_path, index=False)
    else:
        if not hasattr(out, "to_json"):
            raise TypeError(f"{func_name}() returned {type(out).__name__}, not a Plotly figure.")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out.to_json())
    meta["ok"] = True
except BaseException as exc:
    meta["error"] = f"{type(exc).__name__}: {exc}"[:2000]
sys.stdout.write("\\n" + json.dumps(meta) + "\\n")
sys.stdout.flush()
"""


class SandboxUnavailableError(RuntimeError):
    """Docker or the sandbox image is not available. Not the generated code's fault."""


class DockerExecutor:
    def __init__(
        self,
        image: str = "dataquorum/sandbox:0.4",
        *,
        timeout_seconds: float = 60,
        memory: str = "1g",
        cpus: float = 1.0,
        pids_limit: int = 64,
        tmp_size_mb: int = 256,
        max_output_mb: int = 100,
        client: docker.DockerClient | None = None,
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.tmp_size_mb = tmp_size_mb
        self.max_output_bytes = max_output_mb * 1024 * 1024
        self._client = client

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException as exc:
                raise SandboxUnavailableError(f"Docker is not reachable: {exc}") from exc
        return self._client

    def run(
        self,
        code: str,
        input_path: Path,
        workdir: Path,
        kind: OutputKind = "dataframe",
        input_kind: InputKind = "dataframe",
    ) -> ExecutionResult:
        in_dir, out_dir = workdir / "in", workdir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (in_dir / "code.py").write_text(code, encoding="utf-8")
        (in_dir / "runner.py").write_text(RUNNER, encoding="utf-8")
        if input_kind == "dataframe":
            shutil.copy(input_path, in_dir / "input.parquet")
        elif input_kind != "files":
            raise ValueError(f"DockerExecutor cannot run {input_kind} input")

        out_name = OUTPUT_FILES[kind]
        inner_path = f"/tmp/out/{out_name}"
        hold = int(self.timeout_seconds) + 30  # keep the container alive to read the output
        script = (
            f"python /io/in/runner.py {function_name(input_kind, kind)} {inner_path} {input_kind}; "
            f"echo {DONE} $?; sleep {hold}"
        )
        start = time.perf_counter()
        data_dir = input_path if input_kind == "files" else None
        container = self._start(script, in_dir, data_dir)
        try:
            finished, exit_code = self._wait(container, start + self.timeout_seconds)
            elapsed = time.perf_counter() - start
            if not finished:
                return ExecutionResult(
                    False, f"Timed out after {self.timeout_seconds:g} s.", None, elapsed
                )
            meta = _last_meta(container.logs(stdout=True, stderr=False, tail=200))
            if meta is None:
                return ExecutionResult(False, _no_result_message(exit_code), None, elapsed)
            if not meta.get("ok"):
                return ExecutionResult(False, str(meta.get("error")), None, elapsed)
            data, problem = self._read_output(container, inner_path)
            if problem:
                return ExecutionResult(False, problem, None, time.perf_counter() - start)
            target = out_dir / out_name
            target.write_bytes(data)
            return ExecutionResult(True, None, target, time.perf_counter() - start)
        finally:
            with contextlib.suppress(DockerException):
                container.remove(force=True)

    def _start(self, script: str, in_dir: Path, data_dir: Path | None = None) -> Container:
        tmpfs_opts = f"rw,nosuid,nodev,noexec,size={self.tmp_size_mb}m,mode=1777"
        kwargs: dict[str, Any] = {
            "detach": True,
            "network_mode": "none",
            "ipc_mode": "none",
            "read_only": True,
            "tmpfs": {"/tmp": tmpfs_opts},
            "volumes": {
                str(in_dir.resolve()): {"bind": "/io/in", "mode": "ro"},
                **(
                    {str(data_dir.resolve()): {"bind": "/io/data", "mode": "ro"}}
                    if data_dir
                    else {}
                ),
            },
            "user": USER,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "mem_limit": self.memory,
            "memswap_limit": self.memory,
            "nano_cpus": int(self.cpus * 1e9),
            "pids_limit": self.pids_limit,
            "ulimits": [Ulimit(name="nofile", soft=256, hard=256)],
            "environment": {},
            "hostname": "sandbox",
            "labels": {LABEL: "1"},
            "log_config": LogConfig(
                type=LogConfig.types.JSON, config={"max-size": "2m", "max-file": "1"}
            ),
        }
        try:
            container: Container = self.client.containers.run(
                self.image, ["sh", "-c", script], **kwargs
            )
        except ImageNotFound as exc:
            raise SandboxUnavailableError(
                f"Sandbox image {self.image} not found. Build it with: "
                f"docker build -t {self.image} -f images/sandbox/Dockerfile ."
            ) from exc
        except DockerException as exc:
            raise SandboxUnavailableError(f"Could not start the sandbox: {exc}") from exc
        return container

    def _wait(self, container: Container, deadline: float) -> tuple[bool, int | None]:
        """Wait for the DONE marker. Returns (finished, runner exit code)."""
        while time.perf_counter() < deadline:
            for line in container.logs(stdout=True, stderr=False, tail=5).splitlines():
                text = line.decode("utf-8", "replace")
                if text.startswith(DONE):
                    parts = text.split()
                    return True, int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            container.reload()
            if container.status == "exited":
                return True, None
            time.sleep(0.2)
        return False, None

    def _read_output(self, container: Container, inner_path: str) -> tuple[bytes, str | None]:
        limit = self.max_output_bytes
        try:
            result = container.exec_run(
                ["head", "-c", str(limit + 1), inner_path], user=USER, demux=True
            )
        except (NotFound, DockerException) as exc:
            return b"", f"Could not read the output: {exc}"
        stdout, _ = cast(tuple[bytes | None, bytes | None], result.output)
        if result.exit_code != 0 or stdout is None:
            return b"", "The code reported success but wrote no output."
        if len(stdout) > limit:
            return b"", f"Output is larger than {limit // (1024 * 1024)} MB."
        return stdout, None


def _last_meta(logs: bytes) -> dict[str, Any] | None:
    for line in reversed(logs.decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"ok"' in line:
            try:
                meta: dict[str, Any] = json.loads(line)
                return meta
            except json.JSONDecodeError:
                continue
    return None


def _no_result_message(exit_code: int | None) -> str:
    if exit_code == 137:
        return "The code was killed (exit 137), most likely for using too much memory."
    if exit_code is not None:
        return f"The code exited abnormally (exit {exit_code}) without a result."
    return "The sandbox stopped without a result."
