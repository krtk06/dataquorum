"""Run generated `transform(df)` code in a locked-down Docker container (spec S5).

Week-3 version: the flags below were verified on 2026-11-16 (network, root-fs writes,
input writes, chown, memory and fork bombs all blocked). Week 5 adds the escape-test
suite, a dedicated pinned image and the docker SDK instead of the CLI.

Data flow: host writes `code.py`, `runner.py` and `input.parquet` to `<workdir>/in`
(mounted read-only); the container writes `output.parquet` to `<workdir>/out`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from dataquorum.sandbox.executor import FUNCTION_NAME, ExecutionResult

RUNNER = f"""
import json, traceback
import pandas as pd

meta = {{"ok": False, "error": None}}
try:
    namespace = {{}}
    with open("/io/in/code.py", encoding="utf-8") as f:
        exec(f.read(), namespace)
    if "{FUNCTION_NAME}" not in namespace:
        raise NameError("The code does not define {FUNCTION_NAME}(df).")
    df = pd.read_parquet("/io/in/input.parquet")
    out = namespace["{FUNCTION_NAME}"](df.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"{FUNCTION_NAME}() returned {{type(out).__name__}}, not a DataFrame.")
    out.to_parquet("/io/out/output.parquet", index=False)
    meta["ok"] = True
except Exception as exc:
    meta["error"] = f"{{type(exc).__name__}}: {{exc}}"
    meta["traceback"] = traceback.format_exc()[-1500:]
print(json.dumps(meta))
"""


def docker_exe() -> str:
    # On Windows, PATH also holds an extension-less `docker` shim that CreateProcess can't run.
    found = shutil.which("docker.exe") or shutil.which("docker")
    return found or r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


@dataclass(frozen=True)
class DockerExecutor:
    image: str = "dataquorum/autogluon:1.6.3"  # any image with pandas + pyarrow
    timeout_seconds: int = 60
    memory: str = "1g"
    cpus: str = "1"
    pids_limit: int = 64

    def run(self, code: str, input_path: Path, workdir: Path) -> ExecutionResult:
        in_dir, out_dir = workdir / "in", workdir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (in_dir / "code.py").write_text(code, encoding="utf-8")
        (in_dir / "runner.py").write_text(RUNNER, encoding="utf-8")
        shutil.copy(input_path, in_dir / "input.parquet")

        name = f"dq-exec-{uuid.uuid4().hex[:10]}"
        cmd = [
            docker_exe(), "run", "--rm", "--name", name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--memory", self.memory, "--memory-swap", self.memory, "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit), "--user", "10001:10001",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{in_dir.resolve()}:/io/in:ro", "-v", f"{out_dir.resolve()}:/io/out",
            self.image, "python", "/io/in/runner.py",
        ]  # fmt: skip
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            subprocess.run([docker_exe(), "kill", name], capture_output=True)
            return ExecutionResult(
                False,
                f"Timed out after {self.timeout_seconds} s.",
                None,
                time.perf_counter() - start,
            )
        seconds = time.perf_counter() - start

        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            return ExecutionResult(
                False, f"Sandbox gave no result: {proc.stderr[-500:]}", None, seconds
            )
        meta = json.loads(lines[-1])
        if not meta["ok"]:
            return ExecutionResult(False, meta["error"], None, seconds)
        return ExecutionResult(True, None, out_dir / "output.parquet", seconds)
