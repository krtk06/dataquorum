"""Run generated code in a locked-down Docker container (spec S5).

The flags below were verified on 2026-11-16 (network, root-fs writes, input writes, chown,
memory and fork bombs all blocked). Week 5 adds the escape-test suite and the docker SDK.

Data flow: the host writes `code.py`, `runner.py` and `input.parquet` to `<workdir>/in`
(mounted read-only); the container writes its output to `<workdir>/out`.
Image: `docker/sandbox` (pandas, numpy, pyarrow, plotly only).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from dataquorum.sandbox.executor import FUNCTION_NAMES, OUTPUT_FILES, ExecutionResult, OutputKind

# Runs inside the container: python runner.py <function name> <output file>
RUNNER = """
import json, sys, traceback
import pandas as pd

func_name, out_file = sys.argv[1], sys.argv[2]
meta = {"ok": False, "error": None}
try:
    namespace = {}
    with open("/io/in/code.py", encoding="utf-8") as f:
        exec(f.read(), namespace)
    if func_name not in namespace:
        raise NameError(f"The code does not define {func_name}(df).")
    df = pd.read_parquet("/io/in/input.parquet")
    out = namespace[func_name](df.copy())
    if out_file.endswith(".parquet"):
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{func_name}() returned {type(out).__name__}, not a DataFrame.")
        out.to_parquet("/io/out/" + out_file, index=False)
    else:
        if not hasattr(out, "to_json"):
            raise TypeError(f"{func_name}() returned {type(out).__name__}, not a Plotly figure.")
        with open("/io/out/" + out_file, "w", encoding="utf-8") as f:
            f.write(out.to_json())
    meta["ok"] = True
except Exception as exc:
    meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["traceback"] = traceback.format_exc()[-1500:]
print(json.dumps(meta))
"""


def docker_exe() -> str:
    # On Windows, PATH also holds an extension-less `docker` shim that CreateProcess can't run.
    found = shutil.which("docker.exe") or shutil.which("docker")
    return found or r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


@dataclass(frozen=True)
class DockerExecutor:
    image: str = "dataquorum/sandbox:0.1"
    timeout_seconds: int = 60
    memory: str = "1g"
    cpus: str = "1"
    pids_limit: int = 64

    def run(
        self, code: str, input_path: Path, workdir: Path, kind: OutputKind = "dataframe"
    ) -> ExecutionResult:
        in_dir, out_dir = workdir / "in", workdir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (in_dir / "code.py").write_text(code, encoding="utf-8")
        (in_dir / "runner.py").write_text(RUNNER, encoding="utf-8")
        shutil.copy(input_path, in_dir / "input.parquet")

        name = f"dq-exec-{uuid.uuid4().hex[:10]}"
        out_file = OUTPUT_FILES[kind]
        cmd = [
            docker_exe(), "run", "--rm", "--name", name,
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m,mode=1777",
            "--memory", self.memory, "--memory-swap", self.memory, "--cpus", self.cpus,
            "--pids-limit", str(self.pids_limit), "--user", "10001:10001",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{in_dir.resolve()}:/io/in:ro", "-v", f"{out_dir.resolve()}:/io/out",
            self.image, "python", "/io/in/runner.py", FUNCTION_NAMES[kind], out_file,
        ]  # fmt: skip
        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired:
            subprocess.run([docker_exe(), "kill", name], capture_output=True)
            elapsed = time.perf_counter() - start
            return ExecutionResult(
                False, f"Timed out after {self.timeout_seconds} s.", None, elapsed
            )
        seconds = time.perf_counter() - start

        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            error = f"Sandbox gave no result: {proc.stderr[-500:]}"
            return ExecutionResult(False, error, None, seconds)
        meta = json.loads(lines[-1])
        if not meta["ok"]:
            return ExecutionResult(False, meta["error"], None, seconds)
        return ExecutionResult(True, None, out_dir / out_file, seconds)
