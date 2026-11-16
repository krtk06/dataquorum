"""Run LLM-written `transform(df)` code in the Docker sandbox (throwaway version of S5).

Host side writes `code.py` and `input.parquet` into a read-only mount. A tiny runner
inside the container calls `transform(df)` and writes `output.parquet` to a writable
mount, then prints one JSON line. The week-5 `sandbox/docker_executor.py` replaces this.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

IMAGE = "dataquorum/autogluon:1.6.3"  # any image with pandas + pyarrow works

RUNNER = """
import json, traceback
import pandas as pd

meta = {"ok": False, "error": None, "traceback": None}
try:
    namespace = {}
    with open("/io/in/code.py", encoding="utf-8") as f:
        exec(f.read(), namespace)
    if "transform" not in namespace:
        raise NameError("The code does not define transform(df).")
    df = pd.read_parquet("/io/in/input.parquet")
    out = namespace["transform"](df.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"transform() returned {type(out).__name__}, not a DataFrame.")
    out.to_parquet("/io/out/output.parquet", index=False)
    meta["ok"] = True
except Exception as exc:
    meta["error"] = f"{type(exc).__name__}: {exc}"
    meta["traceback"] = traceback.format_exc()[-1500:]
print(json.dumps(meta))
"""


@dataclass(frozen=True)
class SandboxRun:
    ok: bool
    error: str | None
    output_path: Path | None
    seconds: float


def docker_exe() -> str:
    # On Windows, PATH also holds an extension-less `docker` shim that CreateProcess can't run.
    found = shutil.which("docker.exe") or shutil.which("docker")
    return found or r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"


def run_transform(code: str, input_path: Path, workdir: Path, timeout: int = 60) -> SandboxRun:
    """Run `transform(df)` on the Parquet file at `input_path`. Never raises for bad code."""
    in_dir, out_dir = workdir / "in", workdir / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / "code.py").write_text(code, encoding="utf-8")
    (in_dir / "runner.py").write_text(RUNNER, encoding="utf-8")
    shutil.copy(input_path, in_dir / "input.parquet")

    name = f"dq-mini-{uuid.uuid4().hex[:8]}"
    cmd = [
        docker_exe(), "run", "--rm", "--name", name, "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--memory", "512m", "--memory-swap", "512m",
        "--cpus", "1", "--pids-limit", "64", "--user", "10001:10001", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", f"{in_dir.resolve()}:/io/in:ro", "-v", f"{out_dir.resolve()}:/io/out",
        IMAGE, "python", "/io/in/runner.py",
    ]  # fmt: skip
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run([docker_exe(), "kill", name], capture_output=True)
        return SandboxRun(False, f"Timed out after {timeout} s.", None, time.perf_counter() - start)
    seconds = time.perf_counter() - start

    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return SandboxRun(False, f"Sandbox gave no result: {proc.stderr[-500:]}", None, seconds)
    meta = json.loads(lines[-1])
    if not meta["ok"]:
        return SandboxRun(False, meta["error"], None, seconds)
    return SandboxRun(True, None, out_dir / "output.parquet", seconds)
