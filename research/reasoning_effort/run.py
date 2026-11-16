"""Reasoning-effort experiment (week 2): cost, latency and correctness per effort level.

Each run asks DeepSeek V4.1 Flash for a cleaning plan + `clean(df)` via LLMClient, runs the
code in the Docker sandbox (no network, read-only, non-root, 512 MB), and checks the output
against the known answer (harness.py). Levels are interleaved so time-of-day drift is shared.

    uv run python research/reasoning_effort/run.py --runs 8
    uv run python research/reasoning_effort/run.py --analyse-only
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dataquorum.core.budget import BudgetGuard, JsonlLedger
from dataquorum.core.llm import LLMClient, LLMConfig, LLMError

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"
IMAGE = "dataquorum/autogluon:1.6.3"  # has pandas 2.3.3; any pandas image works
LEVELS = ["default", "minimal", "low", "medium", "high"]

SYSTEM = (
    "You are a data-cleaning agent. Return the cleaning steps and Python code that defines "
    "`clean(df: pd.DataFrame) -> pd.DataFrame`. The code may import only pandas and numpy, "
    "must not read or write files, and must return a new DataFrame."
)
USER = """The DataFrame `df` has 12 rows and these columns:
- age: float with some NaN. Leave it exactly as it is (no imputation).
- income: strings such as "$1,200", "$3.4k", " $2,050 ", "980", "1,750.50", "n/a", "N/A", "",
  and None. Convert to float dollars ("$3.4k" means 3400). Missing or unparseable values
  ("n/a", "N/A", "", None) become NaN.
- city: strings such as "new york", "NEW YORK", "boston ", " Chicago", "san  francisco".
  Strip outer whitespace, collapse repeated inner spaces to one, and use title case.
Keep all 12 rows and all 3 columns, in the same order."""


class CleaningPlan(BaseModel):
    steps: list[str]
    python_code: str


def docker_exe() -> str:
    # On Windows, PATH also holds an extension-less `docker` shim that CreateProcess can't run.
    found = shutil.which("docker.exe") or shutil.which("docker")
    fallback = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe")
    return found or str(fallback)


def run_in_sandbox(code: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(dir=HERE.parent.parent / "runs") as tmp:
        work = Path(tmp)
        (work / "code.py").write_text(code, encoding="utf-8")
        shutil.copy(HERE / "harness.py", work / "harness.py")
        name = f"dq-effort-{uuid.uuid4().hex[:8]}"
        cmd = [
            docker_exe(), "run", "--rm", "--name", name, "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=64m,mode=1777", "--memory", "512m", "--memory-swap",
            "512m", "--cpus", "1", "--pids-limit", "64", "--user", "10001:10001",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{work}:/io/in:ro", IMAGE, "python", "/io/in/harness.py",
        ]  # fmt: skip
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            subprocess.run([docker_exe(), "kill", name], capture_output=True)
            return {"checks": {}, "passed": False, "error": "timeout (60 s)"}
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            return {"checks": {}, "passed": False, "error": f"no output: {proc.stderr[-300:]}"}
        result: dict[str, Any] = json.loads(lines[-1])
        return result


def one_run(client: LLMClient, level: str, index: int) -> dict[str, Any]:
    row: dict[str, Any] = {"level": level, "index": index,
                           "started": datetime.now(UTC).isoformat()}  # fmt: skip
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    try:
        result = client.structured(messages, CleaningPlan, reasoning_effort=level)  # type: ignore[arg-type]
    except LLMError as exc:
        usage = exc.usage
        row.update(schema_valid=False, passed=False, error=str(exc)[:300], code=None)
    else:
        usage = result.usage
        sandbox = run_in_sandbox(result.value.python_code)
        row.update(schema_valid=True, code=result.value.python_code, **sandbox)
    row.update(
        prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens, allowance_usd=usage.allowance_usd,
        list_cost_usd=usage.list_cost_usd, seconds=round(usage.seconds, 2),
        calls=usage.calls, retries=usage.retries,
    )  # fmt: skip
    return row


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def analyse(rows: list[dict[str, Any]]) -> str:
    out = ["| Effort | Pass | 95% CI | Median out tokens | Median reasoning | "
           "Median s | Mean allowance $ | Schema fails |",
           "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for level in ["default", "minimal", "low", "medium", "high"]:
        rs = [r for r in rows if r["level"] == level]
        if not rs:
            continue
        n, ok = len(rs), sum(r["passed"] for r in rs)
        lo, hi = wilson(ok, n)
        out.append(
            f"| {level} | {ok}/{n} | {lo:.0%}-{hi:.0%} "
            f"| {statistics.median(r['completion_tokens'] for r in rs):,.0f} "
            f"| {statistics.median(r['reasoning_tokens'] for r in rs):,.0f} "
            f"| {statistics.median(r['seconds'] for r in rs):.1f} "
            f"| {statistics.mean(r['allowance_usd'] for r in rs):.5f} "
            f"| {sum(not r['schema_valid'] for r in rs)} |"
        )
    failed = [r for r in rows if not r["passed"]]
    out.append("\nFailed checks:")
    for r in failed:
        bad = [k for k, v in (r.get("checks") or {}).items() if not v]
        out.append(f"- {r['level']} #{r['index']}: {bad or ''} {r.get('error') or ''}".rstrip())
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=8, help="runs per effort level")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    if not args.analyse_only:
        config = LLMConfig.from_env().model_copy(update={"fallback_model": None})
        budget = BudgetGuard(JsonlLedger(), mode="eval")
        client = LLMClient(config, budget=budget, session_id=f"dq-effort-{uuid.uuid4()}")
        jobs = [(lvl, i) for i in range(args.runs) for lvl in LEVELS]  # interleaved
        (HERE.parent.parent / "runs").mkdir(exist_ok=True)
        with ThreadPoolExecutor(args.workers) as pool, RESULTS.open("a", encoding="utf-8") as f:
            for row in pool.map(lambda job: one_run(client, *job), jobs):
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{row['level']:>8} #{row['index']}: passed={row['passed']} "
                      f"out={row['completion_tokens']} {row['seconds']}s")  # fmt: skip

    rows = [json.loads(ln) for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln]
    print(analyse(rows))


if __name__ == "__main__":
    main()
