"""Runs ONE eval task with the original project's agent, inside dataquorum/original.

Input:  /job/job.json (agent, request, model, base_url, reasoning_effort, max_retries)
        /job/input.parquet
        /job/secret: the API key. Read and deleted before any generated code runs, so it is
        not in this process's environment (generated code runs in a child process and could
        read /proc/<parent>/environ, and this container has network access to reach Go).
Output: /job/out/result.json, plus /job/out/output.parquet (tables) or figure.json (charts)

The original's `invoke_agent(user_instructions=...)` crashes after doing all the work (bug
log #18), so this calls the compiled graph directly with a proper HumanMessage. Explaining
the code is bypassed (it does not change the output; DataQuorum has no such step either).
"""

import json
import time
import traceback
from pathlib import Path

import pandas as pd
import plotly.utils
from ai_data_science_team.agents import (
    DataCleaningAgent,
    DataVisualizationAgent,
    DataWranglingAgent,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

AGENTS = {
    "data_cleaning": (DataCleaningAgent, "data_cleaned", "data_cleaner_error"),
    "data_wrangling": (DataWranglingAgent, "data_wrangled", "data_wrangler_error"),
    "data_visualization": (DataVisualizationAgent, "plotly_graph", "data_visualization_error"),
}


class Usage(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls = self.prompt = self.completion = self.reasoning = 0

    def on_llm_end(self, response, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        for generations in response.generations:
            for gen in generations:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.prompt += meta.get("input_tokens", 0)
                self.completion += meta.get("output_tokens", 0)
                self.reasoning += (meta.get("output_token_details") or {}).get("reasoning", 0)


def main() -> None:
    secret = Path("/job/secret")
    api_key = secret.read_text(encoding="utf-8").strip()
    secret.unlink()
    job = json.loads(Path("/job/job.json").read_text(encoding="utf-8"))
    out = Path("/job/out")
    out.mkdir(exist_ok=True)
    usage = Usage()
    extra = (
        {}
        if job["reasoning_effort"] == "default"
        else {"reasoning_effort": job["reasoning_effort"]}
    )
    llm = ChatOpenAI(
        model=job["model"],
        base_url=job["base_url"],
        api_key=api_key,
        default_headers={
            "User-Agent": "dataquorum-parity/0.1",
            "x-opencode-session": job["session"],
        },
        timeout=180,
        max_retries=2,
        callbacks=[usage],
        **extra,
    )
    agent_cls, output_key, error_key = AGENTS[job["agent"]]
    agent = agent_cls(model=llm, bypass_explain_code=True)
    data = pd.read_parquet("/job/input.parquet")
    state = {
        "messages": [HumanMessage(content=job["request"])],
        "user_instructions": job["request"],
        "data_raw": data.to_dict(),
        "max_retries": job["max_retries"],
        "retry_count": 0,
    }

    start = time.perf_counter()
    crash = None
    try:
        response = agent._compiled_graph.invoke(state)
    except Exception as exc:
        response = {}
        crash = f"{type(exc).__name__}: {exc}"[:1000]
    seconds = time.perf_counter() - start

    output = response.get(output_key)
    kind = None
    if output:
        if job["agent"] == "data_visualization":
            kind = "figure"
            (out / "figure.json").write_text(json.dumps(output, cls=plotly.utils.PlotlyJSONEncoder))
        else:
            kind = "dataframe"
            pd.DataFrame(output).to_parquet(out / "output.parquet", index=False)
    messages = response.get("messages") or []
    final = messages[-1] if messages else None
    result = {
        "error": crash or response.get(error_key),
        "output_kind": kind,
        "final_message": str(getattr(final, "content", final))[:1000] if final else None,
        "code": response.get(
            output_key.replace("cleaned", "cleaner_function")
            .replace("wrangled", "wrangler_function")
            .replace("plotly_graph", "data_visualization_function")
        ),
        "calls": usage.calls,
        "prompt_tokens": usage.prompt,
        "completion_tokens": usage.completion,
        "reasoning_tokens": usage.reasoning,
        "seconds": round(seconds, 2),
        "traceback": traceback.format_exc()[-2000:] if crash else None,
    }
    (out / "result.json").write_text(json.dumps(result, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
