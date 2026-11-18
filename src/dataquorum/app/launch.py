"""Launch the local Streamlit Studio."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataquorum.app.style import THEME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    try:
        from streamlit.web.cli import main as streamlit_main
    except ImportError as exc:
        parser.exit(2, f"Install the studio extra with uv sync --locked --extra studio. {exc}\n")
    streamlit_main(
        args=[
            "run",
            str(Path(__file__).with_name("studio.py")),
            "--server.address=127.0.0.1",
            f"--server.port={args.port}",
            "--server.headless=true",
            "--server.fileWatcherType=none",
            "--server.maxUploadSize=25",
            "--client.toolbarMode=minimal",
            "--browser.gatherUsageStats=false",
            *(f"--theme.{name}={value}" for name, value in THEME.items()),
        ],
        prog_name="dataquorum-studio",
    )
