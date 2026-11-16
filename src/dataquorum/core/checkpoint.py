"""Checkpoint serialisation for the engine's LangGraph state."""

from __future__ import annotations

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Types stored in checkpoints. LangGraph will refuse unregistered types in a future version.
_CHECKPOINT_TYPES = [
    ("dataquorum.core.cost", "Usage"),
    ("dataquorum.core.statuses", "CheckResult"),
    ("dataquorum.core.statuses", "Severity"),
    ("dataquorum.core.statuses", "Status"),
    ("dataquorum.core.state", "AgentResult"),
    ("dataquorum.core.state", "DataFrameOutput"),
    ("dataquorum.core.state", "FigureOutput"),
    ("dataquorum.core.state", "ModelOutput"),
    ("dataquorum.core.state", "TextOutput"),
    ("pathlib", "PosixPath"),
    ("pathlib", "WindowsPath"),
]


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
