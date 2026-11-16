"""Record and replay HTTP calls made by the openai SDK.

The SDK uses `httpx2`, which vcrpy/pytest-recording cannot intercept, so we plug these
transports in through `LLMClient(http_client=httpx2.Client(transport=...))`.

Cassettes store only method, URL, JSON request body, status and JSON response body.
Request headers (including the API key) are never written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2


class RecordingTransport(httpx2.BaseTransport):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.inner = httpx2.HTTPTransport()
        self.records: list[dict[str, Any]] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        response = self.inner.handle_request(request)
        body = response.read()
        self.records.append(
            {
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "body": json.loads(request.content or b"null"),
                },
                "response": {"status": response.status_code, "body": json.loads(body)},
            }
        )
        return httpx2.Response(
            response.status_code, content=body, headers={"content-type": "application/json"}
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")


class ReplayTransport(httpx2.BaseTransport):
    def __init__(self, path: Path) -> None:
        self.records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        self.requests: list[dict[str, Any]] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        if not self.records:
            raise AssertionError(f"Unexpected extra request: {request.method} {request.url}")
        record = self.records.pop(0)
        expected = record["request"]
        assert request.method == expected["method"]
        assert str(request.url) == expected["url"]
        self.requests.append(json.loads(request.content or b"null"))
        return httpx2.Response(record["response"]["status"], json=record["response"]["body"])
