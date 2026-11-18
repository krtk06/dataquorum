"""Experimental MiniLM first-step classifier with an input-kind eligibility filter."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from hashlib import file_digest
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

import numpy as np

from dataquorum.agents import AGENTS
from dataquorum.sandbox.executor import InputKind

Encoder = Literal["minilm", "tfidf"]
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_FILES = {
    "model.onnx": (
        "onnx/model.onnx",
        "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    ),
    "tokenizer.json": (
        "tokenizer.json",
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    ),
}


def _verified(path: Path, digest: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest() == digest


def download_encoder(directory: Path) -> None:
    """Fetch only the two pinned files. Verify before exposing them in the cache."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, (remote, digest) in _FILES.items():
        dest = directory / name
        if _verified(dest, digest):
            continue
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            staged = Path(temp) / name
            url = f"https://huggingface.co/{_MODEL}/resolve/{REVISION}/{remote}?download=true"
            with urlopen(url, timeout=60) as response, staged.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if not _verified(staged, digest):
                raise ValueError(f"Downloaded {name} failed SHA-256 verification")
            staged.replace(dest)


class MiniLM:
    """Use the Sentence Transformers model's masked mean pooling on ONNX Runtime CPU."""

    def __init__(self, directory: Path) -> None:
        from onnxruntime import InferenceSession, SessionOptions
        from tokenizers import Tokenizer

        for name, (_, digest) in _FILES.items():
            if not _verified(directory / name, digest):
                raise ValueError(
                    f"Missing or changed MiniLM file {name}. Fetch the pinned encoder."
                )
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options = SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = InferenceSession(
            str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def transform(self, texts: Sequence[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        batches = []
        for start in range(0, len(texts), 16):
            tokens = self.tokenizer.encode_batch(list(texts[start : start + 16]))
            mask = np.asarray([t.attention_mask for t in tokens], dtype=np.int64)
            inputs = {
                "input_ids": np.asarray([t.ids for t in tokens], dtype=np.int64),
                "attention_mask": mask,
                "token_type_ids": np.asarray([t.type_ids for t in tokens], dtype=np.int64),
            }
            hidden = self.session.run(["last_hidden_state"], inputs)[0]
            weights = mask[:, :, None].astype(np.float32)
            pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
            batches.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12))
        return np.asarray(np.concatenate(batches), dtype=np.float32)


class LearnedRouter:
    """Choose an eligible first-step label. Intended for comparison, with no workflow control."""

    def __init__(self, model: Any, encoder: Any) -> None:
        self._model = model
        self._encoder = encoder

    @classmethod
    def fit(
        cls,
        requests: Sequence[str],
        labels: Sequence[str],
        *,
        encoder: Encoder,
        directory: Path = Path("runs/router-model"),
    ) -> LearnedRouter:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        if encoder == "minilm":
            transformer = MiniLM(directory)
            features = transformer.transform(requests)
        else:
            transformer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            features = transformer.fit_transform(requests)
        classifier = LogisticRegression(max_iter=1000, C=4, random_state=7)
        return cls(classifier.fit(features, labels), transformer)

    def predict(self, request: str, kind: InputKind) -> str:
        scores = self._model.predict_proba(self._encoder.transform([request]))[0]
        labels = [str(label) for label in self._model.classes_]
        eligible = [
            i
            for i, label in enumerate(labels)
            if label == "refuse" or (label in AGENTS and AGENTS[label].input_kind == kind)
        ]
        if not eligible:
            raise ValueError(f"No learned routes for {kind} input")
        return labels[max(eligible, key=lambda i: float(scores[i]))]
