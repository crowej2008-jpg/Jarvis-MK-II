"""Text embeddings for semantic recall over everything JARVIS has seen.

Two backends behind one interface:

    ollama  real semantic vectors from a local model such as nomic-embed-text.
            Best quality, needs Ollama running and the model pulled.
    hash    a signed hashing vectoriser built on the standard library. Purely
            lexical, but instant, offline, and good enough that "the save
            button" finds a screen labelled "Save file". This is the default
            so screen memory works before Ollama is even installed.

Both return L2-normalised float32 rows, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Protocol, Sequence

import numpy as np

log = logging.getLogger("jarvis.embeddings")

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have he her his i in is it its
    me my of on or our she that the their them they this to was were will with
    you your do does did done can could would should there here what when where
    which who whom how why not no yes ok okay just now new all any some more
    most other into over under than then them very also about after before""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    if drop_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return words or _WORD_RE.findall(text.lower())


class Embedder(Protocol):
    """The shape every backend provides."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> "np.ndarray": ...

    def encode_one(self, text: str) -> "np.ndarray": ...


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


class HashEmbedder:
    """Signed hashing bag-of-words plus character trigrams.

    Character trigrams are what let "settings" match "Setting" or
    "preferences" without a stemmer, and they keep the index useful on UI text
    where exact word overlap is rare.
    """

    name = "hash"

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        features = [f"w:{w}" for w in words]
        joined = " ".join(words)
        for word in words:
            padded = f"^{word}$"
            features.extend(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
        features.extend(f"b:{joined[i:i + 4]}" for i in range(0, max(0, len(joined) - 3), 2))
        return features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text or ""):
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, index] += sign
        return _unit(out)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 30.0):
        import requests

        from .config import connect_host

        self._requests = requests
        # connect_host, and this one is the most expensive probe in the app:
        # available() runs at startup, and resolving "localhost" to ::1 and
        # being refused costs about 2s before it falls back to 127.0.0.1. See
        # config.connect_host for the measurement.
        self.host = connect_host(host).rstrip("/")
        self.model = model
        self.timeout = timeout
        self._lock = threading.Lock()
        self.dim = 0
        # A plain requests.get opens a fresh connection every call. With the
        # host resolved to a literal that is ~15ms rather than ~2s, but reusing
        # one session means a long session does not pay it per call either.
        self._session = requests.Session()

    def available(self) -> bool:
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=2.0)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
        except Exception:  # noqa: BLE001
            return False
        base = self.model.split(":")[0]
        return any(n == self.model or n.split(":")[0] == base for n in names)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype=np.float32)
        vectors: list[list[float]] = []
        for text in texts:
            resp = self._session.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text or " "},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = matrix.shape[1]
        return _unit(matrix)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class NullEmbedder:
    name = "none"

    def __init__(self, dim: int = 1):
        self.dim = 1

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 1), dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return np.zeros((1,), dtype=np.float32)


def make_embedder(cfg):
    """Pick an embedder for the current config, falling back gracefully."""
    backend = (cfg.embed_backend or "auto").lower()

    if backend == "none":
        return NullEmbedder()

    if backend in ("auto", "ollama"):
        try:
            candidate = OllamaEmbedder(cfg.ollama_host, cfg.embed_model)
            if candidate.available():
                log.info("embeddings: ollama (%s)", cfg.embed_model)
                return candidate
            if backend == "ollama":
                log.warning(
                    "embed_backend is 'ollama' but %s is not available; using the hashing index",
                    cfg.embed_model,
                )
        except Exception as exc:  # noqa: BLE001
            if backend == "ollama":
                log.warning("ollama embeddings unavailable (%s); using hashing", exc)

    log.info("embeddings: built-in hashing index (%d dims)", cfg.embed_dim)
    return HashEmbedder(cfg.embed_dim)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom < 1e-9 else float(np.dot(a, b) / denom)
