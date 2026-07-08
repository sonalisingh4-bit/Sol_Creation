"""Lightweight persistent vector store (NumPy brute-force cosine search).

No native dependencies — chosen over Chroma/FAISS so it installs cleanly on
Windows without a C++ toolchain. Plenty fast for textbook-scale knowledge
bases (hundreds of thousands of chunks query in a few ms). The public
interface (add / query / sources / delete_source / clear) is deliberately
small so a heavier ANN backend can be dropped in later without touching
callers.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

_VEC_FILE = config.INDEX_DIR / "vectors.npy"
_META_FILE = config.INDEX_DIR / "meta.json"
_SRC_FILE = config.INDEX_DIR / "sources.json"


@dataclass
class Hit:
    text: str
    source: str
    score: float
    metadata: dict


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None  # normalised, float32 (N, dim)
        self._meta: list[dict] = []
        self._sources: dict[str, dict] = {}
        self._load()

    # --- persistence ------------------------------------------------------
    def _load(self) -> None:
        if _VEC_FILE.exists():
            self._vectors = np.load(_VEC_FILE)
        if _META_FILE.exists():
            self._meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
        if _SRC_FILE.exists():
            self._sources = json.loads(_SRC_FILE.read_text(encoding="utf-8"))

    def _save(self) -> None:
        if self._vectors is not None:
            np.save(_VEC_FILE, self._vectors)
        elif _VEC_FILE.exists():
            _VEC_FILE.unlink()
        _META_FILE.write_text(json.dumps(self._meta, ensure_ascii=False), encoding="utf-8")
        _SRC_FILE.write_text(json.dumps(self._sources, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- mutations --------------------------------------------------------
    def add(
        self,
        *,
        source_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
        added_at: str,
    ) -> None:
        if not chunks:
            return
        new = _normalise(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            if self._vectors is None:
                self._vectors = new
            else:
                self._vectors = np.vstack([self._vectors, new])
            for i, text in enumerate(chunks):
                self._meta.append(
                    {"source_id": source_id, "source": filename, "chunk_index": i, "text": text}
                )
            self._sources[source_id] = {
                "filename": filename,
                "n_chunks": len(chunks),
                "added_at": added_at,
            }
            self._save()

    def delete_source(self, source_id: str) -> bool:
        with self._lock:
            if source_id not in self._sources:
                return False
            keep = [i for i, m in enumerate(self._meta) if m["source_id"] != source_id]
            self._meta = [self._meta[i] for i in keep]
            self._vectors = self._vectors[keep] if (self._vectors is not None and keep) else None
            if not keep:
                self._vectors = None
            self._sources.pop(source_id, None)
            self._save()
            return True

    def clear(self) -> None:
        with self._lock:
            self._vectors = None
            self._meta = []
            self._sources = {}
            self._save()

    # --- queries ----------------------------------------------------------
    def query(self, query_embedding: list[float], top_k: int) -> list[Hit]:
        with self._lock:
            if self._vectors is None or not self._meta:
                return []
            q = np.asarray(query_embedding, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            scores = self._vectors @ q
            k = min(top_k, len(scores))
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            return [
                Hit(
                    text=self._meta[i]["text"],
                    source=self._meta[i]["source"],
                    score=float(scores[i]),
                    metadata=self._meta[i],
                )
                for i in idx
            ]

    def sources(self) -> list[dict]:
        with self._lock:
            return [{"source_id": sid, **info} for sid, info in self._sources.items()]

    def count(self) -> int:
        with self._lock:
            return len(self._meta)


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
