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
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

_VEC_FILE = config.INDEX_DIR / "vectors.npy"
_META_FILE = config.INDEX_DIR / "meta.json"
_SRC_FILE = config.INDEX_DIR / "sources.json"
_TOKEN_RE = re.compile(r"[^\W\d_]{2,}|\d+", re.UNICODE)


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


def _tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall((text or "").lower()))


class VectorStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None  # normalised, float32 (N, dim)
        self._meta: list[dict] = []
        self._sources: dict[str, dict] = {}
        self._load()

    # --- persistence ------------------------------------------------------
    def _load(self) -> None:
        try:
            if _VEC_FILE.exists():
                self._vectors = np.load(_VEC_FILE)
            if _META_FILE.exists():
                self._meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
            if _SRC_FILE.exists():
                self._sources = json.loads(_SRC_FILE.read_text(encoding="utf-8"))
            n_vectors = 0 if self._vectors is None else int(self._vectors.shape[0])
            if n_vectors != len(self._meta):
                raise ValueError(
                    f"index inconsistent: {n_vectors} vectors vs {len(self._meta)} chunks"
                )
        except Exception as exc:  # noqa: BLE001
            # A truncated/corrupt index (machine shut down mid-download, partial
            # copy, …) must never brick startup: quarantine it and start empty
            # so the Drive bootstrap can fetch a fresh copy on this launch.
            print(f"WARNING: knowledge-base index unreadable ({exc}); starting empty.")
            self._vectors, self._meta, self._sources = None, [], {}
            for f in (_VEC_FILE, _META_FILE, _SRC_FILE):
                try:
                    if f.exists():
                        f.replace(f.with_suffix(f.suffix + ".corrupt"))
                except OSError:
                    pass

    def reload(self) -> None:
        """Re-read the index files from disk (after an unpacked download)."""
        with self._lock:
            self._vectors = None
            self._meta = []
            self._sources = {}
            self._load()

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
        subject: str | None = None,
        class_level: str | None = None,
        board: str | None = None,
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
                    {
                        "source_id": source_id,
                        "source": filename,
                        "chunk_index": i,
                        "text": text,
                        "subject": subject or None,
                        "class_level": class_level or None,
                        "board": board or None,
                    }
                )
            self._sources[source_id] = {
                "filename": filename,
                "n_chunks": len(chunks),
                "added_at": added_at,
                "subject": subject or None,
                "class_level": class_level or None,
                "board": board or None,
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
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        subject: str | None = None,
        class_level: str | None = None,
        board: str | None = None,
    ) -> list[Hit]:
        with self._lock:
            if self._vectors is None or not self._meta:
                return []
            # Board-neutral chunks (board=None, e.g. material that serves both
            # boards) stay visible to every board-filtered query.
            candidates = [
                i
                for i, m in enumerate(self._meta)
                if (not subject or m.get("subject") == subject)
                and (not class_level or m.get("class_level") == class_level)
                and (not board or m.get("board") in (board, None))
            ]
            if not candidates:
                return []
            q = np.asarray(query_embedding, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
            cand = np.asarray(candidates, dtype=np.int64)
            scores = self._vectors[cand] @ q
            k = min(top_k, len(scores))
            local_idx = np.argpartition(-scores, k - 1)[:k]
            local_idx = local_idx[np.argsort(-scores[local_idx])]
            idx = cand[local_idx]
            return [
                Hit(
                    text=self._meta[i]["text"],
                    source=self._meta[i]["source"],
                    score=float(self._vectors[i] @ q),
                    metadata=self._meta[i],
                )
                for i in idx
            ]

    def query_text(
        self,
        query: str,
        top_k: int,
        *,
        subject: str | None = None,
        class_level: str | None = None,
        board: str | None = None,
    ) -> list[Hit]:
        q = _tokens(query)
        if not q:
            return []
        q_terms = set(q)
        with self._lock:
            scored: list[tuple[float, int]] = []
            for i, m in enumerate(self._meta):
                if subject and m.get("subject") != subject:
                    continue
                if class_level and m.get("class_level") != class_level:
                    continue
                if board and m.get("board") not in (board, None):
                    continue
                doc = _tokens(m.get("text", ""))
                if not doc:
                    continue
                overlap = sum(min(q[t], doc[t]) for t in q_terms & set(doc))
                if overlap <= 0:
                    continue
                coverage = overlap / max(sum(q.values()), 1)
                density = overlap / max(sum(doc.values()), 1)
                scored.append((coverage + density, i))
            scored.sort(reverse=True)
            return [
                Hit(
                    text=self._meta[i]["text"],
                    source=self._meta[i]["source"],
                    score=float(score),
                    metadata=self._meta[i],
                )
                for score, i in scored[:top_k]
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
