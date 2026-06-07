"""
BanglaLex — FAISS Index Manager
=================================
Wraps ``faiss.IndexFlatIP`` (inner-product search on L2-normalised vectors =
cosine-similarity search) and stores chunk metadata alongside the binary index.

Saved artefacts (in *index_dir*)
---------------------------------
  faiss_index.bin   binary FAISS index (portable across OS)
  metadata.json     UTF-8 JSON list of chunk dicts (parallel to index rows)

Usage
-----
  # Build once
  kb = KnowledgeBaseIndex(dim=384)
  kb.build(embeddings, chunks)
  kb.save("data/knowledge_base")

  # Load in the retriever / agents
  kb = KnowledgeBaseIndex.load("data/knowledge_base")
  scores, indices = kb.search(query_vec, k=5)
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing  import Dict, List, Tuple

import faiss

logger = logging.getLogger(__name__)

_INDEX_FILE    = "faiss_index.bin"
_METADATA_FILE = "metadata.json"


class KnowledgeBaseIndex:
    """
    FAISS-backed index with parallel chunk metadata.

    Parameters
    ----------
    dim : int
        Embedding dimension (384 for MiniLM-L12).
    """

    def __init__(self, dim: int = 384):
        self.dim    = dim
        self.index  = faiss.IndexFlatIP(dim)   # exact cosine-sim (IP on L2-normed vecs)
        self.chunks: List[Dict] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Populate the index with *embeddings* and attach *chunks* as metadata.

        Parameters
        ----------
        embeddings : np.ndarray
            shape (N, dim), dtype float32, L2-normalised.
        chunks : List[Dict]
            Parallel list — chunks[i] is the metadata for embeddings[i].
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Shape mismatch: {embeddings.shape[0]} embeddings "
                f"vs {len(chunks)} chunk dicts."
            )
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        logger.info(f"Adding {len(chunks):,} vectors (dim={self.dim}) to FAISS …")
        self.index.add(embeddings)
        self.chunks = list(chunks)
        logger.info(f"Index built — total vectors: {self.index.ntotal:,}")

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """
        Write the FAISS binary and metadata JSON to *index_dir*.
        Creates the directory if it does not exist.
        """
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / _INDEX_FILE))
        with open(index_dir / _METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        idx_mb   = (index_dir / _INDEX_FILE).stat().st_size / 1e6
        meta_mb  = (index_dir / _METADATA_FILE).stat().st_size / 1e6
        logger.info(
            f"Saved → {index_dir}  "
            f"[index {idx_mb:.1f} MB | metadata {meta_mb:.1f} MB]"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "KnowledgeBaseIndex":
        """
        Load a previously saved index from *index_dir*.

        Returns
        -------
        KnowledgeBaseIndex  ready for search
        """
        index_dir   = Path(index_dir)
        index_path  = index_dir / _INDEX_FILE
        meta_path   = index_dir / _METADATA_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run  python scripts/phase2_build_kb.py  first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        fi = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            chunks = json.load(f)

        obj        = cls(dim=fi.d)
        obj.index  = fi
        obj.chunks = chunks
        logger.info(
            f"Index loaded ← {index_dir}  "
            f"[vectors={fi.ntotal:,} | dim={fi.d}]"
        )
        return obj

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the top-*k* most similar chunks for *query_embedding*.

        Parameters
        ----------
        query_embedding : np.ndarray
            shape (1, dim) **or** (dim,), float32, L2-normalised.
        k : int
            Number of neighbours (capped at index.ntotal).

        Returns
        -------
        scores  : np.ndarray  shape (1, k) — cosine similarities ∈ [−1, 1]
        indices : np.ndarray  shape (1, k) — row indices into self.chunks
                              (FAISS returns −1 for unused slots)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        return scores, indices

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return a summary dict (useful for logging / debug)."""
        from collections import Counter
        act_counts = Counter(c.get("act_name", "unknown") for c in self.chunks)
        return {
            "total_chunks":  self.index.ntotal,
            "dim":           self.dim,
            "unique_acts":   len(act_counts),
            "top_5_acts":    act_counts.most_common(5),
        }

    def __repr__(self) -> str:
        return (
            f"KnowledgeBaseIndex("
            f"vectors={self.index.ntotal:,}, dim={self.dim})"
        )
