"""
BanglaLex — Embedder
======================
Thin wrapper around ``sentence_transformers.SentenceTransformer``.

Model: ``paraphrase-multilingual-MiniLM-L12-v2``
  • Embedding dim  : 384
  • Max seq length : 128 word-pieces
  • Languages      : 50+ incl. Bengali (bn) and English (en)
  • CUDA support   : automatic (uses RTX 3050 if torch+cuda is installed)

All embeddings are **L2-normalised**, so cosine similarity equals the
inner product (FAISS ``IndexFlatIP``).
"""

import logging
import numpy  as np
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """
    L2-normalised sentence embedder for English and Bangla legal text.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers model.  Defaults to
        ``paraphrase-multilingual-MiniLM-L12-v2``.

    Examples
    --------
    >>> embedder = Embedder()
    >>> vecs = embedder.embed(["চুক্তি লঙ্ঘনের শাস্তি কী?", "What is breach of contract?"])
    >>> vecs.shape
    (2, 384)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Loading embedding model: '{model_name}' …")
        # Import here so the module is importable without sentence-transformers
        # (e.g., for unit tests with mocks)
        from sentence_transformers import SentenceTransformer
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self.model.get_embedding_dimension()
        logger.info(f"Model ready | dim={self.dim} | device={self._device()}")

    # ── Core methods ───────────────────────────────────────────────────────────

    def embed(
        self,
        texts:         List[str],
        batch_size:    int  = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of strings.

        Parameters
        ----------
        texts         : list of raw strings (English or Bangla)
        batch_size    : sentences per GPU/CPU batch.
                        Reduce to 32 if you hit OOM on RTX 3050 4 GB.
        show_progress : show tqdm bar (useful during KB build)

        Returns
        -------
        np.ndarray  shape (N, 384), dtype float32, L2-normalised
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vecs = self.model.encode(
            texts,
            batch_size         = batch_size,
            show_progress_bar  = show_progress,
            convert_to_numpy   = True,
            normalize_embeddings = True,   # cosine sim = dot product on FAISS IP index
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns
        -------
        np.ndarray  shape (1, 384), dtype float32, L2-normalised
        """
        return self.embed([query], batch_size=1, show_progress=False)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _device(self) -> str:
        """Return the device the model is running on."""
        try:
            return str(next(self.model.parameters()).device)
        except Exception:
            return "unknown"

    def __repr__(self) -> str:
        return f"Embedder(model='{self.model_name}', dim={self.dim}, device={self._device()})"
