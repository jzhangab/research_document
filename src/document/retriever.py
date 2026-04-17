from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from .chunker import Chunk

if TYPE_CHECKING:
    pass


class HybridRetriever:
    """TF-IDF + dense embedding retrieval fused with RRF."""

    def __init__(self, chunks: list[Chunk], top_k: int = 5):
        self.chunks = chunks
        self.top_k = top_k
        self._tfidf = None
        self._tfidf_matrix = None
        self._embedder = None
        self._embeddings = None

    def build_index(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sentence_transformers import SentenceTransformer

        texts = [c.text for c in self.chunks]

        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = self._tfidf.fit_transform(texts)

        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._embeddings = self._embedder.encode(texts, show_progress_bar=False)

    def retrieve(self, query: str) -> list[Chunk]:
        if self._tfidf is None:
            self.build_index()

        tfidf_ranks = self._tfidf_rank(query)
        dense_ranks = self._dense_rank(query)
        fused_indices = self._rrf(tfidf_ranks, dense_ranks)

        return [self.chunks[i] for i in fused_indices[: self.top_k]]

    def _tfidf_rank(self, query: str) -> list[tuple[int, float]]:
        import scipy.sparse

        q_vec = self._tfidf.transform([query])
        scores = (self._tfidf_matrix @ q_vec.T).toarray().flatten()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked

    def _dense_rank(self, query: str) -> list[tuple[int, float]]:
        q_emb = self._embedder.encode([query], show_progress_bar=False)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        q_norm = np.linalg.norm(q_emb) + 1e-10
        scores = (self._embeddings / norms) @ (q_emb / q_norm).T
        scores = scores.flatten()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked

    def _rrf(
        self,
        tfidf_ranks: list[tuple[int, float]],
        dense_ranks: list[tuple[int, float]],
        k: int = 60,
    ) -> list[int]:
        scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(tfidf_ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(dense_ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)
