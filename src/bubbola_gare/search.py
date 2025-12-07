from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .config import DEFAULT_ALPHA, DEFAULT_TOP_K
from .embedding import embed_texts, l2_normalize
from .preprocessing import normalize_text, tokenize_for_bm25

logger = logging.getLogger(__name__)


class HybridIndex:
    def __init__(
        self,
        ids: np.ndarray,
        original_texts: np.ndarray,
        normalized_texts: np.ndarray,
        embeddings: np.ndarray,
        bm25: BM25Okapi,
        meta: Optional[pd.DataFrame] = None,
    ):
        self.ids = ids
        self.original_texts = original_texts
        self.normalized_texts = normalized_texts
        self.embeddings = embeddings
        self.bm25 = bm25
        self.meta = meta

    @classmethod
    def from_embeddings_df(
        cls,
        df: pd.DataFrame,
        text_col: str = "text_normalized",
        id_col: str = "record_id",
    ) -> "HybridIndex":
        logger.info("[index] building from embeddings df rows=%d", len(df))
        norm_texts = df[text_col].fillna("").tolist()
        ids = df[id_col].tolist()
        original_texts = df.get("text_raw", df[text_col]).fillna("").tolist()

        tokenized_corpus = [tokenize_for_bm25(t) for t in norm_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        logger.info("[index] BM25 built over %d documents", len(tokenized_corpus))

        emb = np.vstack(df["embedding"].apply(np.array).to_list()).astype(np.float32)
        emb = l2_normalize(emb)
        meta_cols = [c for c in df.columns if c not in {text_col, id_col, "embedding"}]
        meta = df[[id_col, *meta_cols]].set_index(id_col) if meta_cols else None

        return cls(
            ids=np.array(ids),
            original_texts=np.array(original_texts, dtype=object),
            normalized_texts=np.array(norm_texts, dtype=object),
            embeddings=emb,
            bm25=bm25,
            meta=meta,
        )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> List[Dict[str, Any]]:
        q_norm = normalize_text(query)
        q_tokens = tokenize_for_bm25(q_norm)

        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

        q_emb = embed_texts([q_norm], log_level=logging.DEBUG, log_prefix="[embed-query]")
        q_emb = l2_normalize(q_emb)

        k_vec = min(max(top_k * 5, top_k), len(self.ids))
        vec_scores_all = (self.embeddings @ q_emb[0]).astype(np.float32)
        if k_vec <= 0:
            vec_scores = np.array([], dtype=np.float32)
            vec_idx = np.array([], dtype=int)
        else:
            top_idx = np.argpartition(-vec_scores_all, kth=k_vec - 1)[:k_vec]
            ordered = top_idx[np.argsort(-vec_scores_all[top_idx])]
            vec_idx = ordered
            vec_scores = vec_scores_all[ordered]

        k_bm25 = min(top_k * 5, len(self.ids))
        bm25_top_idx = np.argsort(-bm25_scores)[:k_bm25]

        candidate_idx = np.unique(np.concatenate([vec_idx, bm25_top_idx]))

        cand_bm25 = bm25_scores[candidate_idx]
        cand_vec = np.zeros_like(cand_bm25, dtype=np.float32)

        idx_map = {int(i): s for i, s in zip(vec_idx, vec_scores)}
        for j, gi in enumerate(candidate_idx):
            cand_vec[j] = idx_map.get(int(gi), 0.0)

        def min_max_norm(arr: np.ndarray) -> np.ndarray:
            mn = float(arr.min())
            mx = float(arr.max())
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        bm25_n = min_max_norm(cand_bm25)
        vec_n = min_max_norm(cand_vec)

        combined = alpha * vec_n + (1.0 - alpha) * bm25_n
        order = np.argsort(-combined)[:top_k]

        results = []
        for rank_pos in order:
            corpus_idx = int(candidate_idx[rank_pos])
            rid = int(self.ids[corpus_idx])
            meta_row = self.meta.loc[rid].to_dict() if self.meta is not None else {}
            results.append(
                {
                    "record_id": rid,
                    "score_combined": float(combined[rank_pos]),
                    "score_vector": float(vec_n[rank_pos]),
                    "score_bm25": float(bm25_n[rank_pos]),
                    "text": self.original_texts[corpus_idx],
                    "text_normalized": self.normalized_texts[corpus_idx],
                    **meta_row,
                }
            )
        return results


def load_index_from_parquet(path: Any) -> HybridIndex:
    start = time.perf_counter()
    df = pd.read_parquet(path)
    logger.info("[index] loaded %d rows from %s in %.2fs", len(df), path, time.perf_counter() - start)
    return HybridIndex.from_embeddings_df(df)
