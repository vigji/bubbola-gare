from __future__ import annotations

import hashlib
import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import trange

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

from .config import (
    AVG_CHARS_PER_TOKEN,
    EMBED_BATCH_SIZE,
    EMBED_CACHE_PATH,
    EMBED_MAX_TOKENS,
    EMBED_WARN_MARGIN,
    EMBED_OUTPUT_DIM,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)
_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing. Please set it in your .env.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def reset_embed_cache() -> None:
    stem = EMBED_CACHE_PATH.name
    for f in EMBED_CACHE_PATH.parent.glob(f"{stem}*"):
        try:
            f.unlink()
        except OSError:
            pass


def _load_embed_cache() -> Dict[str, np.ndarray]:
    if not EMBED_CACHE_PATH.exists():
        return {}
    try:
        import pickle

        with open(EMBED_CACHE_PATH, "rb") as fh:
            cache = pickle.load(fh)
            return cache if isinstance(cache, dict) else {}
    except Exception as exc:
        warnings.warn(f"[cache] load failed ({exc}); proceeding without cache.")
        return {}


def _save_embed_cache(cache: Dict[str, np.ndarray]) -> None:
    try:
        import pickle

        EMBED_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = EMBED_CACHE_PATH.with_suffix(".tmp")
        with open(tmp_path, "wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(EMBED_CACHE_PATH)
    except Exception as exc:  # pragma: no cover - IO failure path
        warnings.warn(f"[cache] save failed; skipping cache store. ({exc})")


def _build_tokenizer():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover
        warnings.warn(
            f"[embed] failed to load tokenizer; falling back to char heuristic ({exc})"
        )
        return None


def _clip_to_limit(text: str, tokenizer) -> Tuple[str, int, bool]:
    if tokenizer is not None:
        toks = tokenizer.encode(text)
        if len(toks) > EMBED_MAX_TOKENS:
            clipped_tokens = toks[:EMBED_MAX_TOKENS]
            clipped_text = tokenizer.decode(clipped_tokens)
            return clipped_text, EMBED_MAX_TOKENS, True
        return text, len(toks), False

    safe_char_limit = EMBED_MAX_TOKENS * AVG_CHARS_PER_TOKEN
    if len(text) > safe_char_limit:
        clipped = text[:safe_char_limit]
        return clipped, EMBED_MAX_TOKENS, True
    est_tokens = math.ceil(len(text) / AVG_CHARS_PER_TOKEN)
    return text, est_tokens, False


def embed_texts(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    client: Optional[OpenAI] = None,
    log_level: int = logging.INFO,
    log_prefix: str = "[embed]",
) -> np.ndarray:
    """
    Embed a list of texts using the configured embedding model.
    Returns an array of shape (N, D) in float32.
    """
    logger.log(
        log_level,
        "%s starting batch with %d texts (batch_size=%d)",
        log_prefix,
        len(texts),
        batch_size,
    )
    client = client or get_openai_client()
    tokenizer = _build_tokenizer()

    embeddings: List[np.ndarray] = [None] * len(texts)  # type: ignore
    clipped_texts: List[str] = []
    warning_msgs: List[str] = []

    for t in texts:
        clipped, tokens, did_clip = _clip_to_limit(t, tokenizer)
        if did_clip:
            removed = t[len(clipped) :]
            removed_preview = removed[:160] + ("..." if len(removed) > 160 else "")
            warning_msgs.append(
                f"[embed] Input clipped to {EMBED_MAX_TOKENS} tokens "
                f"(removed {len(removed)} chars): {removed_preview!r}"
            )
        elif tokens > EMBED_WARN_MARGIN * EMBED_MAX_TOKENS:
            warning_msgs.append(
                f"[embed] Input near token limit (~{tokens}/{EMBED_MAX_TOKENS} tokens); clipping skipped."
            )
        clipped_texts.append(clipped)

    for w in warning_msgs:
        print(w)

    cache = _load_embed_cache()
    cache_dirty = False

    pending_texts: List[str] = []
    pending_indices: List[int] = []
    keys: List[str] = []

    for idx, t in enumerate(clipped_texts):
        key = hashlib.sha256(t.encode("utf-8")).hexdigest()
        keys.append(key)
        if key in cache:
            embeddings[idx] = np.array(cache[key], dtype=np.float32)
        else:
            pending_texts.append(t)
            pending_indices.append(idx)

    for i in trange(0, len(pending_texts), batch_size):
        batch = pending_texts[i : i + batch_size]
        if not batch:
            continue
        logger.log(
            log_level,
            "%s requesting %d embeddings (offset=%d)",
            log_prefix,
            len(batch),
            i,
        )
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for local_j, d in enumerate(resp.data):
            global_idx = pending_indices[i + local_j]
            emb = np.array(d.embedding, dtype=np.float32)
            embeddings[global_idx] = emb
            cache[keys[global_idx]] = emb
            cache_dirty = True

    if cache_dirty:
        _save_embed_cache(cache)

    logger.log(log_level, "%s finished; cache size now %d entries", log_prefix, len(cache))
    return np.vstack(embeddings)  # type: ignore


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _reduce_dimension(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    if target_dim <= 0:
        return vectors
    if vectors.shape[1] <= target_dim:
        return vectors
    try:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        reduced = svd.fit_transform(vectors)
        return reduced.astype(np.float32)
    except Exception as exc:  # pragma: no cover - optional reduction path
        warnings.warn(f"[embed] dim reduction failed ({exc}); keeping original dims")
        return vectors


def compute_embeddings(df: pd.DataFrame, text_col: str = "text_normalized") -> pd.DataFrame:
    series = df[text_col].fillna("")
    texts = [
        t if isinstance(t, str) else ("" if t is None else str(t))
        for t in series.tolist()
    ]
    emb = embed_texts(texts)
    emb = np.asarray(emb, dtype=np.float32)
    emb = _reduce_dimension(emb, EMBED_OUTPUT_DIM)
    emb = l2_normalize(emb)
    out = df.copy()
    out["embedding"] = emb.tolist()
    return out
