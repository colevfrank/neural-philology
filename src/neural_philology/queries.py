"""The two core queries.

- ``neighbors_of(word, year, k)``: nearest neighbors within one slice.
- ``equivalent(word, source_year, target_year, k)``: take the word's vector in
  the source slice and rank the *target* slice's vocabulary against it by
  cosine similarity. Because TWEC slices share one coordinate system, this
  cross-space comparison is meaningful in both time directions.

Every result carries the word's raw frequency in its slice and a
``low_frequency`` flag — unknown or below-threshold frequencies are flagged so
unreliable positions are never presented with false confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import QueryConfig
from .embeddings import TemporalEmbeddings

_DEFAULT = QueryConfig()


@dataclass(frozen=True)
class QueryResult:
    word: str
    slice_year: int
    score: float  # cosine similarity to the query vector
    frequency: float | None  # raw count in slice_year; None = unknown
    low_frequency: bool  # True if frequency is unknown or below threshold


def _flag(freq: float | None, config: QueryConfig) -> bool:
    return freq is None or freq < config.low_frequency_threshold


def word_info(
    embeddings: TemporalEmbeddings,
    word: str,
    year: int,
    config: QueryConfig = _DEFAULT,
) -> QueryResult:
    """Reliability info for a word-year itself (use for the query word)."""
    s = embeddings.resolve_slice(year)
    embeddings.vector(word, s)  # raises KeyError if absent
    freq = embeddings.frequency(word, s)
    return QueryResult(word, s, 1.0, freq, _flag(freq, config))


def _rank(
    embeddings: TemporalEmbeddings,
    query_vec: np.ndarray,
    target_slice: int,
    k: int,
    config: QueryConfig,
    exclude: str | None = None,
) -> list[QueryResult]:
    emb = embeddings.slices[target_slice]
    qnorm = np.linalg.norm(query_vec)
    if qnorm < 1e-12:
        raise ValueError("query vector has zero norm")
    sims = emb.normalized @ (query_vec / qnorm)
    if exclude is not None and exclude in emb.index:
        sims[emb.index[exclude]] = -np.inf
    k = min(k, len(emb.words))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    results = []
    for i in top:
        if not np.isfinite(sims[i]):
            continue
        word = emb.words[i]
        freq = embeddings.frequency(word, target_slice)
        results.append(
            QueryResult(word, target_slice, float(sims[i]), freq, _flag(freq, config))
        )
    return results


def neighbors_of(
    embeddings: TemporalEmbeddings,
    word: str,
    year: int,
    k: int = 10,
    config: QueryConfig = _DEFAULT,
) -> list[QueryResult]:
    """Nearest neighbors of ``word`` within its own time slice."""
    s = embeddings.resolve_slice(year)
    vec = embeddings.vector(word, s)
    return _rank(embeddings, vec, s, k, config, exclude=word)


def equivalent(
    embeddings: TemporalEmbeddings,
    word: str,
    source_year: int,
    target_year: int,
    k: int = 10,
    config: QueryConfig = _DEFAULT,
) -> list[QueryResult]:
    """Cross-time equivalence: nearest words in the *target* slice to the
    source-slice vector of ``word``.

    The query word itself is *not* excluded — if it ranks highly in the target
    slice, that is evidence its meaning was already the same back then.
    """
    source = embeddings.resolve_slice(source_year)
    target = embeddings.resolve_slice(target_year)
    vec = embeddings.vector(word, source)
    return _rank(embeddings, vec, target, k, config)
