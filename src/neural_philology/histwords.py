"""Loader for HistWords precomputed diachronic embeddings (Hamilton et al. 2016).

Layout of an extracted ``eng-all_sgns.zip``: per decade ``<year>-vocab.pkl``
(a list of 100k words, Python 2 pickle) and ``<year>-w.npy`` (a (100000, 300)
float64 matrix, SGNS + orthogonal Procrustes alignment). The separately
published ``freqs.pkl`` maps word -> {decade: relative frequency}.

Notes:
- Words that do not occur in a decade have all-zero rows there; we drop them
  from that slice rather than serve fabricated vectors.
- Frequencies are *relative* (fractions of the decade's tokens), not raw
  counts, so pass a low-frequency threshold on that scale (e.g. 1e-6) when
  querying — the default count-scale threshold flags everything.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .corpus import FrequencyTable
from .embeddings import SliceEmbeddings, TemporalEmbeddings

VOCAB_RE = re.compile(r"^(\d{4})-vocab\.pkl$")


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh, encoding="latin1")


def load_histwords(
    sgns_dir: Path | str,
    freqs_path: Path | str | None = None,
) -> TemporalEmbeddings:
    """Build TemporalEmbeddings from an extracted HistWords sgns directory."""
    sgns_dir = Path(sgns_dir)
    decades = sorted(
        int(m.group(1))
        for f in sgns_dir.iterdir()
        if (m := VOCAB_RE.match(f.name))
    )
    if not decades:
        raise ValueError(f"no <year>-vocab.pkl files in {sgns_dir}")

    freqs: dict | None = None
    if freqs_path is not None:
        freqs = _load_pickle(Path(freqs_path))

    slices: dict[int, SliceEmbeddings] = {}
    counts: dict[int, dict[str, float]] = {}
    for decade in tqdm(decades, desc="histwords decades", unit="slice"):
        vocab: list[str] = _load_pickle(sgns_dir / f"{decade}-vocab.pkl")
        matrix = np.load(sgns_dir / f"{decade}-w.npy")
        nonzero = np.linalg.norm(matrix, axis=1) > 0
        words = [w for w, keep in zip(vocab, nonzero) if keep]
        slices[decade] = SliceEmbeddings(
            words=words, vectors=matrix[nonzero].astype(np.float32)
        )
        if freqs is not None:
            counts[decade] = {
                w: float(freqs[w][decade])
                for w in words
                if w in freqs and decade in freqs[w] and freqs[w][decade] > 0
            }

    freq_table = FrequencyTable(counts) if freqs is not None else None
    return TemporalEmbeddings(slices, freq=freq_table, slice_width=10)


def convert_histwords(
    sgns_dir: Path | str,
    out_dir: Path | str,
    freqs_path: Path | str | None = None,
) -> TemporalEmbeddings:
    """Convert HistWords files to the project's serving format on disk."""
    embeddings = load_histwords(sgns_dir, freqs_path)
    embeddings.save(out_dir)
    return embeddings
