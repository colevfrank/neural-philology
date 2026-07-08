"""TWEC training from weighted co-occurrence counts (the ngram path).

Instead of sentences, the training data is per-decade sparse arrays of
(center, ctx, weight) built by the streaming ngram pipeline. An "epoch" is
``pairs_per_epoch`` pairs sampled with probability proportional to weight
(with word2vec-style subsampling applied multiplicatively to both members of
the pair). The compass trains on the decade-summed counts; slice models train
per decade with the compass context matrix frozen, exactly as in the
sentence-based path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import TrainingConfig
from ..corpus import FrequencyTable
from .sgns import keep_probs, train_sgns_pairs
from .trainer import TwecModel


class CoocMatrix:
    """Sparse (center, ctx) -> weight counts over a fixed vocabulary."""

    def __init__(self, center: np.ndarray, ctx: np.ndarray, weight: np.ndarray):
        if not (len(center) == len(ctx) == len(weight)):
            raise ValueError("center/ctx/weight length mismatch")
        if len(center) == 0:
            raise ValueError("empty co-occurrence matrix")
        self.center = center.astype(np.int64)
        self.ctx = ctx.astype(np.int64)
        self.weight = weight.astype(np.float64)

    @classmethod
    def load(cls, path: Path | str) -> CoocMatrix:
        data = np.load(path)
        return cls(data["center"], data["ctx"], data["weight"])

    def center_counts(self, vocab_size: int) -> np.ndarray:
        """Occurrence count per center word (pair mass / window size ~ freq)."""
        counts = np.zeros(vocab_size, dtype=np.float64)
        np.add.at(counts, self.center, self.weight)
        return counts

    @staticmethod
    def merge(matrices: list[CoocMatrix], vocab_size: int) -> CoocMatrix:
        codes = np.concatenate([m.center * vocab_size + m.ctx for m in matrices])
        weights = np.concatenate([m.weight for m in matrices])
        uniq, inverse = np.unique(codes, return_inverse=True)
        summed = np.zeros(len(uniq), dtype=np.float64)
        np.add.at(summed, inverse, weights)
        return CoocMatrix(uniq // vocab_size, uniq % vocab_size, summed)


def iter_cooc_batches(
    cooc: CoocMatrix,
    keep: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    n_pairs: int,
):
    """Sample ``n_pairs`` (center, ctx) pairs proportional to subsampled weight."""
    adjusted = cooc.weight * keep[cooc.center] * keep[cooc.ctx]
    total = adjusted.sum()
    if total <= 0:
        raise ValueError("all co-occurrence mass removed by subsampling")
    cumulative = np.cumsum(adjusted)
    emitted = 0
    while emitted < n_pairs:
        size = min(batch_size, n_pairs - emitted)
        idx = np.searchsorted(cumulative, rng.random(size) * total)
        yield cooc.center[idx], cooc.ctx[idx]
        emitted += size


def train_twec_cooc(
    data_dir: Path | str,
    config: TrainingConfig | None = None,
    pairs_per_epoch: int = 20_000_000,
) -> tuple[TwecModel, FrequencyTable, list[str]]:
    """Full TWEC run over a directory produced by the ngram pipeline.

    Expects ``vocab.txt``, ``counts.json`` and ``merged/cooc_<decade>.npz``.
    """
    config = config or TrainingConfig()
    data_dir = Path(data_dir)
    vocab = data_dir.joinpath("vocab.txt").read_text().split()
    freq_table = FrequencyTable.load(data_dir / "counts.json")
    files = sorted((data_dir / "merged").glob("cooc_*.npz"))
    if not files:
        raise ValueError(f"no merged co-occurrence files under {data_dir / 'merged'}")
    slices = {int(f.stem.split("_")[1]): CoocMatrix.load(f) for f in files}

    def train_one(
        cooc: CoocMatrix,
        epochs: int,
        desc: str,
        w_in_init: np.ndarray | None = None,
        w_out_init: np.ndarray | None = None,
        freeze_context: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Subsampling/negative distributions come from the pair mass itself,
        # which is proportional to token frequency in the underlying corpus.
        counts = cooc.center_counts(len(vocab))
        keep = keep_probs(counts, config.subsample)
        rng = np.random.default_rng(config.seed)

        def batch_factory():
            return iter_cooc_batches(
                cooc, keep, rng, config.batch_size, pairs_per_epoch
            )

        return train_sgns_pairs(
            batch_factory,
            len(vocab),
            counts,
            config,
            epochs,
            est_pairs_per_epoch=pairs_per_epoch,
            w_in_init=w_in_init,
            w_out_init=w_out_init,
            freeze_context=freeze_context,
            desc=desc,
        )

    compass_cooc = CoocMatrix.merge(list(slices.values()), len(vocab))
    compass_in, compass_out = train_one(
        compass_cooc, config.compass_epochs, "compass"
    )
    from ..corpus import Vocab

    model = TwecModel(
        vocab=Vocab(words=tuple(vocab), counts=tuple(compass_cooc.center_counts(len(vocab)))),
        compass_in=compass_in,
        compass_out=compass_out,
    )
    for decade, cooc in slices.items():
        model.slice_in[decade], _ = train_one(
            cooc,
            config.slice_epochs,
            f"slice {decade}",
            w_in_init=compass_in,
            w_out_init=compass_out,
            freeze_context=True,
        )
    return model, freq_table, vocab
