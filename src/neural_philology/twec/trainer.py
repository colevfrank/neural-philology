"""TWEC training orchestration (Di Carlo et al. 2019).

1. Train an atemporal "compass" SGNS model on the concatenation of all slices.
2. For each slice, train a fresh SGNS model on that slice only, initialised
   from the compass target matrix, with the compass *context* matrix frozen.

Because every slice model optimises its target vectors against the same fixed
context space, all slices are mutually aligned by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import TrainingConfig
from ..corpus import FrequencyTable, TimeSlicedCorpus, Vocab
from .sgns import train_sgns


@dataclass
class TwecModel:
    vocab: Vocab
    compass_in: np.ndarray  # (V, d) compass target matrix
    compass_out: np.ndarray  # (V, d) frozen context matrix shared by all slices
    slice_in: dict[int, np.ndarray] = field(default_factory=dict)  # slice -> (V, d)


def stream_counts_for(
    vocab: Vocab, freq_table: FrequencyTable, slice_year: int | None = None
) -> np.ndarray:
    """Counts of the training stream aligned to the compass vocab."""
    if slice_year is None:
        merged = freq_table.merged()
        return np.array([merged.get(w, 0.0) for w in vocab.words])
    return np.array([freq_table.count(w, slice_year) for w in vocab.words])


def train_compass(
    corpus: TimeSlicedCorpus,
    vocab: Vocab,
    freq_table: FrequencyTable,
    config: TrainingConfig,
) -> TwecModel:
    compass_in, compass_out = train_sgns(
        corpus.all_sentences,
        vocab,
        stream_counts_for(vocab, freq_table),
        config,
        epochs=config.compass_epochs,
        desc="compass",
    )
    return TwecModel(vocab=vocab, compass_in=compass_in, compass_out=compass_out)


def train_slice(
    corpus: TimeSlicedCorpus,
    slice_year: int,
    model: TwecModel,
    freq_table: FrequencyTable,
    config: TrainingConfig,
) -> np.ndarray:
    slice_counts = stream_counts_for(model.vocab, freq_table, slice_year)
    if slice_counts.sum() == 0:
        raise ValueError(f"slice {slice_year} has no in-vocabulary tokens")
    w_in, _ = train_sgns(
        lambda: corpus.sentences(slice_year),
        model.vocab,
        slice_counts,
        config,
        epochs=config.slice_epochs,
        w_in_init=model.compass_in,
        w_out_init=model.compass_out,
        freeze_context=True,
        desc=f"slice {slice_year}",
    )
    return w_in


def train_twec(
    corpus: TimeSlicedCorpus,
    config: TrainingConfig | None = None,
    freq_table: FrequencyTable | None = None,
) -> tuple[TwecModel, FrequencyTable]:
    """Full TWEC run: frequency counts, vocab, compass, then every slice."""
    config = config or TrainingConfig()
    freq_table = freq_table or corpus.count_frequencies()
    vocab = Vocab.build(freq_table.merged(), config.min_count)
    model = train_compass(corpus, vocab, freq_table, config)
    for slice_year in corpus.slices:
        model.slice_in[slice_year] = train_slice(
            corpus, slice_year, model, freq_table, config
        )
    return model, freq_table
