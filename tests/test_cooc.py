"""TWEC-from-co-occurrence tests with a planted semantic shift.

Mirrors test_twec.py but the input is weighted (center, ctx) count matrices,
as produced by the streaming ngram pipeline.
"""

import json

import numpy as np
import pytest

from neural_philology.config import TrainingConfig
from neural_philology.embeddings import TemporalEmbeddings
from neural_philology.twec.cooc import CoocMatrix, iter_cooc_batches, train_twec_cooc
from neural_philology.twec.sgns import keep_probs

FRUIT = ["apple", "pear", "plum", "grape", "fig"]
TECH = ["wire", "signal", "circuit", "battery", "antenna"]
VOCAB = FRUIT + TECH + ["shifty"]
IDX = {w: i for i, w in enumerate(VOCAB)}


def cluster_cooc(members: list[str], weight: float) -> list[tuple[int, int, float]]:
    return [
        (IDX[a], IDX[b], weight)
        for a in members
        for b in members
        if a != b
    ]


def build_data_dir(tmp_path):
    """Two decades: shifty co-occurs with FRUIT in 1900, TECH in 2000."""
    data_dir = tmp_path / "ngrams"
    (data_dir / "merged").mkdir(parents=True)
    (data_dir / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    counts = {
        str(decade): {w: 1000.0 for w in VOCAB} for decade in (1900, 2000)
    }
    (data_dir / "counts.json").write_text(json.dumps(counts))
    for decade, shifty_cluster in ((1900, FRUIT), (2000, TECH)):
        triples = cluster_cooc(FRUIT, 200) + cluster_cooc(TECH, 200)
        triples += [
            (IDX["shifty"], IDX[w], 150) for w in shifty_cluster
        ] + [
            (IDX[w], IDX["shifty"], 150) for w in shifty_cluster
        ]
        center, ctx, weight = map(np.array, zip(*triples))
        np.savez(
            data_dir / "merged" / f"cooc_{decade}.npz",
            center=center, ctx=ctx, weight=weight,
        )
    return data_dir


CONFIG = TrainingConfig(
    dim=24,
    subsample=0.0,
    compass_epochs=2,
    slice_epochs=2,
    batch_size=4096,
    seed=7,
    device="cpu",
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    data_dir = build_data_dir(tmp_path_factory.mktemp("cooc"))
    model, freq, vocab = train_twec_cooc(data_dir, CONFIG, pairs_per_epoch=100_000)
    return TemporalEmbeddings.from_twec(model, freq, slice_width=100)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def centroid(emb, words, year):
    vecs = np.stack([emb.vector(w, year) for w in words])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.mean(axis=0)


def test_sampler_respects_weights():
    cooc = CoocMatrix(
        center=np.array([0, 1]), ctx=np.array([1, 2]), weight=np.array([99.0, 1.0])
    )
    keep = keep_probs(np.ones(3), 0.0)
    rng = np.random.default_rng(0)
    centers = np.concatenate(
        [c for c, _ in iter_cooc_batches(cooc, keep, rng, 1000, 10_000)]
    )
    frac_heavy = (centers == 0).mean()
    assert 0.97 < frac_heavy < 1.0


def test_planted_shift_recovered_from_cooc(trained):
    emb = trained
    s1900 = emb.vector("shifty", 1900)
    s2000 = emb.vector("shifty", 2000)
    assert cos(s1900, centroid(emb, FRUIT, 1900)) > cos(s1900, centroid(emb, TECH, 1900))
    assert cos(s2000, centroid(emb, TECH, 2000)) > cos(s2000, centroid(emb, FRUIT, 2000))


def test_stable_words_aligned_from_cooc(trained):
    emb = trained
    for word, other in (("apple", "wire"), ("wire", "apple")):
        same = cos(emb.vector(word, 1900), emb.vector(word, 2000))
        cross = cos(emb.vector(word, 1900), emb.vector(other, 2000))
        assert same > cross
