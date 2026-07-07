"""TWEC correctness tests on a synthetic corpus with a planted semantic shift.

Two word clusters (fruit-like and tech-like) co-occur only within themselves.
The word ``shifty`` co-occurs with the fruit cluster in 1900 and with the tech
cluster in 2000. A correct TWEC implementation must (a) keep the compass
context matrix frozen during slice training, (b) keep stable words aligned
across slices, and (c) move ``shifty`` between clusters.
"""

from pathlib import Path

import numpy as np
import pytest

from neural_philology.config import TrainingConfig
from neural_philology.corpus import TimeSlicedCorpus, Vocab
from neural_philology.embeddings import TemporalEmbeddings
from neural_philology.queries import equivalent, neighbors_of
from neural_philology.twec import train_sgns, train_twec
from neural_philology.twec.trainer import stream_counts_for

FRUIT = ["apple", "pear", "plum", "grape", "fig"]
TECH = ["wire", "signal", "circuit", "battery", "antenna"]

CONFIG = TrainingConfig(
    dim=32,
    window=3,
    min_count=1,
    subsample=0.0,  # keep every token: the corpus is tiny and balanced
    compass_epochs=3,
    slice_epochs=3,
    batch_size=1024,
    seed=7,
    device="cpu",
)


def write_synthetic_corpus(root: Path, n: int = 300) -> TimeSlicedCorpus:
    rng = np.random.default_rng(0)

    def sentences(cluster: list[str], extra: str | None) -> list[str]:
        pool = cluster + ([extra] if extra else [])
        return [" ".join(rng.choice(pool, size=8)) for _ in range(n)]

    for year, shifty_cluster in ((1900, FRUIT), (2000, TECH)):
        lines = (
            sentences(FRUIT, "shifty" if shifty_cluster is FRUIT else None)
            + sentences(TECH, "shifty" if shifty_cluster is TECH else None)
        )
        rng.shuffle(lines)
        (root / f"{year}.txt").write_text("\n".join(lines))
    return TimeSlicedCorpus.from_directory(root, slice_width=100)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    corpus = write_synthetic_corpus(tmp_path_factory.mktemp("corpus"))
    model, freq = train_twec(corpus, CONFIG)
    emb = TemporalEmbeddings.from_twec(model, freq, slice_width=100)
    return corpus, model, freq, emb


def centroid(emb: TemporalEmbeddings, words: list[str], year: int) -> np.ndarray:
    vecs = np.stack([emb.vector(w, year) for w in words])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.mean(axis=0)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_context_matrix_is_frozen(trained):
    corpus, model, freq, _ = trained
    _, w_out = train_sgns(
        lambda: corpus.sentences(1900),
        model.vocab,
        stream_counts_for(model.vocab, freq, 1900),
        CONFIG,
        epochs=1,
        w_in_init=model.compass_in,
        w_out_init=model.compass_out,
        freeze_context=True,
        desc="freeze-check",
    )
    np.testing.assert_array_equal(w_out, model.compass_out)


def test_slice_vocab_drops_absent_words(trained):
    _, _, freq, emb = trained
    # shifty occurs in both slices here, but cluster words are slice-agnostic;
    # verify the invariant directly: every stored word has nonzero count.
    for year in emb.slice_years:
        for word in emb.slices[year].words:
            assert freq.count(word, year) > 0


def test_planted_shift_is_recovered(trained):
    *_, emb = trained
    s1900 = emb.vector("shifty", 1900)
    s2000 = emb.vector("shifty", 2000)
    assert cos(s1900, centroid(emb, FRUIT, 1900)) > cos(s1900, centroid(emb, TECH, 1900))
    assert cos(s2000, centroid(emb, TECH, 2000)) > cos(s2000, centroid(emb, FRUIT, 2000))


def test_stable_words_stay_aligned_across_slices(trained):
    *_, emb = trained
    for word in ("apple", "wire"):
        same = cos(emb.vector(word, 1900), emb.vector(word, 2000))
        other = "wire" if word == "apple" else "apple"
        cross = cos(emb.vector(word, 1900), emb.vector(other, 2000))
        assert same > cross, f"{word}: self-similarity {same} <= cross {cross}"


def test_neighbors_reflect_slice_context(trained):
    *_, emb = trained
    top1900 = {r.word for r in neighbors_of(emb, "shifty", 1900, k=3)}
    top2000 = {r.word for r in neighbors_of(emb, "shifty", 2000, k=3)}
    assert top1900 <= set(FRUIT)
    assert top2000 <= set(TECH)


def test_equivalent_query_crosses_time_correctly(trained):
    *_, emb = trained
    # shifty meant "fruit-ish" in 1900: its 2000-space equivalents are fruits
    forward = [r.word for r in equivalent(emb, "shifty", 1900, 2000, k=6)]
    fruit_rank = min(forward.index(w) for w in FRUIT if w in forward)
    tech_rank = min((forward.index(w) for w in TECH if w in forward), default=len(forward))
    assert fruit_rank < tech_rank

    backward = [r.word for r in equivalent(emb, "shifty", 2000, 1900, k=6)]
    tech_rank_b = min(backward.index(w) for w in TECH if w in backward)
    fruit_rank_b = min((backward.index(w) for w in FRUIT if w in backward), default=len(backward))
    assert tech_rank_b < fruit_rank_b


def test_training_is_deterministic(tmp_path):
    corpus = write_synthetic_corpus(tmp_path, n=50)
    cfg = TrainingConfig(
        dim=8, min_count=1, subsample=0.0, compass_epochs=1, slice_epochs=1,
        seed=7, device="cpu",
    )
    m1, _ = train_twec(corpus, cfg)
    m2, _ = train_twec(corpus, cfg)
    np.testing.assert_array_equal(m1.compass_in, m2.compass_in)
    np.testing.assert_array_equal(m1.slice_in[1900], m2.slice_in[1900])


def test_vocab_min_count():
    with pytest.raises(ValueError):
        Vocab.build({"a": 1}, min_count=5)
