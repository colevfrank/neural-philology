import numpy as np
import pytest

from neural_philology.config import QueryConfig
from neural_philology.corpus import FrequencyTable
from neural_philology.embeddings import SliceEmbeddings, TemporalEmbeddings
from neural_philology.queries import equivalent, neighbors_of, word_info


@pytest.fixture
def embeddings() -> TemporalEmbeddings:
    """Two slices, hand-set vectors. `shifty` moves from the x-axis cluster
    (with `alpha`) to the y-axis cluster (with `beta`)."""
    words = ["alpha", "beta", "shifty", "rare"]
    s1900 = SliceEmbeddings(
        words=words,
        vectors=np.array(
            [[1, 0.0], [0, 1], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32
        ),
    )
    s2000 = SliceEmbeddings(
        words=words,
        vectors=np.array(
            [[1, 0.0], [0, 1], [0.1, 0.9], [0.7, 0.3]], dtype=np.float32
        ),
    )
    freq = FrequencyTable(
        {
            1900: {"alpha": 100, "beta": 100, "shifty": 50, "rare": 3},
            2000: {"alpha": 100, "beta": 100, "shifty": 50, "rare": 2},
        }
    )
    return TemporalEmbeddings({1900: s1900, 2000: s2000}, freq=freq, slice_width=100)


def test_neighbors_excludes_query_word(embeddings):
    res = neighbors_of(embeddings, "shifty", 1900, k=3)
    assert [r.word for r in res] == ["alpha", "rare", "beta"]
    assert all(r.word != "shifty" for r in res)
    assert res[1].score == pytest.approx(
        np.dot([0.9, 0.1], [0.8, 0.2])
        / (np.linalg.norm([0.9, 0.1]) * np.linalg.norm([0.8, 0.2]))
    )


def test_equivalent_ranks_target_slice_by_source_vector(embeddings):
    # shifty@2000 points along y; its 1900 equivalent should be beta
    res = equivalent(embeddings, "shifty", 2000, 1900, k=4)
    assert res[0].word == "beta"
    assert res[0].slice_year == 1900
    # and in the other direction, shifty@1900 (x-ish) maps to alpha@2000
    back = equivalent(embeddings, "shifty", 1900, 2000, k=4)
    assert back[0].word == "alpha"


def test_equivalent_keeps_query_word(embeddings):
    res = equivalent(embeddings, "alpha", 1900, 2000, k=4)
    assert res[0].word == "alpha"  # stable word is its own equivalent


def test_low_frequency_flagging(embeddings):
    config = QueryConfig(low_frequency_threshold=10)
    res = neighbors_of(embeddings, "shifty", 1900, k=3, config=config)
    flags = {r.word: r.low_frequency for r in res}
    assert flags == {"rare": True, "alpha": False, "beta": False}
    assert {r.word: r.frequency for r in res}["rare"] == 3


def test_unknown_frequency_is_flagged(embeddings):
    embeddings.freq = None
    res = neighbors_of(embeddings, "shifty", 1900, k=1)
    assert res[0].frequency is None and res[0].low_frequency


def test_word_info_and_year_resolution(embeddings):
    info = word_info(embeddings, "rare", 1957, config=QueryConfig(low_frequency_threshold=10))
    assert info.slice_year == 1900  # slice_width=100 floors 1957 -> 1900
    assert info.low_frequency


def test_missing_word_and_year_raise(embeddings):
    with pytest.raises(KeyError, match="not in the 1900 slice"):
        neighbors_of(embeddings, "zeppelin", 1900)
    with pytest.raises(KeyError, match="no slice for year"):
        neighbors_of(embeddings, "alpha", 2525)


def test_save_load_roundtrip(embeddings, tmp_path):
    embeddings.save(tmp_path / "emb")
    loaded = TemporalEmbeddings.load(tmp_path / "emb")
    assert loaded.slice_years == [1900, 2000]
    assert loaded.slice_width == 100
    np.testing.assert_allclose(
        loaded.vector("shifty", 1900), embeddings.vector("shifty", 1900)
    )
    assert loaded.frequency("rare", 1900) == 3
    res = equivalent(loaded, "shifty", 2000, 1900, k=1)
    assert res[0].word == "beta"
