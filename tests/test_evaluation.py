import numpy as np
import pytest

from neural_philology.corpus import FrequencyTable
from neural_philology.embeddings import SliceEmbeddings, TemporalEmbeddings
from neural_philology.evaluation import EvalCase, evaluate, load_testset


@pytest.fixture
def embeddings() -> TemporalEmbeddings:
    words = ["a", "b", "c"]
    vecs1900 = np.array([[1, 0], [0, 1], [0.6, 0.4]], dtype=np.float32)
    vecs2000 = np.array([[1, 0], [0, 1], [0.4, 0.6]], dtype=np.float32)
    freq = FrequencyTable(
        {1900: {w: 100 for w in words}, 2000: {w: 100 for w in words}}
    )
    return TemporalEmbeddings(
        {
            1900: SliceEmbeddings(words, vecs1900),
            2000: SliceEmbeddings(words, vecs2000),
        },
        freq=freq,
        slice_width=100,
    )


def test_load_testset(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(
        "query_word,query_year,target_year,expected_word\n"
        "car,1990,1900,carriage\n"
        " radio ,1990,1850,telegraph\n"
    )
    cases = load_testset(path)
    assert cases[0] == EvalCase("car", 1990, 1900, "carriage")
    assert cases[1].query_word == "radio"  # whitespace stripped


def test_load_testset_rejects_bad_header(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("word,year\nfoo,1990\n")
    with pytest.raises(ValueError, match="columns"):
        load_testset(path)


def test_evaluate_metrics(embeddings):
    cases = [
        EvalCase("a", 1900, 2000, "a"),  # rank 1
        EvalCase("b", 1900, 2000, "a"),  # a is orthogonal to b -> rank 3
        EvalCase("zeppelin", 1900, 2000, "a"),  # query OOV -> skipped
        EvalCase("a", 1900, 2000, "notaword"),  # expected absent -> miss
    ]
    report = evaluate(embeddings, cases)
    assert report.n_scored == 3
    assert report.n_skipped == 1
    assert report.mrr == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)
    assert report.mp_at[1] == pytest.approx(1 / 3)
    assert report.mp_at[5] == pytest.approx(2 / 3)
    assert "MRR" in report.summary()


def test_evaluate_all_skipped_raises(embeddings):
    with pytest.raises(ValueError, match="no scorable"):
        evaluate(embeddings, [EvalCase("zeppelin", 1900, 2000, "a")])
