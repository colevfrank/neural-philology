from pathlib import Path

import pytest

from neural_philology.corpus import (
    FrequencyTable,
    TimeSlicedCorpus,
    Vocab,
    slice_of,
    tokenize,
)


def make_corpus(tmp_path: Path) -> TimeSlicedCorpus:
    (tmp_path / "1953_news.txt").write_text("The cat sat.\nA dog ran!\n")
    (tmp_path / "1955.txt").write_text("the cat's dog\n")
    (tmp_path / "1962.txt").write_text("radio radio radio\n\n")
    return TimeSlicedCorpus.from_directory(tmp_path, slice_width=10)


def test_tokenize():
    assert tokenize("The cat's 2 hats!") == ["the", "cat's", "hats"]


def test_slice_of():
    assert slice_of(1999, 10) == 1990
    assert slice_of(1950, 10) == 1950
    assert slice_of(1907, 25) == 1900


def test_ingestion_groups_files_into_slices(tmp_path):
    corpus = make_corpus(tmp_path)
    assert corpus.slices == [1950, 1960]
    assert list(corpus.sentences(1960)) == [["radio"] * 3]
    # streams are re-iterable (one pass per epoch)
    assert list(corpus.sentences(1960)) == list(corpus.sentences(1960))


def test_ingestion_requires_yearful_files(tmp_path):
    (tmp_path / "noyear.txt").write_text("hello\n")
    with pytest.raises(ValueError):
        TimeSlicedCorpus.from_directory(tmp_path)


def test_frequency_table_roundtrip(tmp_path):
    corpus = make_corpus(tmp_path)
    ft = corpus.count_frequencies()
    assert ft.count("dog", 1950) == 2
    assert ft.count("radio", 1960) == 3
    assert ft.count("radio", 1950) == 0
    assert ft.rel_freq("radio", 1960) == 1.0
    assert ft.merged()["dog"] == 2

    path = tmp_path / "counts.json"
    ft.save(path)
    loaded = FrequencyTable.load(path)
    assert loaded.slices == [1950, 1960]
    assert loaded.count("dog", 1950) == 2


def test_vocab_build_applies_min_count(tmp_path):
    ft = make_corpus(tmp_path).count_frequencies()
    vocab = Vocab.build(ft.merged(), min_count=2)
    assert "radio" in vocab and "dog" in vocab and "the" in vocab
    assert "sat" not in vocab
    # deterministic order: by descending count, then alphabetical
    assert vocab.words[0] == "radio"
    assert vocab.index["radio"] == 0
