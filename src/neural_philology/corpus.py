"""Time-sliced corpus ingestion.

A corpus is a set of text files, each carrying a 4-digit year in its filename
(e.g. ``1953_nyt.txt``). Files are grouped into slices of ``slice_width`` years
(decades by default). Each line of a file is treated as one sentence/document
for training purposes. Token streams are read lazily from disk so the corpus
can be iterated multiple times (one pass per training epoch) without holding
it in memory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

Tokenizer = Callable[[str], list[str]]


def tokenize(text: str) -> list[str]:
    """Lowercase and extract alphabetic tokens (apostrophes allowed inside)."""
    return TOKEN_RE.findall(text.lower())


def slice_of(year: int, slice_width: int) -> int:
    """Map a year to its slice label (the slice's first year)."""
    return (year // slice_width) * slice_width


class FrequencyTable:
    """Per-word per-slice frequency counts.

    This is the project's honesty mechanism: every query result is reported
    together with the word's frequency in that slice, and low-frequency
    word-slice positions are flagged as unreliable. Values are floats so that
    corpora for which only relative/scaled frequencies exist (e.g. HistWords)
    can reuse the same machinery.
    """

    def __init__(self, counts: dict[int, dict[str, float]]):
        self._counts = counts
        self._totals = {s: float(sum(c.values())) for s, c in counts.items()}

    @property
    def slices(self) -> list[int]:
        return sorted(self._counts)

    def count(self, word: str, slice_year: int) -> float:
        return self._counts.get(slice_year, {}).get(word, 0.0)

    def total(self, slice_year: int) -> float:
        return self._totals.get(slice_year, 0.0)

    def rel_freq(self, word: str, slice_year: int) -> float:
        total = self.total(slice_year)
        return self.count(word, slice_year) / total if total else 0.0

    def vocab(self, slice_year: int | None = None) -> set[str]:
        if slice_year is not None:
            return set(self._counts.get(slice_year, {}))
        return {w for c in self._counts.values() for w in c}

    def merged(self) -> dict[str, float]:
        """Aggregate counts across all slices (compass vocabulary)."""
        merged: Counter[str] = Counter()
        for c in self._counts.values():
            merged.update(c)
        return dict(merged)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(s): c for s, c in self._counts.items()}
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> FrequencyTable:
        payload = json.loads(Path(path).read_text())
        return cls({int(s): c for s, c in payload.items()})


@dataclass(frozen=True)
class Vocab:
    """Fixed word <-> index mapping shared by compass and slice models."""

    words: tuple[str, ...]
    counts: tuple[float, ...]  # corpus-wide counts, aligned with `words`

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index", {w: i for i, w in enumerate(self.words)}
        )

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    @classmethod
    def build(cls, counts: dict[str, float], min_count: float) -> Vocab:
        kept = sorted(
            ((w, c) for w, c in counts.items() if c >= min_count),
            key=lambda wc: (-wc[1], wc[0]),
        )
        if not kept:
            raise ValueError(f"no words with count >= {min_count}")
        words, kept_counts = zip(*kept)
        return cls(words=words, counts=kept_counts)


class TimeSlicedCorpus:
    """Lazy, re-iterable access to per-slice sentence streams."""

    def __init__(
        self,
        files_by_slice: dict[int, list[Path]],
        slice_width: int,
        tokenizer: Tokenizer = tokenize,
    ):
        if not files_by_slice:
            raise ValueError("corpus has no files")
        self.files_by_slice = {s: sorted(fs) for s, fs in files_by_slice.items()}
        self.slice_width = slice_width
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(
        cls,
        root: Path | str,
        slice_width: int = 10,
        tokenizer: Tokenizer = tokenize,
        glob: str = "*.txt",
    ) -> TimeSlicedCorpus:
        """Group ``root``'s text files into slices by the year in each filename."""
        root = Path(root)
        files_by_slice: dict[int, list[Path]] = {}
        skipped = []
        for f in sorted(root.rglob(glob)):
            m = YEAR_RE.search(f.name)
            if m is None:
                skipped.append(f)
                continue
            files_by_slice.setdefault(
                slice_of(int(m.group(1)), slice_width), []
            ).append(f)
        if not files_by_slice:
            raise ValueError(
                f"no {glob} files with a 4-digit year in their name under {root}"
            )
        if skipped:
            names = ", ".join(f.name for f in skipped[:5])
            print(f"corpus: skipped {len(skipped)} file(s) without a year: {names}")
        return cls(files_by_slice, slice_width, tokenizer)

    @property
    def slices(self) -> list[int]:
        return sorted(self.files_by_slice)

    def sentences(self, slice_year: int) -> Iterator[list[str]]:
        """Yield tokenized sentences (one per non-empty line) for a slice."""
        for f in self.files_by_slice[slice_year]:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    tokens = self.tokenizer(line)
                    if tokens:
                        yield tokens

    def all_sentences(self) -> Iterator[list[str]]:
        """Yield sentences from every slice (compass training stream)."""
        for slice_year in self.slices:
            yield from self.sentences(slice_year)

    def count_frequencies(self) -> FrequencyTable:
        """One pass over the corpus: raw per-word counts for each slice."""
        counts: dict[int, dict[str, float]] = {}
        for slice_year in self.slices:
            counter: Counter[str] = Counter()
            for sent in self.sentences(slice_year):
                counter.update(sent)
            counts[slice_year] = dict(counter)
        return FrequencyTable(counts)
