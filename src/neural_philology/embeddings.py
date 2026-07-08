"""Serving representation: one dense vector per word per time slice.

``TemporalEmbeddings`` holds, for each slice, a vocabulary list and a
``(V, d)`` float32 matrix, plus the per-slice frequency table used for
reliability flagging. Slices may have different vocabularies: a word only
appears in a slice's matrix if it actually occurred in that slice — a vector
for a word with zero occurrences would be pure fabrication.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .corpus import FrequencyTable, slice_of


@dataclass
class SliceEmbeddings:
    words: list[str]
    vectors: np.ndarray  # (V, d) float32

    def __post_init__(self) -> None:
        if len(self.words) != self.vectors.shape[0]:
            raise ValueError("words/vectors length mismatch")
        self.index = {w: i for i, w in enumerate(self.words)}

    @cached_property
    def normalized(self) -> np.ndarray:
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        return self.vectors / np.maximum(norms, 1e-12)


class TemporalEmbeddings:
    def __init__(
        self,
        slices: dict[int, SliceEmbeddings],
        freq: FrequencyTable | None = None,
        slice_width: int = 10,
    ):
        if not slices:
            raise ValueError("no slices")
        self.slices = slices
        self.freq = freq
        self.slice_width = slice_width

    @property
    def slice_years(self) -> list[int]:
        return sorted(self.slices)

    def resolve_slice(self, year: int) -> int:
        """Map an arbitrary year to an available slice (e.g. 2018 -> 2010)."""
        if year in self.slices:
            return year
        floored = slice_of(year, self.slice_width)
        if floored in self.slices:
            return floored
        raise KeyError(
            f"no slice for year {year}; available: {self.slice_years}"
        )

    def vector(self, word: str, year: int) -> np.ndarray:
        s = self.resolve_slice(year)
        emb = self.slices[s]
        if word not in emb.index:
            raise KeyError(f"{word!r} not in the {s} slice vocabulary")
        return emb.vectors[emb.index[word]]

    def frequency(self, word: str, year: int) -> float | None:
        if self.freq is None:
            return None
        return self.freq.count(word, self.resolve_slice(year))

    @classmethod
    def from_twec(cls, model, freq: FrequencyTable, slice_width: int = 10) -> TemporalEmbeddings:
        """Build from a trained TwecModel, dropping zero-count word-slices."""
        slices = {}
        for slice_year, matrix in model.slice_in.items():
            keep = [
                i
                for i, w in enumerate(model.vocab.words)
                if freq.count(w, slice_year) > 0
            ]
            slices[slice_year] = SliceEmbeddings(
                words=[model.vocab.words[i] for i in keep],
                vectors=matrix[keep].astype(np.float32),
            )
        return cls(slices, freq=freq, slice_width=slice_width)

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dims = {e.vectors.shape[1] for e in self.slices.values()}
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "slice_width": self.slice_width,
                    "slices": self.slice_years,
                    "dim": dims.pop() if len(dims) == 1 else sorted(dims),
                }
            )
        )
        if self.freq is not None:
            self.freq.save(directory / "counts.json")
        for slice_year, emb in self.slices.items():
            np.savez(
                directory / f"slice_{slice_year}.npz",
                vectors=emb.vectors,
                words=np.asarray(emb.words, dtype=object),
            )

    @classmethod
    def load(cls, directory: Path | str) -> TemporalEmbeddings:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        freq = None
        if (directory / "counts.json").exists():
            freq = FrequencyTable.load(directory / "counts.json")
        slices = {}
        for slice_year in meta["slices"]:
            data = np.load(directory / f"slice_{slice_year}.npz", allow_pickle=True)
            slices[slice_year] = SliceEmbeddings(
                words=list(data["words"]),
                vectors=data["vectors"].astype(np.float32),
            )
        return cls(slices, freq=freq, slice_width=meta["slice_width"])
