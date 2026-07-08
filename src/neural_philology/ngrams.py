"""Streaming pipeline for Google Books Ngrams v3 (20200217).

Nothing large ever lands on disk: each shard is streamed
(``curl | gunzip | awk``) and reduced on the fly to per-decade aggregates —
unigram frequency counts (the vocabulary + honesty-mechanism data) and
(center, context) co-occurrence counts within a ±2 window. Only the distilled
counts are stored (a few GB for eng-fiction 5-grams vs ~513 GB transferred).

Per-shard outputs are checkpointed, so an interrupted run resumes losing at
most the shards in flight.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .corpus import FrequencyTable

BASE = "http://storage.googleapis.com/books/ngrams/books/20200217"
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

SHARD_COUNTS = {("eng-fiction", 1): 1, ("eng-fiction", 5): 1449}


def shard_url(corpus: str, n: int, index: int) -> str:
    total = SHARD_COUNTS[(corpus, n)]
    return f"{BASE}/{corpus}/{n}-{index:05d}-of-{total:05d}.gz"


def _stream(url: str, awk_args: list[str]) -> list[subprocess.Popen]:
    """curl | gunzip | awk; returns all three processes (awk last).

    Callers must check every stage's exit status: a failed curl looks like a
    clean EOF to awk, which would otherwise checkpoint a truncated shard.
    """
    curl = subprocess.Popen(
        ["curl", "-sf", "--retry", "5", "--retry-all-errors", url],
        stdout=subprocess.PIPE,
    )
    gunzip = subprocess.Popen(
        ["gunzip", "-c"], stdin=curl.stdout, stdout=subprocess.PIPE
    )
    curl.stdout.close()
    awk = subprocess.Popen(
        awk_args, stdin=gunzip.stdout, stdout=subprocess.PIPE, text=True
    )
    gunzip.stdout.close()
    return [curl, gunzip, awk]


def _wait_all(procs: list[subprocess.Popen], what: str) -> None:
    names = ("curl", "gunzip", "awk")
    codes = [p.wait() for p in procs]
    for name, code in zip(names, codes):
        if code != 0:
            raise RuntimeError(f"{what}: {name} exited with {code}")


def build_unigram_table(
    corpus: str = "eng-fiction",
    out_dir: Path | str = "data/ngrams/eng-fiction",
    vocab_size: int = 50_000,
    min_count: float = 500,
) -> tuple[FrequencyTable, list[str]]:
    """Stream the 1-gram shard(s); write counts.json + vocab.txt.

    ``counts.json`` keeps only vocabulary words (full counts would be ~2M
    types); the vocabulary is the top ``vocab_size`` lowercase alphabetic
    words by total 1800-2019 count, subject to ``min_count``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[int, dict[str, float]] = {}
    totals: dict[str, float] = {}
    for i in range(SHARD_COUNTS[(corpus, 1)]):
        procs = _stream(
            shard_url(corpus, 1, i), ["awk", "-f", str(SCRIPTS / "ngram_unigram.awk")]
        )
        awk = procs[-1]
        assert awk.stdout is not None
        for line in tqdm(awk.stdout, desc=f"1-grams shard {i}", unit="type"):
            word, decade, count = line.rstrip("\n").split("\t")
            c = float(count)
            counts.setdefault(int(decade), {})[word] = (
                counts.get(int(decade), {}).get(word, 0.0) + c
            )
            totals[word] = totals.get(word, 0.0) + c
        _wait_all(procs, f"1-gram shard {i}")

    vocab = sorted(
        (w for w, c in totals.items() if c >= min_count),
        key=lambda w: (-totals[w], w),
    )[:vocab_size]
    keep = set(vocab)
    table = FrequencyTable(
        {d: {w: c for w, c in dc.items() if w in keep} for d, dc in counts.items()}
    )
    table.save(out_dir / "counts.json")
    (out_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    return table, vocab


def process_cooc_shard(
    corpus: str, index: int, vocab_index: dict[str, int], out_dir: Path
) -> Path:
    """Stream one 5-gram shard into an npz of (decade, center, ctx, weight)."""
    out_path = out_dir / f"shard-{index:05d}.npz"
    if out_path.exists():
        return out_path
    procs = _stream(
        shard_url(corpus, 5, index),
        [
            "awk", "-f", str(SCRIPTS / "ngram_cooc.awk"),
            str(out_dir.parent / "vocab.txt"), "-",
        ],
    )
    awk = procs[-1]
    decades, centers, ctxs, weights = [], [], [], []
    assert awk.stdout is not None
    for line in awk.stdout:
        decade, center, ctx, weight = line.rstrip("\n").split("\t")
        decades.append(int(decade))
        centers.append(vocab_index[center])
        ctxs.append(vocab_index[ctx])
        weights.append(int(weight))
    _wait_all(procs, f"5-gram shard {index}")
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        decade=np.asarray(decades, dtype=np.int16),
        center=np.asarray(centers, dtype=np.int32),
        ctx=np.asarray(ctxs, dtype=np.int32),
        weight=np.asarray(weights, dtype=np.int64),
    )
    tmp.rename(out_path)
    return out_path


def build_cooccurrences(
    corpus: str = "eng-fiction",
    data_dir: Path | str = "data/ngrams/eng-fiction",
    start: int = 0,
    end: int | None = None,
    workers: int = 4,
) -> None:
    """Stream 5-gram shards [start, end) with per-shard checkpointing."""
    data_dir = Path(data_dir)
    vocab = (data_dir / "vocab.txt").read_text().split()
    vocab_index = {w: i for i, w in enumerate(vocab)}
    out_dir = data_dir / "cooc"
    out_dir.mkdir(parents=True, exist_ok=True)
    end = SHARD_COUNTS[(corpus, 5)] if end is None else end
    todo = [
        i for i in range(start, end)
        if not (out_dir / f"shard-{i:05d}.npz").exists()
    ]
    if not todo:
        print(f"shards {start}..{end}: all done")
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_cooc_shard, corpus, i, vocab_index, out_dir): i
            for i in todo
        }
        for f in tqdm(as_completed(futures), total=len(todo), desc="5-gram shards"):
            f.result()  # propagate failures


def merge_cooccurrences(
    data_dir: Path | str = "data/ngrams/eng-fiction",
    out_name: str = "merged",
) -> None:
    """Combine shard npz files into one (center, ctx, weight) file per decade."""
    data_dir = Path(data_dir)
    shard_files = sorted((data_dir / "cooc").glob("shard-*.npz"))
    if not shard_files:
        raise ValueError(f"no shard files under {data_dir / 'cooc'}")
    out_dir = data_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = len((data_dir / "vocab.txt").read_text().split())

    decades = set()
    for f in shard_files:
        decades.update(np.unique(np.load(f)["decade"]).tolist())

    for decade in tqdm(sorted(decades), desc="merging decades"):
        codes_parts, weight_parts = [], []
        for f in shard_files:
            data = np.load(f)
            mask = data["decade"] == decade
            if not mask.any():
                continue
            codes_parts.append(
                data["center"][mask].astype(np.int64) * vocab_size
                + data["ctx"][mask]
            )
            weight_parts.append(data["weight"][mask])
        codes = np.concatenate(codes_parts)
        weights = np.concatenate(weight_parts)
        uniq, inverse = np.unique(codes, return_inverse=True)
        summed = np.zeros(len(uniq), dtype=np.int64)
        np.add.at(summed, inverse, weights)
        np.savez(
            out_dir / f"cooc_{decade}.npz",
            center=(uniq // vocab_size).astype(np.int32),
            ctx=(uniq % vocab_size).astype(np.int32),
            weight=summed,
        )
