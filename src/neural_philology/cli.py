"""nphil — command-line interface for the semantic word evolution explorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import QueryConfig, TrainingConfig
from .embeddings import TemporalEmbeddings
from .queries import QueryResult, equivalent, neighbors_of, word_info


def _fmt_freq(freq: float | None) -> str:
    if freq is None:
        return "?"
    if freq >= 1:
        return f"{freq:,.0f}"
    return f"{freq:.2e}"  # relative frequency (e.g. HistWords)


def _print_results(results: list[QueryResult], header: str) -> None:
    print(header)
    print(f"{'rank':>4}  {'word':<20} {'cosine':>7}  {'freq':>10}")
    for rank, r in enumerate(results, 1):
        flag = "  ⚠ low frequency — unreliable" if r.low_frequency else ""
        print(
            f"{rank:>4}  {r.word:<20} {r.score:>7.3f}  {_fmt_freq(r.frequency):>10}{flag}"
        )


def _load(emb_dir: str) -> TemporalEmbeddings:
    path = Path(emb_dir)
    if not (path / "meta.json").exists():
        sys.exit(f"error: no embeddings at {path} (expected meta.json)")
    return TemporalEmbeddings.load(path)


def _query_config(args: argparse.Namespace) -> QueryConfig:
    return QueryConfig(low_frequency_threshold=args.low_freq_threshold)


def cmd_neighbors(args: argparse.Namespace) -> None:
    emb = _load(args.embeddings)
    config = _query_config(args)
    info = word_info(emb, args.word, args.year, config)
    note = "  ⚠ low frequency — position unreliable" if info.low_frequency else ""
    print(
        f"query: {args.word} @ {info.slice_year} "
        f"(freq {_fmt_freq(info.frequency)}){note}\n"
    )
    results = neighbors_of(emb, args.word, args.year, k=args.k, config=config)
    _print_results(results, f"nearest neighbors of '{args.word}' in {info.slice_year}:")


def cmd_equivalent(args: argparse.Namespace) -> None:
    emb = _load(args.embeddings)
    config = _query_config(args)
    info = word_info(emb, args.word, args.source_year, config)
    note = "  ⚠ low frequency — position unreliable" if info.low_frequency else ""
    print(
        f"query: {args.word} @ {info.slice_year} "
        f"(freq {_fmt_freq(info.frequency)}){note}\n"
    )
    results = equivalent(
        emb, args.word, args.source_year, args.target_year, k=args.k, config=config
    )
    target = emb.resolve_slice(args.target_year)
    _print_results(
        results,
        f"closest words in {target} to '{args.word}' as used in {info.slice_year}:",
    )


def cmd_convert_histwords(args: argparse.Namespace) -> None:
    from .histwords import convert_histwords

    emb = convert_histwords(args.sgns_dir, args.out, args.freqs)
    print(f"wrote {len(emb.slice_years)} slices to {args.out}: {emb.slice_years}")


def cmd_train(args: argparse.Namespace) -> None:
    from .corpus import TimeSlicedCorpus
    from .twec import train_twec

    corpus = TimeSlicedCorpus.from_directory(args.corpus, slice_width=args.slice_width)
    config = TrainingConfig(
        dim=args.dim,
        window=args.window,
        min_count=args.min_count,
        compass_epochs=args.epochs,
        slice_epochs=args.epochs,
    )
    model, freq = train_twec(corpus, config)
    emb = TemporalEmbeddings.from_twec(model, freq, slice_width=args.slice_width)
    emb.save(args.out)
    print(f"wrote {len(emb.slice_years)} slices to {args.out}")


def cmd_ngram_vocab(args: argparse.Namespace) -> None:
    from .ngrams import build_unigram_table

    table, vocab = build_unigram_table(
        corpus=args.corpus, out_dir=args.data_dir, vocab_size=args.vocab_size
    )
    print(f"decades: {table.slices}")
    print(f"vocab: {len(vocab)} words -> {args.data_dir}/vocab.txt")


def cmd_ngram_cooc(args: argparse.Namespace) -> None:
    from .ngrams import build_cooccurrences

    build_cooccurrences(
        corpus=args.corpus,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        workers=args.workers,
    )


def cmd_ngram_merge(args: argparse.Namespace) -> None:
    from .ngrams import merge_cooccurrences

    merge_cooccurrences(data_dir=args.data_dir)


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluation import evaluate, load_testset

    emb = _load(args.embeddings)
    report = evaluate(
        emb, load_testset(args.testset), max_rank=args.max_rank,
        config=_query_config(args),
    )
    print(report.summary())
    if args.verbose:
        for r in report.results:
            c = r.case
            outcome = (
                f"skipped ({r.skipped})" if r.skipped
                else f"rank {r.rank}" if r.rank else f"not in top {args.max_rank}"
            )
            print(
                f"  {c.query_word}@{c.query_year} -> {c.expected_word}@{c.target_year}: {outcome}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nphil", description="Explore semantic change over time."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_query_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("-e", "--embeddings", default="data/embeddings/histwords")
        p.add_argument("-k", type=int, default=10, help="number of results")
        p.add_argument(
            "--low-freq-threshold",
            type=float,
            default=1e-6,
            help="frequency below which results are flagged unreliable "
            "(relative-frequency scale for HistWords; raw counts for TWEC output)",
        )

    p = sub.add_parser("neighbors", help="nearest neighbors of a word in one slice")
    p.add_argument("word")
    p.add_argument("year", type=int)
    add_query_opts(p)
    p.set_defaults(func=cmd_neighbors)

    p = sub.add_parser(
        "equivalent", help="cross-time equivalents of (word, source_year) in target_year"
    )
    p.add_argument("word")
    p.add_argument("source_year", type=int)
    p.add_argument("target_year", type=int)
    add_query_opts(p)
    p.set_defaults(func=cmd_equivalent)

    p = sub.add_parser(
        "convert-histwords", help="convert extracted HistWords sgns files to serving format"
    )
    p.add_argument("--sgns-dir", default="data/histwords/sgns")
    p.add_argument("--freqs", default=None, help="path to HistWords freqs.pkl")
    p.add_argument("--out", default="data/embeddings/histwords")
    p.set_defaults(func=cmd_convert_histwords)

    p = sub.add_parser("train", help="train TWEC on a time-sliced text corpus")
    p.add_argument("--corpus", required=True, help="directory of <year>*.txt files")
    p.add_argument("--out", required=True)
    p.add_argument("--slice-width", type=int, default=10)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.set_defaults(func=cmd_train)

    def add_ngram_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus", default="eng-fiction")
        p.add_argument("--data-dir", default="data/ngrams/eng-fiction")

    p = sub.add_parser("ngram-vocab", help="stream 1-grams into vocab + frequency table")
    add_ngram_opts(p)
    p.add_argument("--vocab-size", type=int, default=50_000)
    p.set_defaults(func=cmd_ngram_vocab)

    p = sub.add_parser("ngram-cooc", help="stream 5-gram shards into co-occurrence counts")
    add_ngram_opts(p)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_ngram_cooc)

    p = sub.add_parser("ngram-merge", help="merge shard co-occurrence files per decade")
    add_ngram_opts(p)
    p.set_defaults(func=cmd_ngram_merge)

    p = sub.add_parser("eval", help="score a cross-time equivalence testset")
    p.add_argument("--testset", required=True)
    p.add_argument("--max-rank", type=int, default=100)
    p.add_argument("-v", "--verbose", action="store_true")
    add_query_opts(p)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyError as err:
        sys.exit(f"error: {err.args[0]}")


if __name__ == "__main__":
    main()
