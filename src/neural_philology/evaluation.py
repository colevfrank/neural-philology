"""Evaluation harness for cross-time equivalence queries.

Testset format: CSV with header ``query_word,query_year,target_year,expected_word``.
Each row asks: does ``equivalent(query_word, query_year, target_year)`` rank
``expected_word`` highly in the target slice?

Metrics:
- **MRR** (truncated at ``max_rank``): mean reciprocal rank of the expected
  word; 0 when it is not in the top ``max_rank``.
- **MP@k** for k in {1, 5, 10}: fraction of scored cases where the expected
  word appears in the top k.

Cases whose query word is missing from the source slice are *skipped* (the
query cannot be posed) and reported separately. Cases whose expected word is
absent from the target slice count as *misses* — the system had a chance to
know the word and does not, and hiding that would inflate scores.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .config import QueryConfig
from .embeddings import TemporalEmbeddings
from .queries import equivalent

KS = (1, 5, 10)


@dataclass(frozen=True)
class EvalCase:
    query_word: str
    query_year: int
    target_year: int
    expected_word: str


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    rank: int | None  # 1-based rank of expected word; None = not in top max_rank
    skipped: str | None = None  # reason, if the case could not be scored


@dataclass
class EvalReport:
    mrr: float
    mp_at: dict[int, float]
    n_scored: int
    n_skipped: int
    results: list[CaseResult] = field(repr=False, default_factory=list)

    def summary(self) -> str:
        lines = [
            f"scored {self.n_scored} cases ({self.n_skipped} skipped)",
            f"MRR      {self.mrr:.4f}",
        ]
        lines += [f"MP@{k:<3}   {v:.4f}" for k, v in sorted(self.mp_at.items())]
        return "\n".join(lines)


def load_testset(path: Path | str) -> list[EvalCase]:
    cases = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"query_word", "query_year", "target_year", "expected_word"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"testset must have columns {sorted(required)}")
        for row in reader:
            cases.append(
                EvalCase(
                    query_word=row["query_word"].strip(),
                    query_year=int(row["query_year"]),
                    target_year=int(row["target_year"]),
                    expected_word=row["expected_word"].strip(),
                )
            )
    if not cases:
        raise ValueError(f"empty testset: {path}")
    return cases


def evaluate(
    embeddings: TemporalEmbeddings,
    cases: list[EvalCase],
    max_rank: int = 100,
    config: QueryConfig | None = None,
) -> EvalReport:
    config = config or QueryConfig()
    results: list[CaseResult] = []
    for case in cases:
        try:
            ranking = equivalent(
                embeddings,
                case.query_word,
                case.query_year,
                case.target_year,
                k=max_rank,
                config=config,
            )
        except KeyError as err:
            results.append(CaseResult(case, rank=None, skipped=str(err)))
            continue
        rank = next(
            (i + 1 for i, r in enumerate(ranking) if r.word == case.expected_word),
            None,
        )
        results.append(CaseResult(case, rank=rank))

    scored = [r for r in results if r.skipped is None]
    if not scored:
        raise ValueError("no scorable cases (all skipped)")
    mrr = sum(1.0 / r.rank for r in scored if r.rank is not None) / len(scored)
    mp_at = {
        k: sum(1 for r in scored if r.rank is not None and r.rank <= k) / len(scored)
        for k in KS
    }
    return EvalReport(
        mrr=mrr,
        mp_at=mp_at,
        n_scored=len(scored),
        n_skipped=len(results) - len(scored),
        results=results,
    )
