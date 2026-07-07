# neural-philology

Explore how word meanings change over time using diachronic (time-sliced) word
embeddings. Two core queries:

- **Trajectory**: nearest semantic neighbors of a word per time slice, plus a 2D
  projection of its drift.
- **Cross-time equivalence**: given `(word, source_year)` and a `target_year`,
  find the nearest words to that vector *in the target year's embedding space* —
  "what was the equivalent word back then?" (works in both directions).

Every reported word-year carries its raw frequency in that slice; low-frequency
results are flagged as unreliable rather than presented with false confidence.

## Method

Embeddings are trained with the **TWEC / compass** method (Di Carlo et al. 2019):
one atemporal word2vec (SGNS) model is trained on the full corpus, its *context*
matrix is frozen as a shared "compass", and per-slice target embeddings are
trained against it. All slices land in the same coordinate system by
construction — no post-hoc Procrustes alignment.

The reference implementations ([twec](https://github.com/valedica/twec), cade)
hard-pin a forked gensim 3.x that no longer builds on modern Python/ARM, and
gensim 4 removed the internals they patch. We therefore **reimplement SGNS in
PyTorch** (`neural_philology.twec`): a small, testable training loop where
freezing the context matrix is a one-liner (`requires_grad=False`), giving exact
TWEC semantics on modern dependencies.

## Stack

- **Pipeline / training**: Python 3.12+, numpy, PyTorch (MPS/CUDA-capable),
  managed with [uv](https://docs.astral.sh/uv/). Chosen for correctness and
  testability over raw throughput; corpora at decade granularity train fine
  without gensim's C hot loop.
- **Serving (Phase 3)**: precompute per-word-year neighbor lists to static JSON
  where feasible; a thin FastAPI service backed by numpy brute-force cosine
  (FAISS if vocab × slices outgrows it) for live cross-time queries. Most
  traffic never touches the live path.
- **Frontend (Phase 3)**: Vite + React + a declarative plotting layer (Plotly or
  visx) for the trajectory view. Static-first: the site is deployable as flat
  files plus one small query endpoint.

## Layout

```
src/neural_philology/
  corpus.py        # time-sliced corpus ingestion, per-slice vocab + frequency counts
  config.py        # training/query configuration
  twec/            # PyTorch SGNS + compass/slice trainers
  embeddings.py    # TemporalEmbeddings container, save/load
  queries.py       # neighbors_of(), equivalent()
  evaluation.py    # cross-time equivalence testset scoring (MRR, MP@k)
  histwords.py     # loader for HistWords precomputed decade embeddings
  cli.py           # nphil command-line interface
eval/testsets/     # CSV testsets: query_word,query_year,target_year,expected_word
tests/
```

## Quickstart

```bash
uv sync
uv run pytest
uv run nphil --help
```

**Do not keep this repo inside an iCloud-synced folder** (e.g. `~/Documents`):
iCloud re-flags files in dot-directories as hidden, Python skips hidden `.pth`
files (breaking editable installs), and syncing `.git`/venvs invites corruption.
It lives in `~/dev/` for that reason.
