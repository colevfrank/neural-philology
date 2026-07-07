# Project: Semantic Word Evolution Explorer

I'm building a website that lets users explore how word meanings have changed over time using diachronic (time-sliced) word embeddings. I've already made the key research/design decisions; help me scaffold and build the project.

## Core functionality

1. **Word trajectory view**: User enters a word; the site shows its nearest semantic neighbors per time slice and a 2D trajectory visualization (t-SNE/UMAP projection of the word's vectors over time, plotted against its neighbors — similar to Figure 1 of Yao et al. 2018, "Dynamic Word Embeddings for Evolving Semantic Discovery").
2. **Cross-time equivalence search** (the flagship feature): User enters a word-year pair (e.g., "dope, 2018") and a target year (e.g., 1955); the site returns the nearest words to that vector *in the target year's embedding space*. This answers "what was the equivalent word back then?" Both directions should be supported.
3. **Frequency/confidence display**: For every word-year shown, display the word's raw frequency in that time slice, and visually flag (grey out / warn) results where frequency falls below a threshold — low-frequency positions are unreliable and I don't want to present fabricated confidence.

## Technical decisions already made (don't relitigate these)

- **Embedding method: TWEC / "compass" method** (Di Carlo et al. 2019, "Training Temporal Word Embeddings with a Compass"). Train one atemporal word2vec model on the full corpus as a frozen "compass"; train per-slice temporal embeddings against it. All slices are aligned by construction — no post-hoc rotation step. There's an existing Python implementation (github.com/valedica/twec) — evaluate whether to use it, adapt it, or reimplement against modern gensim.
- **Serving representation**: one dense vector per word per time slice (d≈100). Precompute top-k neighbor lists per word-year where feasible; use FAISS (or brute-force numpy for small vocab) for live cross-time queries.
- **Honesty mechanism**: per-word per-slice frequency counts published alongside all results (this substitutes for Bayesian uncertainty estimates — deliberate simplicity tradeoff).

## Phased plan

- **Phase 1 (this session)**: Project scaffold + data pipeline + a working CLI/notebook prototype of the equivalence query on a small test corpus. Suggested test corpus: COHA sample, or the NYT-style yearly slices, or HistWords' precomputed decade embeddings (sgns, English, 1800s–1990s) just to validate the query/serving logic before training anything.
- **Phase 2**: Train TWEC on a real corpus (likely Google Books Ngram–derived co-occurrences at decade granularity for historical depth, possibly a second vernacular track later). Build the evaluation harness: a hand-built testset of ~50 known cross-time equivalences (short- and long-range, e.g., obama-2012→bush-2002, app-2012→software-1990) scored by MRR and precision@k.
- **Phase 3**: Web frontend (trajectory viz, equivalence search UI, frequency flags) + static/precomputed serving layer.

## For this first session, please:

1. Set up the repo structure (Python; suggest and justify the stack for backend + eventual frontend).
2. Write the data-loading module: ingest a time-sliced corpus into per-slice token streams, compute per-word per-slice frequency counts, store vocab + counts.
3. Implement or wrap TWEC training (compass first, then per-slice models), with a config for slice granularity, min-frequency threshold, dimensions.
4. Implement the two core query functions against trained embeddings: (a) neighbors-of(word, year, k), (b) equivalent(word, source_year, target_year, k) — cosine similarity of the source vector against the target slice's matrix.
5. Stub the evaluation harness (testset format: CSV of query_word, query_year, target_year, expected_word; metrics: MRR, MP@1/5/10).

Ask me before making large downloads or long training runs. I'm comfortable with Python/ML (PhD student working in ML evaluation); prioritize correctness and a clean pipeline over premature optimization.

Commit after each completed task with a descriptive message
