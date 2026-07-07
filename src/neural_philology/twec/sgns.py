"""Skip-gram with negative sampling (SGNS) in PyTorch.

A deliberately small reimplementation of word2vec's SGNS objective so that the
TWEC/compass method can be expressed exactly: per-slice models reuse the
compass's context matrix with ``requires_grad=False``. Follows word2vec
conventions: dynamic window (uniform 1..window), frequency subsampling,
unigram^0.75 negative sampling, linear learning-rate decay, target matrix
init uniform(-0.5/dim, 0.5/dim), context matrix init zeros. One deliberate
departure: Adam instead of per-pair SGD, because minibatched gradients scale
as 1/batch_size and Adam's per-parameter normalisation absorbs that.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..config import TrainingConfig
from ..corpus import Vocab

SentenceFactory = Callable[[], Iterable[list[str]]]


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def keep_probs(stream_counts: np.ndarray, sample: float) -> np.ndarray:
    """word2vec subsampling: P(keep w) = (sqrt(f/t) + 1) * t/f, clipped to 1."""
    probs = np.ones_like(stream_counts, dtype=np.float64)
    if sample <= 0:
        return probs
    total = stream_counts.sum()
    if total == 0:
        return probs
    f = stream_counts / total
    nz = f > 0
    probs[nz] = np.minimum(1.0, (np.sqrt(f[nz] / sample) + 1) * sample / f[nz])
    return probs


def iter_pair_batches(
    sentences: Iterable[list[str]],
    vocab: Vocab,
    window: int,
    keep: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (centers, contexts) id batches with subsampling + dynamic window."""
    centers: list[int] = []
    contexts: list[int] = []
    index = vocab.index
    for sent in sentences:
        ids = [index[w] for w in sent if w in index]
        if len(ids) < 2:
            continue
        kept = [i for i in ids if keep[i] >= 1.0 or rng.random() < keep[i]]
        n = len(kept)
        if n < 2:
            continue
        reduced = rng.integers(1, window + 1, size=n)
        for pos, center in enumerate(kept):
            w = int(reduced[pos])
            for ctx_pos in range(max(0, pos - w), min(n, pos + w + 1)):
                if ctx_pos == pos:
                    continue
                centers.append(center)
                contexts.append(kept[ctx_pos])
        while len(centers) >= batch_size:
            yield (
                np.asarray(centers[:batch_size], dtype=np.int64),
                np.asarray(contexts[:batch_size], dtype=np.int64),
            )
            del centers[:batch_size], contexts[:batch_size]
    if centers:
        yield (
            np.asarray(centers, dtype=np.int64),
            np.asarray(contexts, dtype=np.int64),
        )


class SGNS(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        w_in_init: np.ndarray | None = None,
        w_out_init: np.ndarray | None = None,
        freeze_context: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.w_in = torch.nn.Embedding(vocab_size, dim)
        self.w_out = torch.nn.Embedding(vocab_size, dim)
        with torch.no_grad():
            if w_in_init is not None:
                self.w_in.weight.copy_(torch.from_numpy(w_in_init))
            else:
                gen = torch.Generator().manual_seed(seed)
                self.w_in.weight.uniform_(-0.5 / dim, 0.5 / dim, generator=gen)
            if w_out_init is not None:
                self.w_out.weight.copy_(torch.from_numpy(w_out_init))
            else:
                self.w_out.weight.zero_()
        if freeze_context:
            self.w_out.weight.requires_grad_(False)

    def loss(
        self,
        centers: torch.Tensor,
        contexts: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        v = self.w_in(centers)  # (B, d)
        u_pos = self.w_out(contexts)  # (B, d)
        u_neg = self.w_out(negatives)  # (B, n, d)
        pos = F.logsigmoid((v * u_pos).sum(-1))
        neg = F.logsigmoid(-torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


def train_sgns(
    sentence_factory: SentenceFactory,
    vocab: Vocab,
    stream_counts: np.ndarray,
    config: TrainingConfig,
    epochs: int,
    *,
    w_in_init: np.ndarray | None = None,
    w_out_init: np.ndarray | None = None,
    freeze_context: bool = False,
    desc: str = "sgns",
) -> tuple[np.ndarray, np.ndarray]:
    """Train SGNS over ``epochs`` passes; returns (W_in, W_out) as float32.

    ``stream_counts`` are the token counts of *this* training stream aligned to
    ``vocab`` — they drive subsampling and the negative-sampling distribution.
    """
    device = resolve_device(config.device)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    model = SGNS(
        len(vocab),
        config.dim,
        w_in_init=w_in_init,
        w_out_init=w_out_init,
        freeze_context=freeze_context,
        seed=config.seed,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # Adam rather than word2vec's per-pair SGD: minibatching divides each
    # pair's gradient by batch_size, which plain SGD can't absorb (per-pair
    # steps shrink ~1/B) — Adam's per-parameter normalisation is invariant to
    # that scaling, so training behaves consistently across batch sizes.
    optimizer = torch.optim.Adam(params, lr=config.lr)

    noise = np.power(stream_counts, 0.75)
    if noise.sum() == 0:
        raise ValueError("training stream contains no in-vocabulary tokens")
    noise_dist = torch.from_numpy(noise / noise.sum()).to(device)

    keep = keep_probs(stream_counts, config.subsample)
    # Estimated total pairs for linear lr decay: kept tokens * E[2 * reduced window].
    kept_tokens = float((stream_counts * keep).sum())
    est_total_pairs = max(1.0, epochs * kept_tokens * (config.window + 1))
    pairs_seen = 0

    for epoch in range(epochs):
        batches = iter_pair_batches(
            sentence_factory(), vocab, config.window, keep, rng, config.batch_size
        )
        progress = tqdm(batches, desc=f"{desc} epoch {epoch + 1}/{epochs}", unit="batch")
        for centers_np, contexts_np in progress:
            frac = min(1.0, pairs_seen / est_total_pairs)
            lr = max(config.min_lr, config.lr * (1.0 - frac))
            for group in optimizer.param_groups:
                group["lr"] = lr
            pairs_seen += len(centers_np)

            centers = torch.from_numpy(centers_np).to(device)
            contexts = torch.from_numpy(contexts_np).to(device)
            negatives = torch.multinomial(
                noise_dist, len(centers_np) * config.negative, replacement=True
            ).view(len(centers_np), config.negative)

            loss = model.loss(centers, contexts, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix(
                loss=f"{loss.item() / len(centers_np):.4f}", lr=f"{lr:.4f}"
            )

    w_in = model.w_in.weight.detach().cpu().numpy().astype(np.float32)
    w_out = model.w_out.weight.detach().cpu().numpy().astype(np.float32)
    return w_in, w_out
