"""Memorization analysis: how much of what the model writes exists in its training data?

Three complementary measurements per generated text:

1. **Corpus-wide verbatim n-gram overlap.** Every token n-gram of the
   generation (n = 8, 16, 32 by default) is checked against *every* n-gram of
   the packed training stream in a single vectorized pass (64-bit rolling
   hashes; collisions are negligible at this scale). Reported as the fraction
   of the generation's n-grams that occur somewhere in training, and the
   largest checked n with at least one hit.
2. **Nearest training story.** TF-IDF (word uni+bigrams, hashed features)
   cosine similarity against all training stories, streamed in chunks.
3. **Longest common span with that neighbour**, exactly, in words.

TinyStories is extremely formulaic, so *some* overlap is expected from any
fluent in-distribution text. The same measurements are therefore applied to
held-out test stories, which the model never saw; generations are judged
against that baseline rather than against zero.
"""

from __future__ import annotations

import difflib
import heapq
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm.auto import tqdm

_BASE = np.uint64(0x100000001B3)  # FNV prime: odd 64-bit multiplier


# --------------------------------------------------------------------------- n-gram hashing


def ngram_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Polynomial rolling hash (mod 2**64) of every length-``n`` window."""
    t = np.asarray(tokens, dtype=np.uint64) + np.uint64(1)
    if len(t) < n:
        return np.empty(0, dtype=np.uint64)
    h = t[: len(t) - n + 1].copy()
    with np.errstate(over="ignore"):
        for k in range(1, n):
            h = h * _BASE + t[k : len(t) - n + 1 + k]
    return h


def scan_corpus_for_ngrams(
    stream: np.ndarray,
    queries: Sequence[np.ndarray],
    ns: Sequence[int] = (8, 16, 32),
    chunk_tokens: int = 16_000_000,
    progress: bool = True,
) -> dict[int, list[np.ndarray]]:
    """For each query and each n, a boolean array: does n-gram i occur in ``stream``?

    One pass over the corpus computes all requested n incrementally; chunks
    overlap by ``max(ns) - 1`` tokens so no window is missed at a boundary.
    """
    ns = sorted(set(ns))
    max_n = ns[-1]
    q_hashes = {n: [ngram_hashes(q, n) for q in queries] for n in ns}
    targets = {
        n: np.unique(np.concatenate(q_hashes[n])) if q_hashes[n] else np.empty(0, np.uint64)
        for n in ns
    }
    found: dict[int, list[np.ndarray]] = {n: [] for n in ns}
    starts = range(0, len(stream), chunk_tokens)
    for start in tqdm(starts, desc="scanning training n-grams", disable=not progress):
        chunk = np.asarray(
            stream[start : start + chunk_tokens + max_n - 1], dtype=np.uint64
        ) + np.uint64(1)
        h = chunk.copy()
        with np.errstate(over="ignore"):
            for k in range(1, max_n):
                h = h[:-1] * _BASE + chunk[k:]
                n = k + 1
                if n in found and len(targets[n]):
                    pos = np.searchsorted(targets[n], h).clip(max=len(targets[n]) - 1)
                    hit = targets[n][pos] == h
                    if hit.any():
                        found[n].append(np.unique(h[hit]))
            if 1 in found and len(targets[1]):  # n == 1 is not covered by the loop
                pos = np.searchsorted(targets[1], chunk).clip(max=len(targets[1]) - 1)
                found[1].append(np.unique(chunk[targets[1][pos] == chunk]))
    out: dict[int, list[np.ndarray]] = {}
    for n in ns:
        hits = np.unique(np.concatenate(found[n])) if found[n] else np.empty(0, np.uint64)
        # One vectorized lookup for all queries: np.isin per query would re-sort
        # `hits` every time (hours for ~30k documents instead of seconds).
        lengths = [len(qh) for qh in q_hashes[n]]
        flat = np.concatenate(q_hashes[n]) if lengths else np.empty(0, np.uint64)
        if len(hits):
            pos = np.searchsorted(hits, flat).clip(max=len(hits) - 1)
            member = hits[pos] == flat
        else:
            member = np.zeros(len(flat), dtype=bool)
        out[n] = np.split(member, np.cumsum(lengths)[:-1]) if lengths else []
    return out


# --------------------------------------------------------------------------- retrieval


class TfidfRetriever:
    """Streaming nearest-neighbour search over a large corpus with bounded memory."""

    def __init__(self, idf_sample: Sequence[str], n_features: int = 2**20):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

        self.vectorizer = HashingVectorizer(
            ngram_range=(1, 2), n_features=n_features, alternate_sign=False, norm=None
        )
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(self.vectorizer.transform(idf_sample))

    def embed(self, texts: Sequence[str]):
        return self.tfidf.transform(self.vectorizer.transform(texts))  # rows are L2-normalized

    def search(
        self,
        queries: Sequence[str],
        corpus_batches: Iterable[tuple[int, Sequence[str]]],
        top_k: int = 1,
        progress: bool = True,
    ) -> list[list[tuple[float, int]]]:
        """Return, per query, the ``top_k`` (cosine, corpus_index) pairs, best first."""
        q = self.embed(queries)
        heaps: list[list[tuple[float, int]]] = [[] for _ in queries]
        for offset, texts in tqdm(corpus_batches, desc="nearest neighbours", disable=not progress):
            sims = (q @ self.embed(texts).T).toarray()
            k = min(top_k, sims.shape[1])
            best = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            for qi, cols in enumerate(best):
                for c in cols:
                    item = (float(sims[qi, c]), offset + int(c))
                    if len(heaps[qi]) < top_k:
                        heapq.heappush(heaps[qi], item)
                    else:
                        heapq.heappushpop(heaps[qi], item)
        return [sorted(h, reverse=True) for h in heaps]


def longest_common_span(a: str, b: str) -> tuple[int, str]:
    """Longest run of consecutive identical words shared by ``a`` and ``b``."""
    wa, wb = a.split(), b.split()
    m = difflib.SequenceMatcher(None, wa, wb, autojunk=False).find_longest_match(
        0, len(wa), 0, len(wb)
    )
    return m.size, " ".join(wa[m.a : m.a + m.size])


# --------------------------------------------------------------------------- report


@dataclass
class MemorizationThresholds:
    """When a text is surfaced for manual review (documented in results/README)."""

    similarity: float = 0.8  # TF-IDF cosine to the nearest training story
    verbatim_ngram: int = 32  # any verbatim token n-gram of this length
    common_span_words: int = 30  # exact shared word span with the neighbour


@dataclass
class MemorizationRecord:
    source: str  # "generation" or "heldout_test"
    item_id: str
    text: str
    tokens: int
    ngram_overlap: dict[int, float]
    max_matched_ngram: int | None
    nearest_training_index: int
    nearest_training_example: str
    similarity: float
    longest_common_span_words: int
    longest_common_span: str
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["ngram_overlap"] = {str(k): v for k, v in self.ngram_overlap.items()}
        return d


def analyze(
    items: Sequence[dict[str, Any]],
    train_stream: np.ndarray,
    train_texts: Sequence[str],
    ns: Sequence[int] = (8, 16, 32),
    thresholds: MemorizationThresholds | None = None,
    idf_sample_size: int = 100_000,
    batch_size: int = 50_000,
    seed: int = 0,
) -> list[MemorizationRecord]:
    """Analyze texts. Each item needs ``source``, ``item_id``, ``text`` and ``token_ids``."""
    thresholds = thresholds or MemorizationThresholds()
    queries = [np.asarray(it["token_ids"], dtype=np.int64) for it in items]
    membership = scan_corpus_for_ngrams(train_stream, queries, ns)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(
        len(train_texts), size=min(idf_sample_size, len(train_texts)), replace=False
    )
    retriever = TfidfRetriever([train_texts[int(i)] for i in sample_idx])
    batches = ((s, train_texts[s : s + batch_size]) for s in range(0, len(train_texts), batch_size))
    neighbours = retriever.search([it["text"] for it in items], batches, top_k=1)

    records = []
    for i, it in enumerate(items):
        overlap = {n: float(membership[n][i].mean()) if len(membership[n][i]) else 0.0 for n in ns}
        matched = [n for n in ns if len(membership[n][i]) and membership[n][i].any()]
        sim, idx = neighbours[i][0]
        neighbour = train_texts[idx]
        span_len, span = longest_common_span(it["text"], neighbour)
        rec = MemorizationRecord(
            source=it["source"],
            item_id=it["item_id"],
            text=it["text"],
            tokens=len(queries[i]),
            ngram_overlap=overlap,
            max_matched_ngram=max(matched) if matched else None,
            nearest_training_index=idx,
            nearest_training_example=neighbour,
            similarity=sim,
            longest_common_span_words=span_len,
            longest_common_span=span,
        )
        if sim >= thresholds.similarity:
            rec.flag_reasons.append(f"similarity {sim:.2f} >= {thresholds.similarity}")
        if any(n >= thresholds.verbatim_ngram for n in matched):
            rec.flag_reasons.append(f"verbatim {max(matched)}-token n-gram in training data")
        if span_len >= thresholds.common_span_words:
            rec.flag_reasons.append(f"{span_len}-word span shared with nearest story")
        rec.flagged = bool(rec.flag_reasons)
        records.append(rec)
    return records


def summarize(records: Sequence[MemorizationRecord]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in sorted({r.source for r in records}):
        rs = [r for r in records if r.source == source]
        sims = np.array([r.similarity for r in rs])
        out[source] = {
            "count": len(rs),
            "similarity_median": float(np.median(sims)),
            "similarity_p95": float(np.percentile(sims, 95)),
            "similarity_max": float(sims.max()),
            "ngram_overlap_mean": {
                str(n): float(np.mean([r.ngram_overlap[n] for r in rs]))
                for n in rs[0].ngram_overlap
            },
            "share_with_any_verbatim_ngram": {
                str(n): float(np.mean([(r.max_matched_ngram or 0) >= n for r in rs]))
                for n in rs[0].ngram_overlap
            },
            "longest_common_span_words_median": float(
                np.median([r.longest_common_span_words for r in rs])
            ),
            "longest_common_span_words_max": int(max(r.longest_common_span_words for r in rs)),
            "flagged": int(sum(r.flagged for r in rs)),
        }
    return out
