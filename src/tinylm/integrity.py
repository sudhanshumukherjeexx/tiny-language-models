"""Dataset integrity: duplication inside splits and leakage between splits.

* **Exact duplicates** — 64-bit hash of :func:`tinylm.data.dedup_key`
  (NFKC, case-folded, whitespace-collapsed text).
* **Cross-split containment** — for every validation/test story, the fraction
  of its 13-token n-grams that occur anywhere in the packed training stream
  (the n-gram-overlap contamination test of Brown et al., 2020, applied with
  BPE tokens). Computed against the *entire* training set.
* **Near duplicates** — MinHash (5-token shingles) with LSH banding, verified
  by estimated Jaccard similarity. Within-split near-duplicate search is run
  on a random sample of each split to bound cost; the sample size is recorded.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from tinylm.data import content_hash
from tinylm.memorization import ngram_hashes, scan_corpus_for_ngrams

# --------------------------------------------------------------------------- exact


def exact_duplicate_stats(hashes: np.ndarray) -> dict[str, Any]:
    counts = Counter(hashes.tolist())
    dup_docs = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "documents": len(hashes),
        "unique": len(counts),
        "redundant_copies": dup_docs,  # documents beyond the first copy
        "redundant_copy_rate": dup_docs / len(hashes) if len(hashes) else 0.0,
        "largest_cluster": max(counts.values()) if counts else 0,
    }


def cross_split_exact(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """How many documents of ``b`` have an exact (normalized) copy in ``a``."""
    in_a = np.isin(b, a)
    return {
        "documents": len(b),
        "with_copy_in_reference": int(in_a.sum()),
        "rate": float(in_a.mean()) if len(b) else 0.0,
    }


def hash_texts(texts: Sequence[str]) -> np.ndarray:
    return np.fromiter(
        (content_hash(t) for t in tqdm(texts, desc="hashing", leave=False)),
        dtype=np.uint64,
        count=len(texts),
    )


# --------------------------------------------------------------------------- containment


def split_documents(stream: np.ndarray, eos_id: int) -> list[np.ndarray]:
    """Split a packed token stream back into documents at EOS tokens."""
    stream = np.asarray(stream)
    ends = np.flatnonzero(stream == eos_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [stream[s:e] for s, e in zip(starts, ends, strict=True) if e > s]


def document_ends(stream: np.ndarray, eos_id: int, chunk: int = 50_000_000) -> np.ndarray:
    """Positions of every EOS token, found chunk by chunk (bounded memory on memmaps)."""
    parts = [
        np.flatnonzero(np.asarray(stream[s : s + chunk]) == eos_id) + s
        for s in range(0, len(stream), chunk)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def sample_documents(
    stream: np.ndarray, eos_id: int, k: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Uniformly sample ``k`` documents from anywhere in a packed stream."""
    ends = document_ends(stream, eos_id)
    starts = np.concatenate([[0], ends[:-1] + 1])
    pick = np.sort(rng.choice(len(ends), size=min(k, len(ends)), replace=False))
    return [np.asarray(stream[starts[i] : ends[i]]) for i in pick if ends[i] > starts[i]]


def containment(train_stream: np.ndarray, docs: Sequence[np.ndarray], n: int = 13) -> np.ndarray:
    """Per document: fraction of its n-grams found anywhere in ``train_stream``."""
    membership = scan_corpus_for_ngrams(train_stream, docs, ns=(n,))[n]
    return np.array([m.mean() if len(m) else 0.0 for m in membership])


def containment_summary(values: np.ndarray, thresholds=(0.5, 0.8, 1.0)) -> dict[str, Any]:
    return {
        "documents": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        **{f"share_ge_{t}": float((values >= t).mean()) for t in thresholds},
    }


# --------------------------------------------------------------------------- MinHash / LSH


@dataclass
class MinHashLSH:
    num_perm: int = 128
    bands: int = 16  # rows per band = num_perm / bands = 8 -> threshold ~ (1/16)^(1/8) = 0.71
    shingle: int = 5
    threshold: float = 0.8
    seed: int = 0

    def __post_init__(self):
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(self.seed)
        self._a = rng.integers(1, 2**63, self.num_perm, dtype=np.uint64) | np.uint64(1)  # odd
        self._b = rng.integers(0, 2**63, self.num_perm, dtype=np.uint64)

    def signatures(self, docs: Sequence[np.ndarray]) -> np.ndarray:
        """(num_docs, num_perm) MinHash signatures; docs shorter than a shingle get all-ones."""
        sig = np.full((len(docs), self.num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for i, doc in enumerate(tqdm(docs, desc="minhash", leave=False)):
                sh = np.unique(ngram_hashes(doc, self.shingle))
                if len(sh):
                    sig[i] = (sh[None, :] * self._a[:, None] + self._b[:, None]).min(axis=1)
        return sig

    def candidate_pairs(self, sig: np.ndarray) -> set[tuple[int, int]]:
        rows = self.num_perm // self.bands
        pairs: set[tuple[int, int]] = set()
        for band in range(self.bands):
            block = np.ascontiguousarray(sig[:, band * rows : (band + 1) * rows])
            keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
            _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
            order = np.argsort(inverse, kind="stable")  # documents grouped by bucket
            bounds = np.concatenate([[0], np.cumsum(counts)])
            for bucket in np.flatnonzero(counts > 1):
                # Cap very large buckets (templated boilerplate) to bound pair count.
                members = order[bounds[bucket] : bounds[bucket + 1]][:50]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        pairs.add((int(members[i]), int(members[j])))
        return pairs

    def near_duplicates(self, docs: Sequence[np.ndarray]) -> dict[str, Any]:
        sig = self.signatures(docs)
        verified = [
            (i, j, float((sig[i] == sig[j]).mean()))
            for i, j in self.candidate_pairs(sig)
            if (sig[i] == sig[j]).mean() >= self.threshold
        ]
        involved = {i for i, _, _ in verified} | {j for _, j, _ in verified}
        return {
            "documents": len(docs),
            "verified_pairs": len(verified),
            "documents_with_near_duplicate": len(involved),
            "rate": len(involved) / len(docs) if docs else 0.0,
            "params": {
                "num_perm": self.num_perm,
                "bands": self.bands,
                "shingle_tokens": self.shingle,
                "jaccard_threshold": self.threshold,
            },
            "examples": sorted(verified, key=lambda x: -x[2])[:20],
        }

    def cross_near_duplicates(
        self, reference: Sequence[np.ndarray], queries: Sequence[np.ndarray]
    ) -> dict[str, Any]:
        """Share of ``queries`` with a near duplicate in ``reference`` (via shared LSH bands)."""
        ref_sig, q_sig = self.signatures(reference), self.signatures(queries)
        rows = self.num_perm // self.bands
        matched = np.zeros(len(queries), dtype=bool)
        for band in range(self.bands):
            sl = slice(band * rows, (band + 1) * rows)
            r = np.ascontiguousarray(ref_sig[:, sl]).view(np.dtype((np.void, 8 * rows))).ravel()
            q = np.ascontiguousarray(q_sig[:, sl]).view(np.dtype((np.void, 8 * rows))).ravel()
            ref_index: dict[bytes, int] = {}
            for i, key in enumerate(r):
                ref_index.setdefault(key.tobytes(), i)
            for qi, key in enumerate(q):
                if matched[qi]:
                    continue
                ri = ref_index.get(key.tobytes())
                if ri is not None and (ref_sig[ri] == q_sig[qi]).mean() >= self.threshold:
                    matched[qi] = True
        return {
            "queries": len(queries),
            "reference": len(reference),
            "with_near_duplicate": int(matched.sum()),
            "rate": float(matched.mean()) if len(queries) else 0.0,
        }
