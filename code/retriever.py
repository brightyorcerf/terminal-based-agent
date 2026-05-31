"""
retriever.py — Corpus manifest + BM25 retrieval index.

Builds once at startup. All retrieval is deterministic:
  - Files are sorted before indexing → stable chunk IDs
  - BM25Okapi is deterministic given same input
  - No network calls, no embeddings API
"""

import os
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

from config import (
    REPO_ROOT, DATA_ROOT,
    BM25_TOP_K, BM25_CHUNK_SIZE, BM25_CHUNK_OVERLAP, BM25_MIN_CHUNK_LEN,
)


# ── Tokenizer ────────────────────────────────────────────────────────────────

# High-frequency words that add noise without adding signal.
# BM25's IDF naturally down-weights them, but removing them sharpens precision
# for specific product/technical terms (the main retrieval signal here).
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "i", "my", "me", "we", "our", "you",
    "your", "he", "she", "it", "they", "their", "this", "that", "these",
    "those", "not", "no", "so", "if", "as", "up", "out", "about", "can",
    "get", "got", "just", "also", "what", "how", "when", "where", "which",
    "who", "whom", "all", "any", "some", "more", "than", "then", "there",
    "here", "now", "its",
})


def tokenize(text: str) -> list[str]:
    """
    Deterministic tokenizer. Lowercase, alphanumeric tokens only, stop words removed.
    No stemming (stemming libraries can be non-deterministic across versions).
    """
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOP_WORDS
    ]


# ── Corpus manifest ──────────────────────────────────────────────────────────

def load_corpus_manifest() -> frozenset[str]:
    """
    Walk data/ and return a frozenset of all real file paths,
    relative to REPO_ROOT, forward-slash normalised.

    This is THE source of truth for citation validation.
    Any LLM-generated path not in this set is hallucinated.
    """
    paths: set[str] = set()
    for root, dirs, files in os.walk(DATA_ROOT):
        # Skip hidden dirs and __pycache__ in-place (modifies dirs list)
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d != "__pycache__"
        ]
        for fname in files:
            if fname.startswith("."):
                continue
            abs_path = Path(root) / fname
            rel = abs_path.relative_to(REPO_ROOT)
            paths.add(str(rel).replace("\\", "/"))
    return frozenset(paths)


# ── BM25 index ───────────────────────────────────────────────────────────────

def build_bm25_index(
    manifest: frozenset[str],
) -> tuple["BM25Okapi", list[dict]]:
    """
    Chunks every corpus file into overlapping windows and builds a BM25Okapi
    index over them.

    Returns:
        (index, doc_records)

    doc_records[i] is a dict:
        path   — str, relative to REPO_ROOT  (safe to use as citation)
        text   — str, raw chunk text
        tokens — list[str], tokenised chunk

    DETERMINISM: manifest is sorted before iteration so chunk indices are
    identical across runs regardless of OS filesystem ordering.
    """
    records: list[dict] = []

    for rel_path in sorted(manifest):
        abs_path = REPO_ROOT / rel_path
        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        start = 0
        while start < len(text):
            chunk = text[start : start + BM25_CHUNK_SIZE]
            if len(chunk.strip()) >= BM25_MIN_CHUNK_LEN:
                records.append(
                    {
                        "path": rel_path,
                        "text": chunk,
                        "tokens": tokenize(chunk),
                    }
                )
            start += BM25_CHUNK_SIZE - BM25_CHUNK_OVERLAP

    corpus_tokens = [r["tokens"] for r in records]
    index = BM25Okapi(corpus_tokens)
    return index, records


# ── Public retrieve function ─────────────────────────────────────────────────

class Retriever:
    """
    Thin wrapper that holds the pre-built index and exposes a single
    `retrieve(query, top_k)` method.
    """

    def __init__(self) -> None:
        print("  Loading corpus manifest …", flush=True)
        self.manifest: frozenset[str] = load_corpus_manifest()
        print(f"  {len(self.manifest)} corpus files found.", flush=True)

        print("  Building BM25 index …", flush=True)
        self.index, self.records = build_bm25_index(self.manifest)
        print(f"  {len(self.records)} chunks indexed.", flush=True)

    def retrieve(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
    ) -> tuple[list[dict], float]:
        """
        Returns (chunks, best_score).

        chunks  — list of doc_record dicts, deduplicated by path,
                  sorted by score descending, length ≤ top_k
        best_score — raw BM25 score of the top chunk; used downstream
                     as a confidence signal.
        """
        tokens = tokenize(query)
        if not tokens:
            return [], 0.0

        scores = self.index.get_scores(tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k * 2]  # fetch extra before dedup

        best_score = float(scores[top_indices[0]]) if top_indices else 0.0

        # Deduplicate: keep highest-scoring chunk per file path
        seen: set[str] = set()
        chunks: list[dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            rec = self.records[idx]
            if rec["path"] not in seen:
                seen.add(rec["path"])
                chunks.append(rec)
            if len(chunks) >= top_k:
                break

        return chunks, best_score


