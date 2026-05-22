"""
chunking.py
===========
Chunking strategies for RAG document preprocessing:
  - Fixed (non-overlapping)
  - Sliding window (overlapping)
  - Semantic (cosine-similarity boundary detection)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text chunk with provenance metadata."""

    chunk_id: str
    doc_id: str
    text: str
    start_token: int
    end_token: int
    strategy: str
    chunk_size: int


def _simple_tokenise(text: str) -> list[str]:
    """Whitespace tokeniser used for chunk boundary estimation."""
    return text.split()


def _detokenise(tokens: list[str]) -> str:
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Fixed chunking
# ---------------------------------------------------------------------------

def fixed_chunks(
    doc_id: str,
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk]:
    """
    Split text into non-overlapping (or overlapping) fixed-size chunks.

    Args:
        doc_id: Parent document identifier.
        text: Raw text to chunk.
        chunk_size: Target chunk size in whitespace tokens.
        overlap: Number of tokens to overlap between consecutive chunks.
    """
    tokens = _simple_tokenise(text)
    n = len(tokens)
    stride = max(1, chunk_size - overlap)

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = _detokenise(tokens[start:end])
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                doc_id=doc_id,
                text=chunk_text,
                start_token=start,
                end_token=end,
                strategy="fixed" if overlap == 0 else "sliding_window",
                chunk_size=chunk_size,
            )
        )
        start += stride
        chunk_idx += 1

    return chunks


def sliding_window_chunks(
    doc_id: str,
    text: str,
    chunk_size: int,
    overlap_ratio: float = 0.5,
) -> list[Chunk]:
    """Sliding-window chunking with fractional overlap."""
    overlap = int(chunk_size * overlap_ratio)
    return fixed_chunks(doc_id, text, chunk_size, overlap=overlap)


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitter (period/question mark/exclamation mark)."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def semantic_chunks(
    doc_id: str,
    text: str,
    chunk_size: int,
    similarity_threshold: float = 0.75,
    min_chunk_tokens: int = 64,
    max_chunk_multiplier: float = 2.0,
    embedder=None,
) -> list[Chunk]:
    """
    Segment a document by detecting cosine-similarity drops between
    consecutive sentence embeddings (TextTiling-inspired).

    If no embedder is provided, falls back to fixed chunking.
    """
    if embedder is None:
        logger.debug("No embedder provided for semantic chunking; falling back to fixed.")
        return fixed_chunks(doc_id, text, chunk_size)

    sentences = _sentence_split(text)
    if len(sentences) <= 1:
        return fixed_chunks(doc_id, text, chunk_size)

    # Embed sentences
    vecs = embedder.embed_with_cache(sentences, normalize=True)

    # Compute cosine similarity between adjacent sentences
    sims: list[float] = []
    for i in range(len(vecs) - 1):
        cos = float(np.dot(vecs[i], vecs[i + 1]))
        sims.append(cos)

    # Detect boundary: similarity drop below threshold
    boundaries = [0]
    for i, sim in enumerate(sims):
        if sim < similarity_threshold:
            boundaries.append(i + 1)
    boundaries.append(len(sentences))

    # Build segments, merging very small ones
    max_tokens = int(chunk_size * max_chunk_multiplier)
    chunks: list[Chunk] = []
    chunk_idx = 0
    seg_start = boundaries[0]
    token_cursor = 0

    for b_idx in range(1, len(boundaries)):
        seg_end = boundaries[b_idx]
        segment_text = " ".join(sentences[seg_start:seg_end])
        seg_tokens = len(_simple_tokenise(segment_text))

        if seg_tokens < min_chunk_tokens and b_idx < len(boundaries) - 1:
            # Merge with next segment
            continue

        if seg_tokens > max_tokens:
            # Fall back to fixed chunking for oversized segments
            sub_chunks = fixed_chunks(
                doc_id, segment_text, chunk_size, overlap=0
            )
            for sc in sub_chunks:
                sc.chunk_id = f"{doc_id}_s{chunk_idx}"
                sc.strategy = "semantic"
                chunks.append(sc)
                chunk_idx += 1
        else:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_s{chunk_idx}",
                    doc_id=doc_id,
                    text=segment_text,
                    start_token=token_cursor,
                    end_token=token_cursor + seg_tokens,
                    strategy="semantic",
                    chunk_size=chunk_size,
                )
            )
            chunk_idx += 1

        token_cursor += seg_tokens
        seg_start = seg_end

    return chunks if chunks else fixed_chunks(doc_id, text, chunk_size)


# ---------------------------------------------------------------------------
# Corpus-level chunker
# ---------------------------------------------------------------------------

def chunk_corpus(
    corpus: dict[str, str],
    strategy: str,
    chunk_size: int,
    overlap_ratio: float = 0.5,
    similarity_threshold: float = 0.75,
    embedder=None,
) -> tuple[list[Chunk], dict[str, str]]:
    """
    Apply a chunking strategy to an entire corpus.

    Returns:
        chunks: List of all Chunk objects.
        chunk_corpus: Dict mapping chunk_id -> chunk text (for embedding).
    """
    all_chunks: list[Chunk] = []

    for doc_id, text in corpus.items():
        if strategy == "fixed":
            doc_chunks = fixed_chunks(doc_id, text, chunk_size)
        elif strategy == "sliding_window":
            doc_chunks = sliding_window_chunks(doc_id, text, chunk_size, overlap_ratio)
        elif strategy == "semantic":
            doc_chunks = semantic_chunks(
                doc_id, text, chunk_size,
                similarity_threshold=similarity_threshold,
                embedder=embedder,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_chunks.extend(doc_chunks)

    chunk_texts = {c.chunk_id: c.text for c in all_chunks}
    return all_chunks, chunk_texts


def propagate_qrels_to_chunks(
    qrels: dict[str, dict[str, int]],
    chunks: list[Chunk],
) -> dict[str, dict[str, int]]:
    """
    Propagate document-level relevance labels to chunk level.
    A chunk inherits its parent document's relevance label.
    """
    doc_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        doc_to_chunks.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    chunk_qrels: dict[str, dict[str, int]] = {}
    for qid, doc_rels in qrels.items():
        chunk_rels: dict[str, int] = {}
        for doc_id, score in doc_rels.items():
            for cid in doc_to_chunks.get(doc_id, []):
                chunk_rels[cid] = score
        if chunk_rels:
            chunk_qrels[qid] = chunk_rels

    return chunk_qrels
