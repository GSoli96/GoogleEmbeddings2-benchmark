"""
metrics.py
==========
Retrieval evaluation metrics:
  - Recall@k
  - MRR (Mean Reciprocal Rank)
  - nDCG@k (Normalised Discounted Cumulative Gain)
  - Latency statistics
  - Throughput and storage footprint
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """
    Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    Args:
        retrieved: Ranked list of retrieved document IDs.
        relevant: Set of ground-truth relevant document IDs.
        k: Cutoff rank.

    Returns:
        Recall@k in [0, 1].
    """
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(relevant & retrieved_k) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant document.
    Returns 0.0 if no relevant document is found.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """
    DCG@k = Σ (2^r_i - 1) / log2(i + 2)  for i in 0..k-1
    where r_i is the graded relevance of the i-th retrieved document.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def ideal_dcg_at_k(relevance: dict[str, int], k: int) -> float:
    """IDCG@k: DCG of the ideal ranking."""
    sorted_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(sorted_rels):
        idcg += (2**rel - 1) / math.log2(i + 2)
    return idcg


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """
    nDCG@k = DCG@k / IDCG@k

    Returns 0.0 if IDCG@k == 0 (no relevant documents).
    """
    idcg = ideal_dcg_at_k(relevance, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevance, k) / idcg


# ---------------------------------------------------------------------------
# Aggregate over a query set
# ---------------------------------------------------------------------------

def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = [1, 5, 10],
    ndcg_k: int = 10,
) -> dict[str, float]:
    """
    Compute aggregate metrics over all queries.

    Args:
        run: dict mapping query_id -> ranked list of retrieved doc_ids.
        qrels: dict mapping query_id -> {doc_id: relevance_score}.
        k_values: List of cutoffs for Recall@k.
        ndcg_k: Cutoff for nDCG.

    Returns:
        Dictionary of aggregated metric values.
    """
    recall_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n_queries = 0

    for qid, retrieved in run.items():
        if qid not in qrels:
            continue
        relevance = qrels[qid]
        relevant_set = {did for did, score in relevance.items() if score > 0}

        if not relevant_set:
            continue

        n_queries += 1
        mrr_sum += reciprocal_rank(retrieved, relevant_set)
        ndcg_sum += ndcg_at_k(retrieved, relevance, ndcg_k)

        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, relevant_set, k)

    if n_queries == 0:
        return {**{f"Recall@{k}": 0.0 for k in k_values}, "MRR": 0.0, f"nDCG@{ndcg_k}": 0.0}

    results: dict[str, float] = {}
    for k in k_values:
        results[f"Recall@{k}"] = recall_sums[k] / n_queries
    results["MRR"] = mrr_sum / n_queries
    results[f"nDCG@{ndcg_k}"] = ndcg_sum / n_queries
    results["n_queries"] = float(n_queries)

    return results


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def storage_footprint_mb(n_vectors: int, dimension: int, dtype: str = "float32") -> float:
    """Estimated storage in MB for a dense embedding index."""
    bytes_per_element = {"float32": 4, "float16": 2, "int8": 1}[dtype]
    return (n_vectors * dimension * bytes_per_element) / (1024**2)


def throughput_passages_per_sec(
    n_passages: int, total_seconds: float
) -> float:
    """Embedding throughput in passages per second."""
    if total_seconds <= 0:
        return float("inf")
    return n_passages / total_seconds


def cost_estimate_usd(
    n_tokens: int, cost_per_million: float
) -> float:
    """Cost estimate in USD given total tokens and model price."""
    return (n_tokens / 1_000_000) * cost_per_million


def estimate_tokens(texts: list[str], avg_chars_per_token: float = 4.0) -> int:
    """Rough token count estimate using character heuristic."""
    return int(sum(len(t) for t in texts) / avg_chars_per_token)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Return (lower, upper) bootstrap confidence interval for the mean.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - confidence) / 2
    return float(np.quantile(boot_means, alpha)), float(np.quantile(boot_means, 1 - alpha))
