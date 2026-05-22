"""
rag_eval.py
===========
Minimal RAG retrieval evaluation pipeline.
Focuses exclusively on retrieval quality; no generation step.

Pipeline:
  1. Chunk documents
  2. Embed chunks
  3. Build FAISS index
  4. Embed queries
  5. Retrieve top-k chunks
  6. Evaluate retrieval quality (propagated qrels)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from chunking import chunk_corpus, propagate_qrels_to_chunks
from data_loaders import RetrievalDataset
from metrics import evaluate_run, storage_footprint_mb, throughput_passages_per_sec
from retrieval import build_index, run_retrieval

logger = logging.getLogger(__name__)


@dataclass
class RAGEvalConfig:
    """Configuration for a single RAG retrieval experiment."""

    model_name: str
    strategy: str             # "fixed", "sliding_window", "semantic"
    chunk_size: int           # target token count per chunk
    overlap_ratio: float = 0.5
    similarity_threshold: float = 0.75
    top_k: int = 10
    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    index_dir: str = "results/indices"


@dataclass
class RAGEvalResult:
    """Metrics from a RAG retrieval evaluation run."""

    model_name: str
    strategy: str
    chunk_size: int
    metrics: dict[str, float]
    n_chunks: int
    storage_mb: float
    throughput_pps: float


def run_rag_eval(
    dataset: RetrievalDataset,
    embedder,
    config: RAGEvalConfig,
) -> RAGEvalResult:
    """
    Execute the full RAG retrieval evaluation pipeline.

    Args:
        dataset: RetrievalDataset with corpus, queries, qrels.
        embedder: BaseEmbedder instance.
        config: Experiment configuration.

    Returns:
        RAGEvalResult with metrics and efficiency stats.
    """
    logger.info(
        f"[{config.model_name}] RAG eval | "
        f"strategy={config.strategy} chunk_size={config.chunk_size}"
    )

    # Step 1: Chunk corpus
    embedder_for_semantic = embedder if config.strategy == "semantic" else None
    all_chunks, chunk_texts = chunk_corpus(
        corpus=dataset.corpus,
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        overlap_ratio=config.overlap_ratio,
        similarity_threshold=config.similarity_threshold,
        embedder=embedder_for_semantic,
    )
    n_chunks = len(all_chunks)
    logger.info(f"Produced {n_chunks:,} chunks from {len(dataset.corpus):,} docs")

    # Step 2 & 3: Embed chunks and build index
    import time

    index_path = (
        Path(config.index_dir)
        / f"{config.model_name}_{config.strategy}_{config.chunk_size}"
        / "index.faiss"
    )
    
    t_embed_start = time.perf_counter()
    retriever = build_index(
        corpus=chunk_texts,
        embedder=embedder,
        index_path=str(index_path),
        use_hnsw=n_chunks >= 100_000,
    )
    embed_seconds = time.perf_counter() - t_embed_start
    throughput = throughput_passages_per_sec(n_chunks, embed_seconds)
    storage = storage_footprint_mb(n_chunks, embedder.dimension)

    # Step 4 & 5: Retrieve
    chunk_qrels = propagate_qrels_to_chunks(dataset.qrels, all_chunks)
    run = run_retrieval(
        queries=dataset.queries,
        embedder=embedder,
        retriever=retriever,
        top_k=max(config.k_values),
    )

    # Step 6: Evaluate
    metrics = evaluate_run(run, chunk_qrels, k_values=config.k_values)

    return RAGEvalResult(
        model_name=config.model_name,
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        metrics=metrics,
        n_chunks=n_chunks,
        storage_mb=storage,
        throughput_pps=throughput,
    )


def rag_eval_sweep(
    dataset: RetrievalDataset,
    embedder,
    model_name: str,
    strategies: list[str] = ["fixed", "sliding_window", "semantic"],
    chunk_sizes: list[int] = [256, 512, 1024],
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a full sweep over chunking strategies and sizes.

    Returns a DataFrame with all results.
    """
    rows: list[dict] = []

    for strategy in strategies:
        for chunk_size in chunk_sizes:
            cfg = RAGEvalConfig(
                model_name=model_name,
                strategy=strategy,
                chunk_size=chunk_size,
            )
            try:
                result = run_rag_eval(dataset, embedder, cfg)
                row = {
                    "model": model_name,
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "n_chunks": result.n_chunks,
                    "storage_mb": round(result.storage_mb, 2),
                    "throughput_pps": round(result.throughput_pps, 1),
                }
                row.update({k: round(v, 4) for k, v in result.metrics.items()})
                rows.append(row)
                logger.info(
                    f"  nDCG@10={result.metrics.get('nDCG@10', 0):.4f} "
                    f"Recall@10={result.metrics.get('Recall@10', 0):.4f}"
                )
            except Exception as e:
                logger.error(f"Failed {model_name}/{strategy}/{chunk_size}: {e}")

    df = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved RAG eval results to {output_csv}")

    return df
