"""
retrieval.py
============
Dense retrieval index construction and search using FAISS (HNSW / Flat)
and optional Qdrant integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

CHECKPOINT_DIR = Path(".embedding_checkpoints")


def _corpus_checkpoint_key(corpus: dict, embedder) -> str:
    doc_ids = sorted(corpus.keys())
    n = len(doc_ids)
    first = doc_ids[0] if doc_ids else ""
    last = doc_ids[-1] if doc_ids else ""
    raw = f"{embedder.model_id}|{n}|{first}|{last}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a retrieval run."""

    query_id: str
    retrieved_ids: list[str]
    scores: list[float]
    latency_ms: float


# ---------------------------------------------------------------------------
# FAISS-based retriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Dense retriever backed by FAISS.
    Uses HNSW for large corpora, flat L2/IP for small ones.
    """

    EXACT_THRESHOLD = 100_000

    def __init__(
        self,
        dimension: int,
        use_hnsw: bool = True,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 100,
        use_gpu: bool = False,
    ):
        import faiss

        # Prevent the macOS segfault caused by two OpenMP runtimes (numpy MKL +
        # FAISS LLVM) racing during parallel search.
        faiss.omp_set_num_threads(1)

        self.dimension = dimension
        self.use_hnsw = use_hnsw
        self._id_map: list[str] = []

        if use_hnsw:
            # Inner-product HNSW (cosine with L2-normalised vectors)
            self._index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efConstruction = ef_construction
            self._index.hnsw.efSearch = ef_search
        else:
            self._index = faiss.IndexFlatIP(dimension)

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"GPU FAISS failed: {e}. Using CPU.")

    def add(self, doc_ids: list[str], vectors: np.ndarray) -> None:
        """Add document embeddings to the index."""
        assert vectors.shape[1] == self.dimension, \
            f"Dimension mismatch: {vectors.shape[1]} != {self.dimension}"
        vecs_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index.add(vecs_f32)
        self._id_map.extend(doc_ids)
        logger.info(f"Indexed {len(doc_ids)} vectors (total: {len(self._id_map)})")

    def search(
        self,
        query_ids: list[str],
        query_vectors: np.ndarray,
        top_k: int = 100,
    ) -> list[RetrievalResult]:
        """Search the index for each query vector."""
        vecs_f32 = np.ascontiguousarray(query_vectors, dtype=np.float32)
        t0 = time.perf_counter()
        scores_batch, indices_batch = self._index.search(vecs_f32, top_k)
        elapsed = (time.perf_counter() - t0) * 1000  # ms total

        per_query_ms = elapsed / max(len(query_ids), 1)
        results: list[RetrievalResult] = []

        for i, qid in enumerate(query_ids):
            idxs = indices_batch[i]
            scrs = scores_batch[i]
            retrieved_ids = [
                self._id_map[idx] for idx in idxs if idx >= 0 and idx < len(self._id_map)
            ]
            scores = [float(s) for s, idx in zip(scrs, idxs) if idx >= 0]
            results.append(
                RetrievalResult(
                    query_id=qid,
                    retrieved_ids=retrieved_ids,
                    scores=scores,
                    latency_ms=per_query_ms,
                )
            )

        return results

    def save(self, path: str) -> None:
        import faiss
        faiss.write_index(self._index, path)
        import json
        with open(path + ".ids.json", "w") as f:
            json.dump(self._id_map, f)

    @classmethod
    def load(cls, path: str, dimension: int) -> "FAISSRetriever":
        import faiss, json
        faiss.omp_set_num_threads(1)
        obj = cls.__new__(cls)
        obj.dimension = dimension
        obj._index = faiss.read_index(path)
        with open(path + ".ids.json") as f:
            obj._id_map = json.load(f)
        return obj


# ---------------------------------------------------------------------------
# High-level retrieval pipeline
# ---------------------------------------------------------------------------

def build_index(
    corpus: dict[str, str],
    embedder,
    index_path: Optional[str] = None,
    batch_size: int = 512,
    use_hnsw: bool = True,
) -> FAISSRetriever:
    """
    Embed all corpus documents and build a FAISS index.

    Args:
        corpus: dict mapping doc_id -> text.
        embedder: BaseEmbedder instance.
        index_path: Optional path to cache the index.
        batch_size: Embedding batch size.
        use_hnsw: Whether to use HNSW (True) or flat IP (False).

    Returns:
        Populated FAISSRetriever.
    """
    if index_path and Path(index_path).exists() and Path(index_path + ".ids.json").exists():
        logger.info(f"Loading cached FAISS index from {index_path}")
        return FAISSRetriever.load(index_path, embedder.dimension)

    use_hnsw_flag = use_hnsw and len(corpus) >= FAISSRetriever.EXACT_THRESHOLD
    retriever = FAISSRetriever(embedder.dimension, use_hnsw=use_hnsw_flag)

    doc_ids = list(corpus.keys())
    texts = list(corpus.values())
    n = len(texts)

    # --- checkpoint setup ---
    ckpt_key = _corpus_checkpoint_key(corpus, embedder)
    ckpt_dir = CHECKPOINT_DIR / ckpt_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    emb_path = ckpt_dir / "embeddings.npy"
    progress_path = ckpt_dir / "progress.json"

    done_count = 0
    if emb_path.exists() and progress_path.exists():
        try:
            with open(progress_path) as f:
                saved = json.load(f)
            if saved.get("total") == n and 0 < saved.get("done_count", 0) <= n:
                done_count = saved["done_count"]
                logger.info(f"Resuming embedding: {done_count}/{n} already computed")
        except Exception:
            done_count = 0

    mode = "r+" if done_count > 0 else "w+"
    try:
        all_embs = np.memmap(
            str(emb_path), dtype=np.float32, mode=mode,
            shape=(n, embedder.dimension),
        )
    except Exception:
        # Shape mismatch or corrupt file — start fresh
        done_count = 0
        all_embs = np.memmap(
            str(emb_path), dtype=np.float32, mode="w+",
            shape=(n, embedder.dimension),
        )

    logger.info(f"Embedding {n} documents (batch_size={batch_size})")
    from tqdm.auto import tqdm

    all_batch_starts = list(range(0, n, batch_size))
    done_batches = sum(1 for i in all_batch_starts if i < done_count)
    remaining_starts = [i for i in all_batch_starts if i >= done_count]

    for i in tqdm(remaining_starts, desc="Embedding corpus",
                  initial=done_batches, total=len(all_batch_starts)):
        batch_texts = texts[i : i + batch_size]
        # FIX: pass is_query=False for documents
        vecs = embedder.embed_with_cache(batch_texts, normalize=True, is_query=False)
        end = i + len(batch_texts)
        all_embs[i:end] = vecs
        all_embs.flush()
        with open(progress_path, "w") as f:
            json.dump({"done_count": end, "total": n}, f)

    corpus_matrix = np.array(all_embs, dtype=np.float32)
    retriever.add(doc_ids, corpus_matrix)

    if index_path:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        retriever.save(index_path)
        logger.info(f"Saved FAISS index to {index_path}")

    # clean up checkpoint after successful completion
    try:
        emb_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        ckpt_dir.rmdir()
    except Exception:
        pass

    return retriever


def run_retrieval(
    queries: dict[str, str],
    embedder,
    retriever: FAISSRetriever,
    top_k: int = 100,
    batch_size: int = 256,
) -> dict[str, list[str]]:
    """
    Embed all queries and retrieve top-k documents.

    Returns:
        dict mapping query_id -> ranked list of doc_ids.
    """
    from tqdm.auto import tqdm

    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    run: dict[str, list[str]] = {}
    latencies: list[float] = []

    for i in tqdm(range(0, len(query_ids), batch_size), desc="Retrieving"):
        batch_qids = query_ids[i : i + batch_size]
        batch_texts = query_texts[i : i + batch_size]
        # FIX: pass is_query=True for queries
        vecs = embedder.embed_with_cache(batch_texts, normalize=True, is_query=True)
        results = retriever.search(batch_qids, vecs, top_k=top_k)
        for res in results:
            run[res.query_id] = res.retrieved_ids
            latencies.append(res.latency_ms)

    logger.info(
        f"Retrieval complete. Mean latency: {np.mean(latencies):.2f} ms/query"
    )
    return run


def measure_retrieval_latency(
    sample_queries: dict[str, str],
    embedder,
    retriever: FAISSRetriever,
    n_warmup: int = 5,
    n_measure: int = 50,
) -> dict[str, float]:
    """Measure end-to-end query latency (embed + search) in ms."""
    import time

    qids = list(sample_queries.keys())[:1]
    qtexts = list(sample_queries.values())[:1]

    latencies: list[float] = []
    for trial in range(n_warmup + n_measure):
        t0 = time.perf_counter()
        # FIX: pass is_query=True for latency measurement
        vecs = embedder.embed(qtexts, normalize=True, is_query=True)
        retriever.search(qids, vecs, top_k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if trial >= n_warmup:
            latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }