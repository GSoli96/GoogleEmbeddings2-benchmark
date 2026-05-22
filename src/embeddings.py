"""
embeddings.py
=============
Multi-provider embedding client with caching, async API support,
and local HuggingFace model inference.

Supported providers: Google, OpenAI, Cohere, VoyageAI, HuggingFace (local).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import diskcache
import httpx
import numpy as np
import torch
import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)
_disk_cache = diskcache.Cache(str(CACHE_DIR), size_limit=int(10e9))  # 10 GB


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _cache_key(model_id: str, text: str, normalize: bool) -> str:
    h = hashlib.sha256(f"{model_id}||{text}||{normalize}".encode()).hexdigest()
    return h


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder:
    """Abstract base for all embedding providers."""

    model_id: str
    dimension: int
    max_tokens: int
    name: str
    cost_per_million_tokens: float

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        raise NotImplementedError

    def embed_with_cache(
        self, texts: list[str], normalize: bool = True, is_query: bool = False
    ) -> np.ndarray:
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, t in enumerate(texts):
            key = _cache_key(self.model_id, t, normalize)
            # NOTE: cache key does NOT include is_query to keep it simple;
            # if a text appears as both query and doc, the first cached version wins.
            # For BGE-M3 this is fine because prefix is the only difference.
            val = _disk_cache.get(key)
            if val is not None:
                results[i] = np.frombuffer(val, dtype=np.float32)
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            vecs = self.embed(uncached_texts, normalize=normalize, is_query=is_query)
            for local_i, global_i in enumerate(uncached_indices):
                key = _cache_key(self.model_id, uncached_texts[local_i], normalize)
                _disk_cache.set(key, vecs[local_i].astype(np.float32).tobytes())
                results[global_i] = vecs[local_i]

        return np.stack(results)
# ---------------------------------------------------------------------------
# Local HuggingFace models (SentenceTransformers)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """Wraps sentence-transformers for local GPU/CPU inference."""

    def __init__(self, cfg: dict[str, Any]):
        from sentence_transformers import SentenceTransformer

        self.model_id = cfg["model_id"]
        self.name = cfg["name"]
        self.dimension = cfg["dimension"]
        self.max_tokens = cfg["max_tokens"]
        self.cost_per_million_tokens = cfg.get("cost_per_million_tokens", 0.0)
        self.batch_size = cfg.get("batch_size", 64)
        self.query_prefix = cfg.get("query_prefix", "")
        self.passage_prefix = cfg.get("passage_prefix", "")

        device = cfg.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info(f"Loading {self.model_id} on {device}")
        self._model = SentenceTransformer(self.model_id, device=device)
        self._model.max_seq_length = self.max_tokens

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.passage_prefix
        prefixed = [prefix + t for t in texts] if prefix else texts

        vecs = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# OpenAI Embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """Calls the OpenAI embeddings API with async batching."""

    def __init__(self, cfg: dict[str, Any]):
        self.model_id = cfg["model_id"]
        self.name = cfg["name"]
        self.dimension = cfg["dimension"]
        self.max_tokens = cfg["max_tokens"]
        self.cost_per_million_tokens = cfg.get("cost_per_million_tokens", 0.13)
        self.batch_size = cfg.get("batch_size", 100)
        # Chiave letta direttamente dal YAML — nessuna variabile d'ambiente
        self._api_key = cfg.get("api_key", "")
        if not self._api_key or self._api_key.startswith("sk-..."):
            raise ValueError(
                "openai_text_emb_3_large: imposta 'api_key' in configs/models.yaml"
            )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
    )
    def _call_api(self, batch: list[str]) -> list[list[float]]:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"model": self.model_id, "input": batch}
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    def embed(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vecs.extend(self._call_api(batch))
        arr = np.array(all_vecs, dtype=np.float32)
        return _normalise(arr) if normalize else arr


# ---------------------------------------------------------------------------
# Cohere Embedder
# ---------------------------------------------------------------------------

class CohereEmbedder(BaseEmbedder):
    """Calls the Cohere embeddings API."""

    def __init__(self, cfg: dict[str, Any]):
        self.model_id = cfg["model_id"]
        self.name = cfg["name"]
        self.dimension = cfg["dimension"]
        self.max_tokens = cfg["max_tokens"]
        self.cost_per_million_tokens = cfg.get("cost_per_million_tokens", 0.1)
        self.batch_size = cfg.get("batch_size", 96)
        # Chiave letta direttamente dal YAML — nessuna variabile d'ambiente
        self._api_key = cfg.get("api_key", "")
        if not self._api_key or self._api_key == "...":
            raise ValueError(
                "cohere_embed_v3: imposta 'api_key' in configs/models.yaml"
            )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
    )
    def _call_api(self, batch: list[str], input_type: str) -> list[list[float]]:
        url = "https://api.cohere.ai/v1/embed"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "texts": batch,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        return resp.json()["embeddings"]["float"]

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        input_type = "search_query" if is_query else "search_document"
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vecs.extend(self._call_api(batch, input_type))
        arr = np.array(all_vecs, dtype=np.float32)
        return _normalise(arr) if normalize else arr


# ---------------------------------------------------------------------------
# VoyageAI Embedder
# ---------------------------------------------------------------------------

class VoyageEmbedder(BaseEmbedder):
    """Calls the Voyage AI embeddings API."""

    def __init__(self, cfg: dict[str, Any]):
        self.model_id = cfg["model_id"]
        self.name = cfg["name"]
        self.dimension = cfg["dimension"]
        self.max_tokens = cfg["max_tokens"]
        self.cost_per_million_tokens = cfg.get("cost_per_million_tokens", 0.06)
        self.batch_size = cfg.get("batch_size", 128)
        # Chiave letta direttamente dal YAML — nessuna variabile d'ambiente
        self._api_key = cfg.get("api_key", "")
        if not self._api_key or self._api_key == "...":
            raise ValueError(
                "voyage_3: imposta 'api_key' in configs/models.yaml"
            )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
    )
    def _call_api(self, batch: list[str], input_type: str) -> list[list[float]]:
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_id, "input": batch, "input_type": input_type}
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        input_type = "query" if is_query else "document"
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vecs.extend(self._call_api(batch, input_type))
        arr = np.array(all_vecs, dtype=np.float32)
        return _normalise(arr) if normalize else arr


# ---------------------------------------------------------------------------
# Google Embeddings 2  —  Vertex AI via google-genai SDK
# ---------------------------------------------------------------------------

class GoogleEmbedder(BaseEmbedder):
    """
    Calls Google Embeddings 2 (text-embedding-004) via the Vertex AI backend
    using the google-genai SDK, exactly as in the unit-test reference.

    Auth: Application Default Credentials — run once:
        gcloud auth application-default login

    No API key is needed; project_id and location are read from the YAML config.

    Install:
        pip install google-genai
    """

    def __init__(self, cfg: dict[str, Any]):
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError as exc:
            raise ImportError(
                "google-genai not installed. Run: pip install google-genai"
            ) from exc

        self.model_id   = cfg.get("model_id", "text-embedding-004")
        self.name       = cfg["name"]
        self.dimension  = cfg["dimension"]
        self.max_tokens = cfg["max_tokens"]
        self.cost_per_million_tokens = cfg.get("cost_per_million_tokens", 0.025)
        self.batch_size = cfg.get("batch_size", 100)

        project_id = cfg.get("project_id")
        location   = cfg.get("location", "us-central1")

        if not project_id:
            raise ValueError(
                "google_embeddings_2 in models.yaml must have 'project_id' set."
            )

        logger.info(
            f"GoogleEmbedder: project={project_id} "
            f"location={location} model={self.model_id}"
        )

        # Build the Vertex AI client — identical pattern to the test file
        self._client = _genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        self._types = _types

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
    )
    def _call_api(self, batch: list[str], task_type: str) -> list[list[float]]:
        """
        Call embed_content for a batch of texts.
        task_type is one of: RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT,
                             SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING.
        """
        response = self._client.models.embed_content(
            model=self.model_id,
            contents=batch,
            config=self._types.EmbedContentConfig(task_type=task_type),
        )
        # response.embeddings is a list of ContentEmbedding objects
        return [emb.values for emb in response.embeddings]

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        all_vecs: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_vecs.extend(self._call_api(batch, task_type))

        arr = np.array(all_vecs, dtype=np.float32)
        return _normalise(arr) if normalize else arr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_embedder(model_key: str, config_path: str = "configs/models.yaml") -> BaseEmbedder:
    """Load an embedder by model key from the YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["models"][model_key]
    model_cfg.setdefault("name", model_key)

    provider = model_cfg["type"]
    if provider == "local":
        return LocalEmbedder(model_cfg)
    elif model_cfg.get("provider") == "openai":
        return OpenAIEmbedder(model_cfg)
    elif model_cfg.get("provider") == "cohere":
        return CohereEmbedder(model_cfg)
    elif model_cfg.get("provider") == "voyageai":
        return VoyageEmbedder(model_cfg)
    elif model_cfg.get("provider") == "google":
        return GoogleEmbedder(model_cfg)
    else:
        raise ValueError(f"Unknown provider for model {model_key}: {model_cfg}")


def measure_embedding_latency(
    embedder: BaseEmbedder,
    sample_texts: list[str],
    n_warmup: int = 3,
    n_measure: int = 20,
) -> dict[str, float]:
    """Measure per-query embedding latency in milliseconds."""
    # Warm-up
    for _ in range(n_warmup):
        embedder.embed(sample_texts[:1])

    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        embedder.embed(sample_texts[:1])
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }
