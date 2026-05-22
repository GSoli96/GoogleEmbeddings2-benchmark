# Benchmarking Google Embeddings 2 for Multilingual Dense Retrieval and RAG Systems

[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](https://arxiv.org)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Companion code for the IEEE paper:**  
> *"Benchmarking Google Embeddings 2 for Multilingual Dense Retrieval and RAG Systems"*

---

## Overview

This repository provides a **fully reproducible benchmarking framework** comparing six embedding models across:

| Task | Datasets |
|------|----------|
| English retrieval | BEIR (NFCorpus, SciFact, TREC-COVID, FiQA) |
| Multilingual retrieval | MIRACL (8 languages: en, it, fr, de, es, ar, zh, ja) |
| Italian RAG quality | IT-RAG-Bench (synthetic, 3,200 passages) |
| Chunking ablation | Fixed / Sliding-window / Semantic × 8/16/32/64/128/256/512/1024 tokens |
| Latency & cost | Per-query latency, throughput, cost/1M tokens |

### Models Evaluated

| Short Name | Full Name | Type |
|------------|-----------|------|
| GE2 | Google Embeddings 2 | API |
| OAI-3L | OpenAI text-embedding-3-large | API |
| Cohere-v3 | Cohere Embed v3 | API |
| Voyage-3 | Voyage-3 | API |
| BGE-M3 | BAAI/bge-m3 | Open |
| E5-L | intfloat/e5-large-v2 | Open |

---

## Project Structure

```
project/ 
├── paper/
│   ├── main.tex              # IEEE LaTeX source (compile with pdflatex + bibtex)
│   ├── references.bib        # BibTeX bibliography
│   └── figures/              # PDF figures (auto-generated)
│
├── src/
│   ├── benchmark.py          # CLI entry point (run-all, run-beir, etc.)
│   ├── embeddings.py         # Multi-provider embedding clients
│   ├── retrieval.py          # FAISS ANN index and retrieval
│   ├── datasets.py           # BEIR, MIRACL, IT-RAG-Bench loaders
│   ├── chunking.py           # Fixed / sliding-window / semantic chunking
│   ├── metrics.py            # Recall@k, MRR, nDCG@k, efficiency metrics
│   ├── plotting.py           # Publication-quality matplotlib figures
│   ├── rag_eval.py           # RAG retrieval pipeline evaluation
│   └── utils.py              # LaTeX tables, CSV export, logging, seeds
│
├── configs/
│   └── models.yaml           # Model and dataset configuration
│
├── results/                  # Auto-created; holds CSVs and indices
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/anonymous/ge2-benchmark.git
cd ge2-benchmark

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set API keys

Create a `.env` file (never commit this):

```bash
cp .env.example .env
```

Edit `.env`:

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=sk-...
COHERE_API_KEY=
VOYAGE_API_KEY=...
```

Load before running:

```bash
export $(cat .env | xargs)
```

---

## Reproducing All Experiments

### Quick start (all models, all datasets)

```bash
cd src
python benchmark.py run-all
```

### Step-by-step

```bash
# 1. BEIR evaluation
python benchmark.py run-beir --models ge2 oai cohere voyage bge e5

# 2. MIRACL multilingual evaluation
python benchmark.py run-miracl --models ge2 oai cohere voyage bge e5

# 3. Italian RAG benchmark
python benchmark.py run-italian --models ge2 oai cohere voyage bge e5

# 4. Chunking ablation (GE2, OAI, BGE only for speed)
python benchmark.py run-chunking --models ge2 oai bge \
    --strategies fixed sliding_window semantic \
    --chunk-sizes 256 512 1024

# 5. Latency measurement
python benchmark.py run-latency --models ge2 oai cohere voyage bge e5

# 6. Generate figures
python benchmark.py figures
```

### Offline / mock mode (no API keys required)

All API embedders fall back gracefully when keys are absent:  
the datasets module generates synthetic data, and the caching layer
prevents redundant API calls across runs.

```bash
# Run full pipeline in offline mode (uses synthetic datasets + random embeddings)
OFFLINE_MODE=1 python benchmark.py run-all
```

---

## Compiling the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires a standard TeX distribution (TeX Live 2022+ or MiKTeX 22+)  
with the `IEEEtran`, `booktabs`, `multirow`, `microtype` packages.

---

## Key Configuration

Edit `configs/models.yaml` to:
- Adjust batch sizes for your API rate limits
- Add new embedding models
- Change BEIR/MIRACL subsets
- Tune HNSW parameters

---

## Metrics Implemented

| Metric | Description |
|--------|-------------|
| Recall@1/5/10 | Fraction of relevant docs in top-k |
| MRR | Mean Reciprocal Rank |
| nDCG@10 | Normalised DCG with graded relevance |
| Latency (p50/p95 ms) | End-to-end query latency |
| Throughput (passages/sec) | Corpus embedding speed |
| Storage (MB) | Index memory footprint |
| Cost (USD/1M tokens) | Estimated API cost |

---

## Caching Strategy

All embeddings are cached on disk using `diskcache` with SHA-256 keys  
derived from `(model_id, text, normalize_flag)`. This ensures:

1. Identical results on re-runs (reproducibility).
2. Zero redundant API calls across experiments.
3. Cross-experiment sharing when the same text appears in multiple datasets.

Cache location: `.embedding_cache/` (configurable).  
Cache size limit: 10 GB (configurable in `embeddings.py`).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB |
| GPU | None (CPU fallback) | NVIDIA A100 40GB |
| Storage | 20 GB | 100 GB (for full caches) |
| Python | 3.11 | 3.11 |

Open-source models (BGE-M3, E5-large) use GPU automatically if available.

---

## Reproducibility Checklist

- [x] Fixed random seed (`--seed 42`) throughout
- [x] Deterministic CUDA operations (`cudnn.deterministic = True`)
- [x] Disk-cached embeddings (SHA-256 keyed)
- [x] Pinned dependency versions (`requirements.txt`)
- [x] Synthetic Italian dataset generated with fixed seed
- [x] Hardware configuration logged via `python benchmark.py hardware`
- [x] API retry with exponential backoff + jitter (max 5 retries)

---

## Citation

```bibtex
@article{anonymous2026ge2benchmark,
  title   = {Benchmarking Google Embeddings 2 for Multilingual Dense Retrieval and RAG Systems},
  author  = {Anonymous},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
