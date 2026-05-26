---
language:
  - it
license: cc-by-nc-4.0
task_categories:
  - text-retrieval
task_ids:
  - document-retrieval
pretty_name: IT-RAG-Bench
size_categories:
  - 1K<n<10K
tags:
  - retrieval
  - RAG
  - Italian
  - embeddings
  - benchmark
  - synthetic
configs:
  - config_name: corpus
    data_files:
      - split: train
        path: data/corpus.jsonl
  - config_name: queries
    data_files:
      - split: train
        path: data/queries.jsonl
  - config_name: qrels
    data_files:
      - split: train
        path: data/qrels.jsonl
---

# IT-RAG-Bench: Italian Retrieval & RAG Benchmark

**IT-RAG-Bench** is a synthetic Italian-language retrieval benchmark designed to evaluate dense embedding models on document retrieval and Retrieval-Augmented Generation (RAG) tasks in Italian.

This dataset is the companion resource for the paper:

> **Benchmarking Google Embeddings 2 against Open-Source Models for Multilingual Dense Retrieval and RAG Systems**  
> Stefano Cirillo, Domenico Desiato, Giuseppe Polese, Giandomenico Solimando — arXiv 2026  
> 📄 [https://arxiv.org/abs/2605.23618](https://arxiv.org/abs/2605.23618)  
> 💻 [https://github.com/cciro94/GoogleEmbeddings2-benchmark](https://github.com/cciro94/GoogleEmbeddings2-benchmark)

---

## Dataset Summary

IT-RAG-Bench provides **3,200 Italian passages** and **640 natural-language queries** spanning three document styles representative of real Italian information retrieval scenarios:

- **Encyclopedic** passages (Wikipedia-style)
- **FAQ** passages (public-administration question–answer pairs)
- **Legal/regulatory** article excerpts (Italian legislative style)

The dataset was generated synthetically with a fixed random seed (42) using Italian-language topic vocabularies drawn from AI/NLP, legal, and public-administration domains, ensuring full reproducibility without relying on crawled or licensed data.

---

## Dataset Structure

### Configurations

| Config | File | Rows | Description |
|--------|------|------|-------------|
| `corpus` | `data/corpus.jsonl` | 3,200 | Passages to retrieve from |
| `queries` | `data/queries.jsonl` | 640 | Italian natural-language queries |
| `qrels` | `data/qrels.jsonl` | 1,246 | Query–document relevance judgements |

### Corpus Composition

| Type | Count | Style description |
|------|-------|-------------------|
| `wiki` | 1,200 | Encyclopedic passages about AI, NLP, and Italian regulatory topics |
| `faq` | 800 | Public-administration FAQ: question + structured answer |
| `legal` | 1,200 | Synthetic Italian legal articles (Art. N format) |

**Total passages:** 3,200 | **Total queries:** 640 | **Total relevance pairs:** 1,246

### Field Schema

**`corpus.jsonl`**
```json
{
  "_id":  "wiki_it_0",
  "text": "Il documento esamina intelligenza artificiale nel contesto di edilizia ...",
  "type": "wiki"
}
```

**`queries.jsonl`**
```json
{
  "_id":  "q_it_0",
  "text": "Quali sono i risultati dell'uso di intelligenza artificiale nel settore sanità?"
}
```

**`qrels.jsonl`**
```json
{
  "query_id":  "q_it_0",
  "corpus_id": "wiki_it_42",
  "score":     1
}
```

Relevance scores are binary (0 / 1). Each query has between 1 and 3 relevant documents.

---

## Dataset Creation

### Motivation

Existing Italian retrieval benchmarks are scarce and often require special licensing. IT-RAG-Bench was created to provide a freely available, reproducible Italian evaluation set for comparing embedding models across typical Italian enterprise retrieval scenarios (administrative portals, legal databases, public FAQs).

### Generation Process

The corpus is generated from parameterised templates filled with Italian-language vocabulary lists:

- **Topics:** intelligenza artificiale, reti neurali, recupero dell'informazione, elaborazione del linguaggio naturale, contratti digitali, normativa GDPR, diritto del lavoro, appalti pubblici, tutela ambientale, previdenza sociale, proprietà intellettuale, riforma fiscale
- **Sectors:** pubblica amministrazione, sanità, istruzione, finanza, edilizia, agricoltura, trasporti, commercio elettronico, sicurezza informatica, energia rinnovabile

Queries are generated from six Italian question templates (e.g., *"Quali sono i risultati dell'uso di {topic} nel settore {sector}?"*). Relevance labels are assigned by randomly associating each query with 1–3 documents of a randomly selected type.

**Random seed:** 42 (fully deterministic and reproducible)

### Caveats

Because relevance labels are randomly assigned rather than annotated by humans, absolute metric scores are lower than on human-annotated benchmarks. The dataset is best used for **comparative evaluation** (ranking models against each other) rather than measuring absolute retrieval performance.

---

## Benchmark Results

The following results were obtained in the companion paper using FAISS HNSW indexing and nDCG@10 as the primary metric (640 queries):

| Model | Type | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|-------|------|----------|----------|-----------|-----|---------|
| **GE2** (Google Embeddings 2) | API | 0.061 | 0.288 | 0.476 | 0.259 | **0.282** |
| **mE5** (multilingual-e5-large) | Open | 0.051 | 0.280 | 0.489 | 0.243 | 0.279 |
| **E5-large** | Open | 0.053 | 0.279 | 0.439 | 0.247 | 0.262 |
| **mpnet** | Open | 0.054 | 0.238 | 0.397 | 0.240 | 0.243 |
| **BGE-M3** | Open | 0.046 | 0.253 | 0.404 | 0.224 | 0.238 |
| **LaBSE** | Open | 0.048 | 0.190 | 0.315 | 0.184 | 0.189 |

### Experimental Setup

- **Index:** FAISS HNSW (M=32, efConstruction=200, efSearch=128)
- **Embedding cache:** SHA-256 keyed disk cache (diskcache), preventing redundant API calls
- **Reproducibility:** fixed seed 42, deterministic CUDA operations
- **Hardware:** NVIDIA A100 40GB (open-source models), API calls for commercial models

---

## Usage

```python
from datasets import load_dataset

# Load corpus, queries, and relevance judgements
corpus  = load_dataset("Siando/it-rag-bench", "corpus",  split="train")
queries = load_dataset("Siando/it-rag-bench", "queries", split="train")
qrels   = load_dataset("Siando/it-rag-bench", "qrels",   split="train")

# Example: build a corpus dict
corpus_dict = {row["_id"]: row["text"] for row in corpus}
```

### Minimal retrieval evaluation example

```python
from datasets import load_dataset

corpus  = {r["_id"]: r["text"] for r in load_dataset("Siando/it-rag-bench", "corpus",  split="train")}
queries = {r["_id"]: r["text"] for r in load_dataset("Siando/it-rag-bench", "queries", split="train")}

qrels = {}
for r in load_dataset("Siando/it-rag-bench", "qrels", split="train"):
    qrels.setdefault(r["query_id"], {})[r["corpus_id"]] = r["score"]
```

---

## Repository & Code

The full benchmarking framework (embedding clients, FAISS retrieval, chunking ablations, plotting) is available at:

**[https://github.com/cciro94/GoogleEmbeddings2-benchmark](https://github.com/cciro94/GoogleEmbeddings2-benchmark)**

---

## Citation

If you use IT-RAG-Bench in your research, please cite:

```bibtex
@misc{cirillo2026benchmarkinggoogleembeddings2,
    title     = {Benchmarking Google Embeddings 2 against Open-Source Models for Multilingual Dense Retrieval and RAG Systems},
    author    = {Stefano Cirillo and Domenico Desiato and Giuseppe Polese and Giandomenico Solimando},
    year      = {2026},
    eprint    = {2605.23618},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    url       = {https://arxiv.org/abs/2605.23618}
}
```

---

## License

This dataset is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
You are free to use and adapt it for non-commercial purposes with proper attribution.  
See [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/) for details.
