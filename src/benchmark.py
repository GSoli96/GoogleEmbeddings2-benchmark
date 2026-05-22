"""
benchmark.py
============
Main CLI entry point for the Google Embeddings 2 benchmark.

Usage examples:
  # Run full benchmark (all models, all datasets)
  python benchmark.py run-all

  # Run BEIR only on specific models
  python benchmark.py run-beir --models ge2 oai

  # Run Italian RAG evaluation
  python benchmark.py run-italian --model ge2

  # Run chunking ablation
  python benchmark.py run-chunking --model ge2

  # Generate all figures from saved results
  python benchmark.py figures

  # Show hardware info
  python benchmark.py hardware
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # fix macOS OMP duplicate-lib crash (FAISS + numpy)

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loaders import load_beir_dataset, load_miracl_dataset, load_italian_benchmark
from embeddings import load_embedder, measure_embedding_latency
from metrics import evaluate_run
from plotting import generate_all_figures
from rag_eval import rag_eval_sweep
from retrieval import build_index, measure_retrieval_latency, run_retrieval
from utils import (
    get_hardware_info, load_config, print_results_table,
    save_results_csv, set_seed, setup_logging,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_KEYS = {
    "ge2":    "google_embeddings_2",
    "oai":    "openai_text_emb_3_large",
    "cohere": "cohere_embed_v3",
    "voyage": "voyage_3",
    "bge":    "bge_m3",
    "e5":     "e5_large",
    "me5":    "multilingual_e5_large",
    "labse":  "labse",
    "mpnet":  "mpnet_multilingual",
}

ALL_MODEL_KEYS = list(MODEL_KEYS.values())

BEIR_SUBSETS = ["nfcorpus", "scifact", "trec-covid", "fiqa"]
MIRACL_LANGS = ["en", "it", "fr", "de", "es", "ar", "zh", "ja"]


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--seed", default=42, help="Random seed.")
@click.option("--log-level", default="INFO", help="Logging level.")
def cli(seed: int, log_level: str):
    """Google Embeddings 2 Benchmarking Framework."""
    setup_logging(log_level)
    set_seed(seed)


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

@cli.command()
def hardware():
    """Print hardware configuration."""
    info = get_hardware_info()
    click.echo(json.dumps(info, indent=2))


# ---------------------------------------------------------------------------
# BEIR benchmark
# ---------------------------------------------------------------------------

@cli.command("run-beir")
@click.option(
    "--models", multiple=True, default=list(MODEL_KEYS.keys()),
    help="Model short keys to evaluate (default: all)."
)
@click.option("--datasets", multiple=True, default=BEIR_SUBSETS)
@click.option("--top-k", default=100)
def run_beir(models, datasets, top_k):
    """Run BEIR evaluation across models and datasets."""
    all_rows = []

    for model_key in models:
        full_key = MODEL_KEYS.get(model_key, model_key)
        print("full_key", full_key)
        try:
            embedder = load_embedder(full_key)
        except Exception as e:
            logger.error(f"Failed to load {model_key}: {e}")
            continue

        for ds_name in datasets:
            logger.info(f"[BEIR] {model_key} / {ds_name}")
            try:
                dataset = load_beir_dataset(ds_name)
                retriever = build_index(dataset.corpus, embedder)
                run = run_retrieval(dataset.queries, embedder, retriever, top_k=top_k)
                metrics = evaluate_run(run, dataset.qrels)
                row = {"model": model_key, "dataset": ds_name, **metrics}
                all_rows.append(row)
                logger.info(f"  nDCG@10={metrics.get('nDCG@10', 0):.4f}")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    if all_rows:
        df = save_results_csv(all_rows, str(RESULTS_DIR / "beir_results.csv"))
        print_results_table(df)


# ---------------------------------------------------------------------------
# MIRACL benchmark
# ---------------------------------------------------------------------------

@cli.command("run-miracl")
@click.option("--models", multiple=True, default=list(MODEL_KEYS.keys()))
@click.option("--languages", multiple=True, default=MIRACL_LANGS)
@click.option("--top-k", default=100)
def run_miracl(models, languages, top_k):
    """Run MIRACL multilingual evaluation."""
    all_rows = []

    for model_key in models:
        full_key = MODEL_KEYS.get(model_key, model_key)
        try:
            embedder = load_embedder(full_key)
        except Exception as e:
            logger.error(f"Failed to load {model_key}: {e}")
            continue

        for lang in languages:
            logger.info(f"[MIRACL] {model_key} / {lang}")
            try:
                dataset = load_miracl_dataset(lang)
                retriever = build_index(dataset.corpus, embedder)
                run = run_retrieval(dataset.queries, embedder, retriever, top_k=top_k)
                metrics = evaluate_run(run, dataset.qrels)
                row = {"model": model_key, "language": lang, **metrics}
                all_rows.append(row)
                logger.info(f"  Recall@10={metrics.get('Recall@10', 0):.4f}")
            except Exception as e:
                logger.error(f"  Failed: {e}")

    if all_rows:
        df = save_results_csv(all_rows, str(RESULTS_DIR / "miracl_results.csv"))
        print_results_table(df)


# ---------------------------------------------------------------------------
# Italian RAG benchmark
# ---------------------------------------------------------------------------

@cli.command("run-italian")
@click.option("--models", multiple=True, default=list(MODEL_KEYS.keys()))
@click.option("--top-k", default=10)
def run_italian(models, top_k):
    """Run IT-RAG-Bench evaluation."""
    dataset = load_italian_benchmark()
    logger.info(f"IT-RAG-Bench: {dataset.summary()}")

    all_rows = []
    for model_key in models:
        full_key = MODEL_KEYS.get(model_key, model_key)
        try:
            embedder = load_embedder(full_key)
            retriever = build_index(dataset.corpus, embedder)
            run = run_retrieval(dataset.queries, embedder, retriever, top_k=top_k)
            metrics = evaluate_run(run, dataset.qrels)
            row = {"model": model_key, **metrics}
            all_rows.append(row)
            logger.info(f"[{model_key}] nDCG@10={metrics.get('nDCG@10', 0):.4f}")
        except Exception as e:
            logger.error(f"  Failed {model_key}: {e}")

    if all_rows:
        df = save_results_csv(all_rows, str(RESULTS_DIR / "italian_results.csv"))
        print_results_table(df)


# ---------------------------------------------------------------------------
# Chunking ablation
# ---------------------------------------------------------------------------

@cli.command("run-chunking")
@click.option("--models", multiple=True, default=["ge2", "oai", "bge"])
@click.option(
    "--strategies", multiple=True,
    default=["fixed", "sliding_window", "semantic"]
)
@click.option("--chunk-sizes", multiple=True, default=[8,16,32,64,128], type=int)
def run_chunking(models, strategies, chunk_sizes):
    """Run chunking ablation on IT-RAG-Bench."""
    dataset = load_italian_benchmark()
    all_dfs = []

    for model_key in models:
        full_key = MODEL_KEYS.get(model_key, model_key)
        try:
            embedder = load_embedder(full_key)
            df = rag_eval_sweep(
                dataset, embedder, model_name=model_key,
                strategies=list(strategies),
                chunk_sizes=list(chunk_sizes),
                output_csv=str(RESULTS_DIR / f"chunking_{model_key}.csv"),
            )
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed {model_key}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(str(RESULTS_DIR / "chunking_results.csv"), index=False)
        print_results_table(combined)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@cli.command("run-latency")
@click.option("--models", multiple=True, default=list(MODEL_KEYS.keys()))
@click.option("--n-samples", default=50, help="Number of latency measurements.")
def run_latency(models, n_samples):
    """Measure embedding and retrieval latency per model."""
    dataset = load_italian_benchmark()
    sample_queries = dict(list(dataset.queries.items())[:5])
    all_rows = []

    for model_key in models:
        full_key = MODEL_KEYS.get(model_key, model_key)
        try:
            embedder = load_embedder(full_key)
            sample_corpus = dict(list(dataset.corpus.items())[:500])
            retriever = build_index(sample_corpus, embedder)
            latency = measure_retrieval_latency(
                sample_queries, embedder, retriever,
                n_measure=n_samples,
            )
            row = {"model": model_key, **latency}
            all_rows.append(row)
            logger.info(f"[{model_key}] p50={latency['p50_ms']:.1f} ms")
        except Exception as e:
            logger.error(f"Failed {model_key}: {e}")

    if all_rows:
        df = save_results_csv(all_rows, str(RESULTS_DIR / "latency_results.csv"))
        print_results_table(df)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

@cli.command("figures")
def figures():
    """Generate all publication-quality figures from saved results."""
    generate_all_figures(str(RESULTS_DIR))
    click.echo("Figures saved to paper/figures/")


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

@cli.command("run-all")
@click.option("--models", multiple=True, default=list(MODEL_KEYS.keys()))
@click.pass_context
def run_all(ctx, models):
    """Run the full benchmark pipeline (BEIR + MIRACL + Italian + Chunking + Latency)."""
    click.echo("=== Full Benchmark Pipeline ===\n")
    ctx.invoke(run_beir, models=models, datasets=BEIR_SUBSETS, top_k=100)
    ctx.invoke(run_miracl, models=models, languages=MIRACL_LANGS, top_k=100)
    ctx.invoke(run_italian, models=models, top_k=10)
    ctx.invoke(run_chunking, models=models[:3])
    ctx.invoke(run_latency, models=models)
    ctx.invoke(figures)
    click.echo("\n=== Benchmark complete. Results in results/ ===")


if __name__ == "__main__":
    cli()
