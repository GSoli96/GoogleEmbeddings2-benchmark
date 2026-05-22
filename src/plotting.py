"""
plotting.py
===========
Publication-quality figure generation for the benchmark paper.
All data is read dynamically from CSVs in the results/ directory —
no hardcoded numbers anywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

# IEEE column widths
SINGLE_COL = 3.5
DOUBLE_COL = 7.16

# Seed palette / markers for known model keys; unknown models get auto-assigned.
# Maps short CLI keys (as they appear in CSV "model" column) to YAML model keys
_MODEL_SHORT_TO_YAML: dict[str, str] = {
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

# Cells where the model ran out of memory on that dataset: (model_short, dataset)
_ML_CELLS: set[tuple[str, str]] = {("bge", "trec-covid")}

_PALETTE: dict[str, str] = {
    "ge2":    "#1f77b4",
    "oai":    "#ff7f0e",
    "cohere": "#2ca02c",
    "voyage": "#d62728",
    "bge":    "#9467bd",
    "e5":     "#8c564b",
    "me5":    "#e377c2",
    "labse":  "#7f7f7f",
    "mpnet":  "#bcbd22",
}
_MARKERS: dict[str, str] = {
    "ge2":    "o",
    "oai":    "s",
    "cohere": "^",
    "voyage": "D",
    "bge":    "v",
    "e5":     "P",
    "me5":    "*",
    "labse":  "X",
    "mpnet":  "h",
}
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<", ">", "p"]


def _color(model: str) -> str:
    if model not in _PALETTE:
        idx = len(_PALETTE)
        _PALETTE[model] = matplotlib.colormaps["tab20"](idx % 20)
    return _PALETTE[model]


def _marker(model: str) -> str:
    if model not in _MARKERS:
        idx = len(_MARKERS)
        _MARKERS[model] = _MARKER_CYCLE[idx % len(_MARKER_CYCLE)]
    return _MARKERS[model]


plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.labelsize":   8,
    "axes.titlesize":   9,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "lines.linewidth":  1.2,
    "lines.markersize": 5,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved: {path}")


def _load_results(results_dir: str) -> dict[str, pd.DataFrame]:
    """
    Scan results_dir for every known CSV pattern and return a dict of DataFrames.
    Per-model files (chunking_bge.csv, chunking_e5.csv, …) are merged together.
    """
    rp = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}

    def _read_glob(pattern: str) -> pd.DataFrame | None:
        frames = []
        for f in sorted(rp.glob(pattern)):
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True) if frames else None

    for key, pattern in [
        ("beir",    "beir_results*.csv"),
        ("italian", "italian_results*.csv"),
        ("miracl",  "miracl_results*.csv"),
        ("latency", "latency_results*.csv"),
    ]:
        df = _read_glob(pattern)
        if df is not None:
            out[key] = df

    # chunking: merge chunking_results.csv + all chunking_<model>.csv
    chunk_frames = []
    for f in sorted(rp.glob("chunking*.csv")):
        try:
            chunk_frames.append(pd.read_csv(f))
        except Exception:
            pass
    if chunk_frames:
        combined = pd.concat(chunk_frames, ignore_index=True)
        dedup_cols = [c for c in ("model", "strategy", "chunk_size") if c in combined.columns]
        out["chunking"] = combined.drop_duplicates(subset=dedup_cols or None)

    return out


def _load_model_costs(config_path: str = "configs/models.yaml") -> dict[str, float]:
    """Read cost_per_million_tokens from models.yaml, keyed by short CLI key."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        yaml_costs = {k: float(v.get("cost_per_million_tokens", 0.0))
                      for k, v in cfg.get("models", {}).items()}
        # Invert _MODEL_SHORT_TO_YAML to map yaml_key -> short_key
        yaml_to_short = {v: k for k, v in _MODEL_SHORT_TO_YAML.items()}
        return {yaml_to_short.get(yaml_k, yaml_k): cost
                for yaml_k, cost in yaml_costs.items()}
    except Exception:
        return {}


def _short_name(model: str, config_path: str = "configs/models.yaml") -> str:
    """Return short_name from YAML if available, otherwise the raw key."""
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg["models"].get(model, {}).get("short_name", model)
    except Exception:
        return model


# ---------------------------------------------------------------------------
# Figure 1: Italian / BEIR bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison_bar(
    df: pd.DataFrame,
    metric: str = "nDCG@10",
    title: str = "Model comparison",
    output_path: str = "paper/figures/model_comparison.pdf",
) -> None:
    """Horizontal bar chart comparing models on a single aggregated metric."""
    if metric not in df.columns or "model" not in df.columns:
        print(f"  Skipping {output_path}: missing columns")
        return

    # Average across datasets / strategies if multiple rows per model
    agg = df.groupby("model")[metric].mean().sort_values(ascending=True)
    models = list(agg.index)
    values = list(agg.values)
    colors = [_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, max(1.8, 0.35 * len(models) + 0.6)))
    bars = ax.barh(models, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=6)
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.set_xlim(0, min(1.0, agg.max() * 1.25))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2: BEIR per-dataset heatmap
# ---------------------------------------------------------------------------

def plot_beir_heatmap(
    df: pd.DataFrame,
    metric: str = "nDCG@10",
    output_path: str = "paper/figures/beir_heatmap.pdf",
) -> None:
    """Heatmap of models × BEIR datasets."""
    if "model" not in df.columns or "dataset" not in df.columns or metric not in df.columns:
        print(f"  Skipping {output_path}: missing columns")
        return

    pivot = df.pivot_table(index="model", columns="dataset", values=metric, aggfunc="mean")
    if pivot.empty:
        return

    # Row order: ge2 first, bge last, rest sorted alphabetically in between
    _row_priority = {"ge2": -1, "bge": len(pivot)}
    ordered_rows = sorted(pivot.index, key=lambda m: (_row_priority.get(m, 0), m))
    pivot = pivot.loc[ordered_rows]

    # Mask ML cells so imshow shows them as white / NaN
    display = pivot.copy().astype(float)
    for model, dataset in _ML_CELLS:
        if model in display.index and dataset in display.columns:
            display.loc[model, dataset] = np.nan

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, max(1.5, 0.4 * len(pivot) + 0.8)))
    masked = np.ma.masked_invalid(display.values)
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for r, model in enumerate(pivot.index):
        for c, dataset in enumerate(pivot.columns):
            if (model, dataset) in _ML_CELLS:
                ax.text(c, r, "ML", ha="center", va="center",
                        fontsize=7, color="darkred", fontweight="bold")
            else:
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color="white" if val > 0.6 else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label=metric)
    ax.set_title(f"BEIR {metric}")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3: Chunking ablation
# ---------------------------------------------------------------------------

def plot_chunking_ablation(
    df: pd.DataFrame,
    metric: str = "nDCG@10",
    output_path: str = "paper/figures/chunking_ablation.pdf",
) -> None:
    """Line plot of metric vs chunk_size, faceted by strategy, coloured by model."""
    required = {"model", "strategy", "chunk_size", metric}
    if not required.issubset(df.columns):
        print(f"  Skipping {output_path}: missing columns {required - set(df.columns)}")
        return

    strategies = sorted(df["strategy"].unique())
    models = sorted(df["model"].unique())
    n_strats = len(strategies)

    fig, axes = plt.subplots(1, n_strats, figsize=(DOUBLE_COL, 2.4), sharey=True)
    if n_strats == 1:
        axes = [axes]

    chunk_sizes = sorted(df["chunk_size"].unique())

    for ax, strategy in zip(axes, strategies):
        df_s = df[df["strategy"] == strategy]
        for model in models:
            df_m = df_s[df_s["model"] == model].sort_values("chunk_size")
            if df_m.empty:
                continue
            ax.plot(
                df_m["chunk_size"], df_m[metric],
                marker=_marker(model), color=_color(model),
                label=model,
            )
        ax.set_title(strategy.replace("_", " ").title())
        ax.set_xlabel("Chunk size (tokens)")
        ax.set_xticks(chunk_sizes)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper center", ncol=6, shadow=True,
                   bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4: Latency vs nDCG scatter (Pareto plot)
# ---------------------------------------------------------------------------

def plot_latency_ndcg_scatter(
    latency_df: pd.DataFrame,
    ndcg_df: pd.DataFrame,
    metric: str = "nDCG@10",
    output_path: str = "paper/figures/latency_ndcg_scatter.pdf",
) -> None:
    """Scatter of per-query latency vs nDCG@10, with Pareto frontier."""
    if "model" not in latency_df.columns or "mean_ms" not in latency_df.columns:
        print(f"  Skipping {output_path}: latency data incomplete")
        return
    if metric not in ndcg_df.columns or "model" not in ndcg_df.columns:
        print(f"  Skipping {output_path}: nDCG data incomplete")
        return

    lat_cols = [c for c in ("mean_ms", "std_ms") if c in latency_df.columns]
    lat = latency_df.groupby("model")[lat_cols].mean()
    ndcg = ndcg_df.groupby("model")[metric].mean()
    merged = lat.join(ndcg, how="inner")
    if merged.empty:
        return

    models = list(merged.index)
    latencies = merged["mean_ms"].values
    stds = merged["std_ms"].values if "std_ms" in merged.columns else np.zeros(len(models))
    ndcgs = merged[metric].values

    pareto_mask = _pareto_front(latencies, ndcgs)
    pareto_lat = latencies[pareto_mask]
    pareto_ndcg = ndcgs[pareto_mask]
    sort_idx = np.argsort(pareto_lat)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    ax.plot(pareto_lat[sort_idx], pareto_ndcg[sort_idx],
            "--", color="gray", linewidth=0.8, zorder=1, label="Pareto front")

    for m, lat_v, std_v, ndcg_v in zip(models, latencies, stds, ndcgs):
        ax.errorbar(lat_v, ndcg_v, xerr=std_v,
                    fmt=_marker(m), color=_color(m), markersize=6,
                    capsize=2, elinewidth=0.7, label=m, zorder=5)

    ax.set_xlabel("Per-query latency (ms)")
    ax.set_ylabel(f"{metric}")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_xlim(left=0)
    fig.tight_layout()
    _save(fig, output_path)


def _pareto_front(latencies: np.ndarray, ndcgs: np.ndarray) -> np.ndarray:
    n = len(latencies)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and latencies[j] <= latencies[i] and ndcgs[j] >= ndcgs[i] \
               and (latencies[j] < latencies[i] or ndcgs[j] > ndcgs[i]):
                dominated[i] = True
                break
    return ~dominated


# ---------------------------------------------------------------------------
# Figure 5: Throughput & cost dual-axis bar
# ---------------------------------------------------------------------------

def plot_throughput_cost(
    chunking_df: pd.DataFrame,
    costs: dict[str, float],
    output_path: str = "paper/figures/throughput_cost.pdf",
) -> None:
    """Dual-axis bar: throughput (passages/sec) left, cost/1M tokens right."""
    if "model" not in chunking_df.columns or "throughput_pps" not in chunking_df.columns:
        print(f"  Skipping {output_path}: throughput data missing")
        return

    thr = chunking_df.groupby("model")["throughput_pps"].mean().sort_values(ascending=False)
    models = list(thr.index)
    if not models:
        return

    x = np.arange(len(models))
    width = 0.35
    throughputs = thr.values
    cost_vals = [costs.get(m, 0.0) for m in models]
    colors = [_color(m) for m in models]

    fig, ax1 = plt.subplots(figsize=(SINGLE_COL, 2.6))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, throughputs, width, color=colors, alpha=0.85, label="Throughput")
    ax2.bar(x + width / 2, cost_vals,   width, color="lightcoral", alpha=0.7, label="Cost/1M tok")

    ax1.set_ylabel("Passages/sec", color="steelblue")
    ax2.set_ylabel("Cost per 1M tokens (USD)", color="coral")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right")
    ax1.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=6)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 6: MIRACL radar
# ---------------------------------------------------------------------------

def plot_miracl_radar(
    df: pd.DataFrame,
    metric: str = "Recall@10",
    output_path: str = "paper/figures/miracl_radar.pdf",
) -> None:
    """Radar / spider chart of metric across MIRACL languages."""
    if "model" not in df.columns or "language" not in df.columns or metric not in df.columns:
        print(f"  Skipping {output_path}: missing columns")
        return

    pivot = df.pivot_table(index="language", columns="model", values=metric, aggfunc="mean")
    if pivot.empty:
        return

    languages = list(pivot.index)
    n_langs = len(languages)
    if n_langs < 3:
        return

    angles = np.linspace(0, 2 * np.pi, n_langs, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL), subplot_kw=dict(polar=True))
    for model in pivot.columns:
        values = pivot[model].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.0, color=_color(model),
                label=model, markersize=3)
        ax.fill(angles, values, alpha=0.05, color=_color(model))

    ax.set_thetagrids(np.degrees(angles[:-1]), languages)
    ax.set_ylim(0, 1.0)
    ax.tick_params(pad=3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6)
    ax.set_title(f"MIRACL {metric}", pad=12, fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str = "results") -> None:
    """
    Load all available CSVs from results_dir and produce every figure
    for which data exists. Missing files are skipped with a warning.
    """
    data = _load_results(results_dir)
    costs = _load_model_costs()

    if not data:
        print(f"No result files found in '{results_dir}' — run the benchmark first.")
        return

    # Figure: Italian benchmark bar chart
    if "italian" in data:
        plot_model_comparison_bar(
            data["italian"], metric="nDCG@10", title="IT-RAG-Bench nDCG@10",
            output_path="paper/figures/italian_ndcg_bar.pdf",
        )
    else:
        print("  Skipping Italian bar: italian_results*.csv not found")

    # Figure: BEIR heatmap
    if "beir" in data:
        plot_beir_heatmap(data["beir"], output_path="paper/figures/beir_heatmap.pdf")
        plot_model_comparison_bar(
            data["beir"], metric="nDCG@10", title="BEIR avg nDCG@10",
            output_path="paper/figures/beir_avg_ndcg_bar.pdf",
        )
    else:
        print("  Skipping BEIR figures: beir_results*.csv not found")

    # Figure: Chunking ablation
    if "chunking" in data:
        plot_chunking_ablation(data["chunking"], output_path="paper/figures/chunking_ablation.pdf")
        plot_throughput_cost(
            data["chunking"], costs,
            output_path="paper/figures/throughput_cost.pdf",
        )
    else:
        print("  Skipping chunking figures: chunking*.csv not found")

    # Figure: Latency vs nDCG scatter
    if "latency" in data:
        ndcg_source = data.get("italian") if "italian" in data else data.get("beir")
        if ndcg_source is not None:
            plot_latency_ndcg_scatter(
                data["latency"], ndcg_source,
                output_path="paper/figures/latency_ndcg_scatter.pdf",
            )
        else:
            print("  Skipping latency scatter: no nDCG source (italian/beir) found")
    else:
        print("  Skipping latency scatter: latency_results*.csv not found")

    # Figure: MIRACL radar
    if "miracl" in data:
        plot_miracl_radar(data["miracl"], output_path="paper/figures/miracl_radar.pdf")
    else:
        print("  Skipping MIRACL radar: miracl_results*.csv not found")

    print("\nAll figures generated in paper/figures/")
