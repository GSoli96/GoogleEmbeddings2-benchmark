"""
utils.py
========
Shared utilities: LaTeX table generation, CSV export,
logging setup, seed management, and configuration loading.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/models.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    highlight_best: bool = True,
    highlight_second: bool = True,
    float_fmt: str = ".3f",
    output_path: Optional[str] = None,
) -> str:
    """
    Convert a DataFrame to an IEEE-style LaTeX table with bold best
    and underlined second-best values per row.

    Args:
        df: Results DataFrame (rows = datasets, columns = models).
        caption: Table caption string.
        label: LaTeX label string.
        highlight_best: Bold the best value in each row.
        highlight_second: Underline the second-best value.
        float_fmt: Python format string for float formatting.
        output_path: If provided, write .tex file.

    Returns:
        LaTeX table string.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    str_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    col_spec = "l" + "c" * len(numeric_cols)
    lines = [
        r"\begin{table}[t]",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}" + col_spec + r"@{}}",
        r"\toprule",
    ]

    # Header
    header_parts = str_cols + [c.replace("@", r"@\,") for c in numeric_cols]
    lines.append(" & ".join([r"\textbf{" + h + r"}" for h in header_parts]) + r" \\")
    lines.append(r"\midrule")

    # Rows
    for _, row in df.iterrows():
        row_vals = []
        for col in str_cols:
            row_vals.append(str(row[col]))

        num_vals = [row[c] for c in numeric_cols]
        if highlight_best or highlight_second:
            sorted_unique = sorted(set(num_vals), reverse=True)
            best = sorted_unique[0] if sorted_unique else None
            second = sorted_unique[1] if len(sorted_unique) > 1 else None

        for val in num_vals:
            formatted = format(val, float_fmt)
            if highlight_best and val == best:
                formatted = r"\textbf{" + formatted + r"}"
            elif highlight_second and val == second:
                formatted = r"\underline{" + formatted + r"}"
            row_vals.append(formatted)

        lines.append(" & ".join(row_vals) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex)
        logging.getLogger(__name__).info(f"LaTeX table saved to {output_path}")

    return latex


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(
    results: list[dict],
    output_path: str,
) -> pd.DataFrame:
    """Append new results to CSV, replacing any row with the same key columns."""
    df_new = pd.DataFrame(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df_old = pd.read_csv(path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        key_cols = df_combined.select_dtypes(exclude=[np.number]).columns.tolist()
        if key_cols:
            df_combined = df_combined.drop_duplicates(subset=key_cols, keep="last")
        df_combined.to_csv(output_path, index=False)
        logging.getLogger(__name__).info(f"Results appended to {output_path}")
        return df_combined

    df_new.to_csv(output_path, index=False)
    logging.getLogger(__name__).info(f"Results saved to {output_path}")
    return df_new


# ---------------------------------------------------------------------------
# Pretty print results
# ---------------------------------------------------------------------------

def print_results_table(df: pd.DataFrame, metric: str = "nDCG@10") -> None:
    """Print a formatted summary table to stdout."""
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
    except ImportError:
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def get_hardware_info() -> dict[str, Any]:
    """Collect hardware configuration for reproducibility logging."""
    info: dict[str, Any] = {"cpu": "unknown", "gpu": [], "ram_gb": "unknown"}
    try:
        import platform
        info["platform"] = platform.platform()
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu"] = platform.processor()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = [
                {"name": torch.cuda.get_device_name(i),
                 "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return info
