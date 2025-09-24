#!/usr/bin/env python3
"""
plot.py — HEST results visualization in one file (HPC-friendly)

Features
--------
- Load a run (dataset_results.json) and pick best encoder by pearson_mean.
- Extract k-fold per-gene correlations (results_kfold.json).
- Tidy per-split corrs and merge with test-split CSV metadata.
- Optionally annotate genes with curated Excel (panel=480, cell_type, condition).
- Optionally enrich splits with HEST directory metadata for XeniumPR1/pilot.
- Generate standard plots (matplotlib-only): barplot, histogram, per-sample grouped.

Design
------
- Single file for easy drop-in: no extra module splitting.
- Defensive: optional data sources are truly optional; code still runs without them.
- Minimal dependencies: pandas, numpy, matplotlib.

CLI
---
python plot.py \
  --run run_25-09-05-16-12-17 \
  --runs-root /project/simmons_hts/kxu/hest/eval/runs \
  --splits-root /project/simmons_hts/kxu/hest/eval/data \
  --curated-xlsx /project/simmons_hts/kxu/hest/curated_gene_list.xlsx \
  --extra-metadata /project/simmons_hts/kxu/hest/hest_directory.csv \
  --out plots/ --top-n 30 --show
"""

from __future__ import annotations

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------
# Defaults / constants
# -----------------------

DEFAULT_RUNS_ROOT = "/project/simmons_hts/kxu/hest/eval/ST_pred_results"
DEFAULT_SPLITS_ROOT = "/project/simmons_hts/kxu/hest/eval/data"
DEFAULT_CURATED_XLSX = "/project/simmons_hts/kxu/hest/curated_gene_list.xlsx"
DEFAULT_EXTRA_METADATA = "/project/simmons_hts/kxu/hest/hest_directory.csv"


# -----------------------
# Core IO
# -----------------------

def summarize_runs(root_dir):
    """
    List runs in ST_pred_results and summarize config.json details along with
    highest Pearson mean and std from dataset_results.json.

    Args:
        root_dir (str): Root directory containing the run folders.

    Returns:
        pd.DataFrame: Summary dataframe for all runs.
    """
    summary = []

    for run in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run)
        if not os.path.isdir(run_path) or not run.startswith("run_"):
            continue

        config_data = {}
        gene_list = ""
        config_found = False
        best_model = None

        # Search for config.json
        for dirpath, _, filenames in os.walk(run_path):
            if "config.json" in filenames:
                config_path = os.path.join(dirpath, "config.json")
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    gene_list = config_data.get("gene_list", "")
                config_found = True
                break

        # Search for dataset_results.json
        dataset_results_path = os.path.join(run_path, "dataset_results.json")
        highest_mean = None
        highest_std = None
        if os.path.isfile(dataset_results_path):
            with open(dataset_results_path, 'r') as f:
                data = json.load(f)
                all_results = data.get("results", [])[0].get("results", [])
                if all_results:
                    best_entry = max(all_results, key=lambda x: x["pearson_mean"])
                    highest_mean = best_entry["pearson_mean"]
                    highest_std = best_entry["pearson_std"]
                    best_model = best_entry['encoder_name']

        summary.append({
            "run": run,
            "gene_list": gene_list,
            "alpha": config_data.get("alpha"),
            "batch_size": config_data.get("batch_size"),
            "dimreduce": config_data.get("dimreduce"),
            "encoders": ", ".join(config_data.get("encoders", [])) if config_found else None,
            "normalize": config_data.get("normalize"),
            "latent_dim": config_data.get("latent_dim"),
            "method": config_data.get("method"),
            "dataset": config_data.get("datasets", [None])[0] if config_found else None,
            "best_model":best_model,
            "highest_pearson_mean": highest_mean,
            "highest_pearson_std": highest_std
        })

    df = pd.DataFrame(summary)
    return df

def best_results_by_gene_and_dataset(df):
    """
    Return the best result for each unique combination of gene_list and dataset.

    Args:
        df (pd.DataFrame): DataFrame returned by summarize_runs().

    Returns:
        pd.DataFrame: Best results per (gene_list, dataset).
    """
    if df.empty:
        return df

    # For each (gene_list, dataset), pick row with max highest_pearson_mean
    idx = (
        df.groupby(["gene_list", "dataset"])["highest_pearson_mean"]
        .idxmax()
        .dropna()
    )
    
    return df.loc[idx].reset_index(drop=True)


def _safe_read_json(path: Path) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing JSON: {path}")
    with open(path, "r") as f:
        return json.load(f)


def load_run(run: str, runs_root: str = DEFAULT_RUNS_ROOT) -> Tuple[str, pd.DataFrame, dict]:
    """Load dataset_results.json from a run directory."""
    run_dir = Path(runs_root) / run
    ds = _safe_read_json(run_dir / "dataset_results.json")
    ds0 = (ds.get("results") or [None])[0] or {}
    dataset_name = ds0.get("dataset_name", "Unknown")
    entries = ds0.get("results") or []

    results_df = pd.DataFrame(entries)
    if results_df.empty:
        raise ValueError(f"No 'results' entries in {run_dir}/dataset_results.json")

    for col in ["encoder_name", "pearson_mean", "pearson_std"]:
        if col not in results_df:
            results_df[col] = np.nan

    idx = results_df["pearson_mean"].astype(float).idxmax()
    best_model_info: dict = results_df.loc[idx].to_dict()

    best_model_info.setdefault("gene_corrs", {})
    best_model_info.setdefault("per_sample_corrs", [])

    return dataset_name, results_df, best_model_info


def extract_best_model_gene_corrs(run: str,
                                  runs_root: str = DEFAULT_RUNS_ROOT,
                                  verbose: bool = True):
    """
    From a run folder:
      - choose best model by pearson_mean
      - read <run>/<dataset>/<encoder>/results_kfold.json
      - build df_genes with ['gene','mean_corr','std_corr','corr_per_split']
    """
    run_path = Path(runs_root) / run
    dataset_name, _, best_model_info = load_run(run_path)
    encoder_name = best_model_info.get("encoder_name", "Unknown")

    if verbose:
        print(f"[extract] dataset={dataset_name} encoder={encoder_name} "
              f"pearson_mean={best_model_info.get('pearson_mean')} "
              f"std={best_model_info.get('pearson_std')}")

    # results_kfold.json path
    kfold_path = run_path / dataset_name / encoder_name / "results_kfold.json"
    if not kfold_path.is_file():
        raise FileNotFoundError(f"results_kfold.json not found at {kfold_path}")

    kfold = _safe_read_json(kfold_path)
    pearson_corrs = kfold.get("pearson_corrs", [])

    records = []
    for g in pearson_corrs:
        if not g or "name" not in g:
            continue
        records.append({
            "gene": g.get("name"),
            "mean_corr": g.get("mean"),
            "std_corr": g.get("std"),
            "corr_per_split": g.get("pearson_corrs", [])
        })

    df_genes = pd.DataFrame(records).convert_dtypes().fillna(pd.NA)
    return best_model_info, dataset_name, df_genes


# -----------------------
# Curated gene list (optional)
# -----------------------

import pandas as pd

def annotate_genes_with_curated(df_genes: pd.DataFrame, path_meta = "/project/simmons_hts/kxu/hest/curated_gene_list.xlsx", case_insensitive: bool = True) -> pd.DataFrame:
    """
    Minimal annotation:
      - panel: '480' if gene in '480 panel full list', else <NA>
      - cell_type from 'Cell Type Specific' + 'Where?'
      - condition from 'Conditional' + 'Where?.1'
    Merges onto df_genes by 'gene' (case-insensitive by default).

    Args:
        df_genes: DataFrame with at least ['gene'].
        df_meta: curated Excel sheet as a DataFrame.
        case_insensitive: if True, match by uppercased gene symbols.

    Returns:
        df_genes with columns ['panel','cell_type','condition'] added when available.
    """

    if "gene" not in df_genes.columns:
        raise ValueError("df_genes must contain a 'gene' column")
    
    df_meta = pd.read_excel(path_meta, sheet_name=0)

    m = df_meta.copy()

    # 1) rename the two columns exactly as you showed
    m = m.rename(columns={
        "Where?": "cell_type",
        "Where?.1": "condition"
    })

    # 2) build the three mapping frames (one row per gene)
    # panel (480 list)
    if "480 panel full list" in m.columns:
        df_panel = (
            m[["480 panel full list"]]
            .dropna()
            .rename(columns={"480 panel full list": "gene"})
        )
        df_panel["panel"] = "480"
        df_panel = df_panel.drop_duplicates(subset=["gene"])
    else:
        df_panel = pd.DataFrame(columns=["gene", "panel"])

    # cell_type
    if "Cell Type Specific" in m.columns and "cell_type" in m.columns:
        df_celltype = (
            m[["Cell Type Specific", "cell_type"]]
            .dropna(subset=["Cell Type Specific"])
            .rename(columns={"Cell Type Specific": "gene"})
            .drop_duplicates(subset=["gene"])
        )
    else:
        df_celltype = pd.DataFrame(columns=["gene", "cell_type"])

    # condition
    if "Conditional" in m.columns and "condition" in m.columns:
        df_condition = (
            m[["Conditional", "condition"]]
            .dropna(subset=["Conditional"])
            .rename(columns={"Conditional": "gene"})
            .drop_duplicates(subset=["gene"])
        )
    else:
        df_condition = pd.DataFrame(columns=["gene", "condition"])

    # 3) (optional) case-insensitive merge keys
    def _prep_key(df, col="gene"):
        out = df.copy()
        out[col] = out[col].astype(str).str.strip()
        if case_insensitive:
            out["_gk"] = out[col].str.upper()
        else:
            out["_gk"] = out[col]
        return out

    g   = _prep_key(df_genes, "gene")
    p   = _prep_key(df_panel, "gene")
    ct  = _prep_key(df_celltype, "gene")
    cond= _prep_key(df_condition, "gene")

    # 4) left-merge the three annotations
    out = g.merge(p[["_gk", "panel"]], on="_gk", how="left")
    out = out.merge(ct[["_gk", "cell_type"]], on="_gk", how="left")
    out = out.merge(cond[["_gk", "condition"]], on="_gk", how="left")

    # 5) clean up
    out = out.drop(columns=["_gk"]).convert_dtypes().fillna(pd.NA)
    return out


# -----------------------
# Splits & metadata
# -----------------------

def get_test_splits(run: str,
                    runs_root: str = DEFAULT_RUNS_ROOT,
                    splits_root: str = DEFAULT_SPLITS_ROOT,
                    extra_metadata_csv: str = DEFAULT_EXTRA_METADATA) -> pd.DataFrame:
    """Load test splits and optionally merge HEST directory metadata."""
    run_dir = Path(runs_root) / run
    dataset_name, _, _ = load_run(run_dir)

    dataset_split_dir = Path(splits_root) / dataset_name / "splits"
    test_files = sorted(glob.glob(str(dataset_split_dir / "test_*.csv")))
    if not test_files:
        raise FileNotFoundError(f"No test_*.csv files found in {dataset_split_dir}")

    rows = []
    for tf in test_files:
        split_num = int(os.path.basename(tf).replace("test_", "").replace(".csv", ""))
        df_split = pd.read_csv(tf)
        for sample in df_split["sample_id"].tolist():
            rows.append({"split": split_num, "test_sample": str(sample)})

    df_test = pd.DataFrame(rows).convert_dtypes().fillna(pd.NA)

    ds_l = str(dataset_name).strip().lower()
    if ds_l in {"xeniumpr1", "pilot"} and Path(extra_metadata_csv).exists():
        meta = pd.read_csv(extra_metadata_csv)
        df_test = df_test.merge(meta, left_on="test_sample", right_on="SampleID", how="left")
        df_test = df_test.drop(columns=["SampleID"], errors="ignore").convert_dtypes().fillna(pd.NA)

    return df_test


# -----------------------
# Tidy/merge utilities
# -----------------------

def merge_kfold_gene_corrs_with_test_metadata(df_genes: pd.DataFrame,
                                              df_test_splits: pd.DataFrame) -> pd.DataFrame:
    """Explode corr_per_split and merge with test splits."""
    if "gene" not in df_genes or "corr_per_split" not in df_genes:
        return pd.DataFrame()

    df_long = df_genes.explode("corr_per_split").rename(columns={"corr_per_split": "corr"})
    df_long["split"] = df_long.groupby("gene").cumcount()
    return df_long.merge(df_test_splits, on="split", how="left").convert_dtypes().fillna(pd.NA)


# -----------------------
# Plotting
# -----------------------

def _format_title(prefix, dataset_name, encoder_name):
    return f"{prefix} | Data: {dataset_name} | Model: {encoder_name}"

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

def compare_models(
    run: str,
    runs_root: str = DEFAULT_RUNS_ROOT,
    show: bool = False,
):
    """
    Visualize encoder-level dataset results for a given run as a point plot with error bars.
    Compatible with generate_all_plots: uses load_run() and returns a matplotlib Figure.
    If show=True, displays the figure inline (Jupyter-friendly).

    Args:
        run (str): Run folder name.
        runs_root (str): Root directory containing run folders.
        show (bool): If True, display the figure inline.

    Returns:
        matplotlib.figure.Figure | None
    """
    run_dir = Path(runs_root) / run

    # Use the shared loader to parse dataset_results.json and pick up dataset name
    try:
        dataset_name, results_df, _best = load_run(run_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[visualize_dataset_results] {e}")
        return None

    if results_df.empty:
        print(f"[visualize_dataset_results] No encoder results found in {run_dir}")
        return None

    # Extract encoders + stats
    encoders = results_df.get("encoder_name", pd.Series([], dtype="string")).astype(str).tolist()
    means = pd.to_numeric(results_df.get("pearson_mean", pd.Series([], dtype="float")), errors="coerce").to_numpy()
    stds  = pd.to_numeric(results_df.get("pearson_std",  pd.Series([], dtype="float")), errors="coerce").to_numpy()

    # Find config.json anywhere in the run dir to show gene list info (optional)
    gene_list_name = ""
    gene_count = None
    for dirpath, _, filenames in os.walk(run_dir):
        if "config.json" in filenames:
            config_path = Path(dirpath) / "config.json"
            try:
                with open(config_path, "r") as cf:
                    config = json.load(cf)
                gene_list_name = config.get("gene_list", "") or config.get("genes", "") or ""
                # Try to count genes if file exists and contains a JSON list
                if gene_list_name:
                    gl_path = Path(gene_list_name)
                    if not gl_path.is_file():
                        gl_path = (Path(dirpath) / gene_list_name)
                    if gl_path.is_file():
                        try:
                            with open(gl_path, "r") as gf:
                                genes = json.load(gf)
                            if isinstance(genes, list):
                                gene_count = len(genes)
                        except Exception:
                            # Silently ignore if it's not JSON; you can extend to txt/tsv if needed
                            pass
            except Exception:
                pass
            break

    # Build the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(encoders))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=5, markersize=8, linestyle="")

    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation")

    title = f"Dataset: {dataset_name}"
    if gene_list_name:
        title += f" | Gene List: {gene_list_name}"
    if gene_count is not None:
        title += f" ({gene_count} genes)"
    ax.set_title(title)

    fig.tight_layout()

    if show:
        display(fig)

    return fig


def plot_gene_correlation_barplot(df, dataset_name, encoder_name,
                                          value_col="mean_corr",
                                          top_n=30, figsize=(12, 7)):
    """
    Horizontal barplot of top-N genes, showing mean correlations with std dev as error bars.

    Args:
        df: DataFrame with at least ['gene', value_col]. Optional ['std_corr'] for error bars.
        dataset_name: str
        encoder_name: str
        group_col: str, metadata column for coloring (currently not used in this minimal version).
        value_col: str, column with mean correlations (default 'mean_corr').
        top_n: int, number of top genes to show.
        figsize: tuple
    """
    if df.empty or value_col not in df:
        fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
        ax.text(0.5, 0.5, "No gene correlations to plot", ha="center", va="center")
        return fig

    d = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    means = d[value_col].astype(float).to_numpy()
    y_pos = np.arange(len(d))

    # Use std_corr if available
    if "std_corr" in d.columns:
        errs = d["std_corr"].astype(float).to_numpy()
        ax.barh(y_pos, means, xerr=errs, align="center", alpha=0.7, ecolor="black", capsize=3)
    else:
        ax.barh(y_pos, means, align="center", alpha=0.7)

    ax.set_yticks(y_pos, d["gene"].astype(str))
    ax.set_xlabel("Pearson correlation (mean ± std)")
    ax.set_title(_format_title("Gene Correlation", dataset_name, encoder_name))
    fig.tight_layout()
    return fig

def plot_gene_correlation_barplot_grouped(df_genes, group_by='cell_type', show_mean=True):
    """
    Plot individual gene mean correlations grouped by a specified column (cell_type or condition).
    
    Args:
        df_genes (pd.DataFrame): Must contain 'gene', 'mean_corr', 'std_corr', and the grouping column.
        group_by (str): Column name to group by ('cell_type' or 'condition').
        show_mean (bool): Whether to show a horizontal line per group indicating mean correlation.
    
    Returns:
        matplotlib.figure.Figure: Figure object for further saving or manipulation.
    """
    if group_by not in ['cell_type', 'condition', 'panel']:
        raise ValueError("group_by must be 'cell_type' or 'condition' or 'panel")
    
    # Filter out rows where grouping column is NA
    df_plot = df_genes[df_genes[group_by].notna()].copy()
    
    # Compute average mean_corr per group for ordering
    group_order = df_plot.groupby(group_by)['mean_corr'].mean().sort_values(ascending=False).index.tolist()
    
    # Sort genes by group and descending mean_corr
    df_plot[f'{group_by}_ordered'] = pd.Categorical(df_plot[group_by], categories=group_order, ordered=True)
    df_plot = df_plot.sort_values([f'{group_by}_ordered', 'mean_corr'], ascending=[True, False])
    
    # Map colors
    cmap = plt.get_cmap('tab20', len(group_order))
    color_map = {grp: cmap(i) for i, grp in enumerate(group_order)}
    colors = [color_map[grp] for grp in df_plot[group_by]]
    
    # X positions
    x = np.arange(len(df_plot))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(df_plot)*0.2), 6))
    
    # Plot bars with error bars
    ax.bar(x, df_plot['mean_corr'], yerr=df_plot['std_corr'], color=colors, edgecolor='black', capsize=4)
    
    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot['gene'], rotation=90)
    ax.set_ylabel("Mean Pearson Correlation")
    ax.set_title(f"Gene Mean Correlations by {group_by.capitalize()}")
    
    # Draw horizontal line per group if requested
    if show_mean:
        start_idx = 0
        for grp in group_order:
            grp_genes = df_plot[df_plot[group_by] == grp]
            if len(grp_genes) == 0:
                continue
            mean_corr = grp_genes['mean_corr'].mean()
            end_idx = start_idx + len(grp_genes) - 1
            ax.hlines(y=mean_corr, xmin=start_idx-0.4, xmax=end_idx+0.4,
                      colors=color_map[grp], linestyles='dashed', linewidth=2, alpha=0.7)
            start_idx = end_idx + 1
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1,color=color_map[grp]) for grp in group_order]
    ax.legend(handles, group_order, title=group_by.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig

def plot_gene_correlation_histogram(df, dataset_name, encoder_name,
                                    value_col="mean_corr", bins=30, figsize=(8, 5)):
    if df.empty or value_col not in df:
        fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
        return fig
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[value_col].dropna().astype(float), bins=bins)
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Number of genes")
    ax.set_title(_format_title("Gene Correlation Histogram", dataset_name, encoder_name))
    return fig



def plot_corrs_by_sample(
    df,
    dataset_name,
    encoder_name,
    group_by: str | None = None,
    figsize=(16, 6)
):
    """
    Plot per-split gene correlations grouped by metadata, and return the figure object.

    Args:
        df (pd.DataFrame): must contain columns:
            [split, gene, corr, test_sample, Sample_type, Location, cell_type, condition]
        group_by (str or None): column to color/group samples by
            (e.g. 'Sample_type' or 'Location'). If None, no grouping (all samples same color).

    Returns:
        matplotlib.figure.Figure
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = df.copy()

    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"{group_by} must be a column in df")

    # ---------------- ordering of x-axis ----------------
    if group_by is not None:
        # Order groups by mean corr, then samples within each group
        group_means = df.groupby(group_by)["corr"].mean().sort_values(ascending=False)
        group_order = group_means.index.tolist()
        x_order = []
        for grp in group_order:
            samples = (
                df.loc[df[group_by] == grp, "test_sample"]
                  .dropna()
                  .drop_duplicates()
                  .tolist()
            )
            x_order.extend(samples)
        # deduplicate while preserving order
        x_order = list(dict.fromkeys(x_order))
    else:
        # Just order test_sample by mean corr
        sample_means = df.groupby("test_sample")["corr"].mean().sort_values(ascending=False)
        x_order = sample_means.index.tolist()

    df["test_sample"] = pd.Categorical(df["test_sample"], categories=x_order, ordered=True)

    # ---------------- plotting ----------------
    fig, ax = plt.subplots(figsize=figsize)

    if group_by is not None:
        sns.boxplot(
            data=df,
            x="test_sample",
            y="corr",
            hue=group_by,
            showfliers=False,
            palette="tab20",
            ax=ax,
        )
        ax.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel(f"Test Sample (grouped by {group_by})")
    else:
        sns.boxplot(
            data=df,
            x="test_sample",
            y="corr",
            color="skyblue",
            showfliers=False,
            ax=ax,
        )
        # no legend
        ax.set_xlabel("Test Sample")

    # ---------------- styling ----------------
    plt.xticks(rotation=60, ha="right")
    ax.set_ylabel("Gene Correlation")
    ax.set_title(_format_title("Per-sample correlation", dataset_name, encoder_name))
    plt.tight_layout()

    return fig


# -----------------------
# High-level workflow
# -----------------------

from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

def generate_all_plots(
    run: str,
    group_by: Union[str, List[str], None] = None,
    runs_root: str = DEFAULT_RUNS_ROOT,
    splits_root: str = DEFAULT_SPLITS_ROOT,
    curated_xlsx: Optional[str] = DEFAULT_CURATED_XLSX,
    extra_metadata_csv: Optional[str] = DEFAULT_EXTRA_METADATA,
    top_n: int = 30,
    show: bool = False,
) -> Dict[str, Path]:
    """
    New workflow:
      - Always produce base plots (gene_barplot, gene_hist, per_sample).
      - If `group_by` is provided (str or list[str]), produce only those grouped plots
        *when the needed metadata/columns exist*.
        * Gene-level grouping supported for: panel, cell_type, condition (needs curated_xlsx).
        * Per-sample grouping supported for any column present in df_long (e.g., Sample_type, Location, panel, cell_type, condition).
    Saves into <runs_root>/<run>/plots with simple filenames.
    """
    # Paths
    run_dir = Path(runs_root) / run
    outdir = run_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Best model + gene correlations (k-fold aware)
    best, dataset_name, df_genes = extract_best_model_gene_corrs(run, runs_root=runs_root, verbose=False)
    encoder_name = best.get("encoder_name", "Unknown")

    # 2) Splits (+ optional HEST directory metadata inside get_test_splits)
    df_splits = get_test_splits(run, runs_root=runs_root, splits_root=splits_root, extra_metadata_csv=extra_metadata_csv)

    # 3) Long per-split df
    df_long = merge_kfold_gene_corrs_with_test_metadata(df_genes, df_splits)

    # 4) Curated annotations (optional; needed for gene-level grouping by panel/cell_type/condition)
    curated_ok = isinstance(curated_xlsx, str) and Path(curated_xlsx).exists()
    if curated_ok:
        df_genes_annot = annotate_genes_with_curated(df_genes, curated_xlsx)
        df_long_annot  = annotate_genes_with_curated(df_long,  curated_xlsx)
    else:
        df_genes_annot = df_genes
        df_long_annot  = df_long

    arts: Dict[str, Path] = {}

    # ---------------- Base plots (always) ----------------
    fig = compare_models(run, runs_root=runs_root, show=show)
    if fig is not None:
        p = (Path(runs_root) / run / "plots" / "model_comparison.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        if show:
            display(fig)
        plt.close(fig)
        arts["model_comparison"] = p

    # Gene barplot (ungrouped)
    fig = plot_gene_correlation_barplot(df_genes, dataset_name, encoder_name)
    p = outdir / "gene_barplot.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if show: display(fig)
    plt.close(fig)
    arts["gene_barplot"] = p

    # Histogram
    fig = plot_gene_correlation_histogram(df_genes, dataset_name, encoder_name)
    p = outdir / "gene_hist.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if show: display(fig)
    plt.close(fig)
    arts["gene_hist"] = p

    # Per-sample (no grouping)
    fig = plot_corrs_by_sample(df_long, dataset_name, encoder_name, group_by=None)
    p = outdir / "per_sample.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if show: display(fig)
    plt.close(fig)
    arts["per_sample"] = p

    # ---------------- Gene-level grouped barplots (only what user requested) ----------------
    # normalize group_by to a list
    requested_groups: List[str] = []
    if group_by is not None:
        requested_groups = [group_by] if isinstance(group_by, str) else list(group_by)

    # Your grouped gene barplot supports: panel / cell_type / condition
    valid_gene_groups = {"panel", "cell_type", "condition"}

    for gb in requested_groups:
        if gb not in valid_gene_groups:
            # skip silently if user asked for something not supported by this function
            continue
        # only plot if curated annotations are available and column is present with non-NA values
        if curated_ok and (gb in df_genes_annot.columns) and df_genes_annot[gb].notna().any():
            fig = plot_gene_correlation_barplot_grouped(
                df_genes_annot,           # ← your function signature
                group_by=gb,              # pass the requested group
                show_mean=True
            )
            p = outdir / f"gene_barplot_by_{gb}.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            if show: display(fig)
            plt.close(fig)
            arts[f"gene_barplot_by_{gb}"] = p
        # if not curated / or column empty → skip

    # ---------------- Per-sample grouped (optional, if user also wants these) ----------------
    # If you also want to produce per-sample grouped plots based on the same `group_by` items:
    for gb in requested_groups:
        # pick annotated long df if it has the column; else fall back to df_long
        if gb in df_long_annot.columns and df_long_annot[gb].notna().any():
            dplot = df_long_annot
        elif gb in df_long.columns and df_long[gb].notna().any():
            dplot = df_long
        else:
            continue

        fig = plot_corrs_by_sample(dplot, dataset_name, encoder_name, group_by=gb)
        p = outdir / f"per_sample_by_{gb}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        if show: display(fig)
        plt.close(fig)
        arts[f"per_sample_by_{gb}"] = p

    return arts


## plot gene panels overlap between broad & in house Xenium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _find_col(df, candidates, required=True, what="column"):
    """Return the first column from `candidates` that exists in df.columns."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find {what}. Looked for: {candidates}. "
                       f"Available: {list(df.columns)}")
    return None

import matplotlib.pyplot as plt

def compare_runs_and_plot(run_a, label_a, run_b, label_b):
    # extract
    _, _, df_a = extract_best_model_gene_corrs(run=run_a)
    _, _, df_b = extract_best_model_gene_corrs(run=run_b)

    # get intersection
    genes_a = set(df_a["gene"])
    genes_b = set(df_b["gene"])
    common = sorted(genes_a & genes_b)

    # filter to common
    df_a_common = df_a[df_a["gene"].isin(common)].rename(
        columns={"mean_corr": f"mean_corr_{label_a}"}
    )
    df_b_common = df_b[df_b["gene"].isin(common)].rename(
        columns={"mean_corr": f"mean_corr_{label_b}"}
    )

    # merge
    wide = df_a_common.merge(df_b_common, on="gene")

    # reshape for plotting
    df_plot = wide.melt(
        id_vars="gene",
        value_vars=[f"mean_corr_{label_a}", f"mean_corr_{label_b}"],
        var_name="dataset",
        value_name="mean_corr"
    )
    df_plot["dataset"] = df_plot["dataset"].str.replace("mean_corr_", "")

    # order genes by highest correlation
    gene_order = (
        df_plot.groupby("gene")["mean_corr"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    df_plot["gene"] = pd.Categorical(df_plot["gene"], categories=gene_order, ordered=True)

    # scale figure size
    n_genes = len(gene_order)
    fig_height = max(6, n_genes * 0.25)  # at least 6 inches tall
    fig_width = 12

    # plot horizontal grouped bars
    plt.figure(figsize=(fig_width, fig_height))
    for i, dataset in enumerate(df_plot["dataset"].unique()):
        subset = df_plot[df_plot["dataset"] == dataset]
        plt.barh(
            [y + i*0.4 for y in range(len(subset))],
            subset["mean_corr"],
            height=0.4,
            label=dataset,
        )

    plt.yticks([y + 0.2 for y in range(len(gene_order))], gene_order, fontsize=6)
    plt.xlabel("Mean correlation")
    plt.ylabel("Gene")
    plt.legend(title="Dataset")
    plt.title(f"Mean correlation per gene ({label_a} vs {label_b})")
    plt.gca().invert_yaxis()  # highest correlation at the top
    plt.tight_layout()
    plt.show()

    return common, wide
