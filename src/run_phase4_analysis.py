"""Phase 4 aggregation, statistics, tables, and figures for retrieval fusion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_ORDER = ["bm25", "dense", "rrf", "zscore_fusion", "minmax_fusion", "validation_weighted_fusion"]
FUSION_METHODS = ["rrf", "zscore_fusion", "minmax_fusion", "validation_weighted_fusion"]
METRICS = ["ndcg@10", "mrr@10", "recall@100"]
UNIT_COLS = ["dataset", "budget", "seed"]
STATIC_BASELINE_CONTRASTS = [("validation_weighted_fusion", "minmax_fusion")]
METHOD_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "rrf": "RRF",
    "zscore_fusion": "Z-score fusion",
    "minmax_fusion": "Min-max fusion",
    "validation_weighted_fusion": "Validation-weighted fusion",
}


def contrast_record(
    method_label: str,
    baseline_label: str,
    metric: str,
    deltas: np.ndarray,
    *,
    stats_module: Any | None,
) -> dict[str, Any]:
    ci_low, ci_high = bootstrap_ci(deltas)
    p_value = None
    statistic = None
    if stats_module is not None and len(deltas) >= 2:
        if np.allclose(deltas, 0.0):
            p_value = 1.0
            statistic = 0.0
        else:
            result = stats_module.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            p_value = float(result.pvalue)
            statistic = float(result.statistic)
    std_delta = float(np.nanstd(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    return {
        "method_label": method_label,
        "baseline_label": baseline_label,
        "metric": metric,
        "paired_unit": "dataset-budget-seed",
        "n_units": int(len(deltas)),
        "mean_delta": float(np.nanmean(deltas)) if len(deltas) else float("nan"),
        "median_delta": float(np.nanmedian(deltas)) if len(deltas) else float("nan"),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_statistic": statistic,
        "p_value_raw": p_value,
        "cohen_dz": float(np.nanmean(deltas) / std_delta) if std_delta and not math.isnan(std_delta) else None,
        "positive_units": int(np.sum(deltas > 0.0)) if len(deltas) else 0,
        "negative_units": int(np.sum(deltas < 0.0)) if len(deltas) else 0,
        "zero_units": int(np.sum(np.isclose(deltas, 0.0))) if len(deltas) else 0,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def bootstrap_ci(values: np.ndarray, reps: int = 2000, seed: int = 20260719) -> tuple[float | None, float | None]:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(reps, len(clean)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and not math.isnan(float(p))]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: float(item[1]))
    m = len(indexed)
    running = 0.0
    for rank, (idx, p_value) in enumerate(indexed):
        raw = min(1.0, (m - rank) * float(p_value))
        running = max(running, raw)
        adjusted[idx] = min(1.0, running)
    return adjusted


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    baseline_rows: list[dict[str, Any]] = []
    for key, group in df.groupby(UNIT_COLS, sort=True):
        singles = group.set_index("method")
        if "bm25" not in singles.index or "dense" not in singles.index:
            continue
        row = dict(zip(UNIT_COLS, key))
        for metric in METRICS:
            bm25_value = float(singles.loc["bm25", metric])
            dense_value = float(singles.loc["dense", metric])
            row[f"bm25_{metric}"] = bm25_value
            row[f"dense_{metric}"] = dense_value
            row[f"average_single_{metric}"] = (bm25_value + dense_value) / 2.0
            row[f"best_single_{metric}"] = max(bm25_value, dense_value)
            row[f"worst_single_{metric}"] = min(bm25_value, dense_value)
        baseline_rows.append(row)
    baselines = pd.DataFrame(baseline_rows)
    enriched = df.merge(baselines, on=UNIT_COLS, how="left")
    for metric in METRICS:
        enriched[f"{metric}_delta_vs_average_single"] = enriched[metric] - enriched[f"average_single_{metric}"]
        enriched[f"{metric}_delta_vs_best_single"] = enriched[metric] - enriched[f"best_single_{metric}"]
    enriched["ndcg_regret_vs_best_single"] = enriched["best_single_ndcg@10"] - enriched["ndcg@10"]
    enriched["average_single_ndcg_regret"] = enriched["best_single_ndcg@10"] - enriched["average_single_ndcg@10"]
    enriched["worst_single_ndcg_regret"] = enriched["best_single_ndcg@10"] - enriched["worst_single_ndcg@10"]
    enriched["worst_case_regret_reduction"] = np.where(
        enriched["worst_single_ndcg_regret"].abs() > 1e-12,
        (enriched["worst_single_ndcg_regret"] - enriched["ndcg_regret_vs_best_single"])
        / enriched["worst_single_ndcg_regret"],
        0.0,
    )
    return enriched


def aggregate_outputs(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    aggregate_dir = output_dir / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    agg_spec = {
        "count": ("ndcg@10", "count"),
        "mean_ndcg@10": ("ndcg@10", "mean"),
        "std_ndcg@10": ("ndcg@10", "std"),
        "mean_mrr@10": ("mrr@10", "mean"),
        "std_mrr@10": ("mrr@10", "std"),
        "mean_recall@100": ("recall@100", "mean"),
        "std_recall@100": ("recall@100", "std"),
        "mean_latency_query_ms": ("latency_query_ms", "mean"),
        "mean_dense_weight": ("dense_weight", "mean"),
        "mean_ndcg_delta_vs_average_single": ("ndcg@10_delta_vs_average_single", "mean"),
        "mean_mrr_delta_vs_average_single": ("mrr@10_delta_vs_average_single", "mean"),
        "mean_ndcg_regret_vs_best_single": ("ndcg_regret_vs_best_single", "mean"),
        "mean_worst_case_regret_reduction": ("worst_case_regret_reduction", "mean"),
    }
    files: dict[str, str] = {}
    for name, keys in {
        "method_dataset_budget_summary.csv": ["dataset", "budget", "method"],
        "method_dataset_summary.csv": ["dataset", "method"],
        "method_budget_summary.csv": ["budget", "method"],
        "method_overall_summary.csv": ["method"],
    }.items():
        summary = df.groupby(keys, sort=True).agg(**agg_spec).reset_index()
        if "method" in summary.columns:
            summary["method_label"] = summary["method"].map(METHOD_LABELS)
        path = aggregate_dir / name
        summary.to_csv(path, index=False)
        files[name] = str(path)
    enriched_path = aggregate_dir / "formal_records_enriched.csv"
    df.to_csv(enriched_path, index=False)
    files["formal_records_enriched.csv"] = str(enriched_path)
    return files


def planned_contrasts(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    indexed = {method: group.set_index(UNIT_COLS) for method, group in df.groupby("method")}
    baseline_df = df.drop_duplicates(UNIT_COLS).set_index(UNIT_COLS)
    try:
        from scipy import stats
    except Exception:
        stats = None

    for method in FUSION_METHODS:
        if method not in indexed:
            continue
        method_df = indexed[method]
        for metric in METRICS:
            for baseline in ["bm25", "dense", "average_single", "best_single"]:
                units = method_df.index
                method_values = method_df.loc[units, metric].astype(float)
                if baseline in indexed:
                    common = units.intersection(indexed[baseline].index)
                    deltas = method_df.loc[common, metric].astype(float) - indexed[baseline].loc[common, metric].astype(float)
                else:
                    baseline_column = f"{baseline}_{metric}"
                    common = units.intersection(baseline_df.index)
                    deltas = method_df.loc[common, metric].astype(float) - baseline_df.loc[common, baseline_column].astype(float)
                delta_values = deltas.to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_ci(delta_values)
                p_value = None
                statistic = None
                if stats is not None and len(delta_values) >= 2:
                    if np.allclose(delta_values, 0.0):
                        p_value = 1.0
                        statistic = 0.0
                    else:
                        result = stats.wilcoxon(delta_values, zero_method="wilcox", alternative="two-sided")
                        p_value = float(result.pvalue)
                        statistic = float(result.statistic)
                std_delta = float(np.nanstd(delta_values, ddof=1)) if len(delta_values) > 1 else float("nan")
                records.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "baseline": baseline,
                        "metric": metric,
                        "paired_unit": "dataset-budget-seed",
                        "n_units": int(len(delta_values)),
                        "mean_delta": float(np.nanmean(delta_values)) if len(delta_values) else float("nan"),
                        "median_delta": float(np.nanmedian(delta_values)) if len(delta_values) else float("nan"),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_value_raw": p_value,
                        "cohen_dz": float(np.nanmean(delta_values) / std_delta) if std_delta and not math.isnan(std_delta) else None,
                    }
                )
    contrast_df = pd.DataFrame(records)
    if not contrast_df.empty:
        contrast_df["p_value_holm"] = None
        for metric, idx in contrast_df.groupby("metric").groups.items():
            adjusted = holm_adjust(contrast_df.loc[idx, "p_value_raw"].tolist())
            contrast_df.loc[idx, "p_value_holm"] = adjusted
    path = stats_dir / "planned_contrasts.csv"
    contrast_df.to_csv(path, index=False)
    return contrast_df, {"planned_contrasts.csv": str(path)}


def static_baseline_contrasts(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    indexed = {method: group.set_index(UNIT_COLS) for method, group in df.groupby("method")}
    try:
        from scipy import stats
    except Exception:
        stats = None

    records: list[dict[str, Any]] = []
    for method, baseline in STATIC_BASELINE_CONTRASTS:
        if method not in indexed or baseline not in indexed:
            continue
        common = indexed[method].index.intersection(indexed[baseline].index)
        for metric in ["ndcg@10", "mrr@10"]:
            deltas = (
                indexed[method].loc[common, metric].astype(float)
                - indexed[baseline].loc[common, metric].astype(float)
            ).to_numpy(dtype=float)
            record = contrast_record(
                METHOD_LABELS[method],
                METHOD_LABELS[baseline],
                metric,
                deltas,
                stats_module=stats,
            )
            record.update({"method": method, "baseline": baseline})
            records.append(record)

    contrast_df = pd.DataFrame(records)
    if not contrast_df.empty:
        contrast_df["p_value_holm"] = holm_adjust(contrast_df["p_value_raw"].tolist())
    path = stats_dir / "static_baseline_contrasts.csv"
    contrast_df.to_csv(path, index=False)
    return contrast_df, {"static_baseline_contrasts.csv": str(path)}


def friedman_tests(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    try:
        from scipy import stats
    except Exception:
        stats = None
    records: list[dict[str, Any]] = []
    for metric in METRICS:
        pivot = df.pivot_table(index=UNIT_COLS, columns="method", values=metric, aggfunc="mean")
        methods = [method for method in METHOD_ORDER if method in pivot.columns]
        pivot = pivot[methods].dropna()
        if stats is None or len(methods) < 3 or len(pivot) < 2:
            records.append(
                {
                    "metric": metric,
                    "paired_unit": "dataset-budget-seed",
                    "n_units": int(len(pivot)),
                    "methods": ",".join(methods),
                    "statistic": None,
                    "p_value": None,
                    "note": "descriptive only: repeated-measures test unavailable or insufficient paired units",
                }
            )
            continue
        result = stats.friedmanchisquare(*[pivot[method].to_numpy(dtype=float) for method in methods])
        records.append(
            {
                "metric": metric,
                "paired_unit": "dataset-budget-seed",
                "n_units": int(len(pivot)),
                "methods": ",".join(methods),
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "note": "Friedman repeated-measures test across methods",
            }
        )
    friedman_df = pd.DataFrame(records)
    path = stats_dir / "friedman_tests.csv"
    friedman_df.to_csv(path, index=False)
    return friedman_df, {"friedman_tests.csv": str(path)}


def write_tables(df: pd.DataFrame, contrast_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    overall = (
        df.groupby("method", sort=False)
        .agg(
            mean_ndcg=("ndcg@10", "mean"),
            mean_mrr=("mrr@10", "mean"),
            mean_recall=("recall@100", "mean"),
            mean_latency_ms=("latency_query_ms", "mean"),
            mean_ndcg_delta_vs_avg=("ndcg@10_delta_vs_average_single", "mean"),
        )
        .reindex([method for method in METHOD_ORDER if method in df["method"].unique()])
        .reset_index()
    )
    overall.insert(1, "method_label", overall["method"].map(METHOD_LABELS))
    dataset = (
        df.groupby(["dataset", "method"], sort=True)
        .agg(mean_ndcg=("ndcg@10", "mean"), mean_mrr=("mrr@10", "mean"), mean_regret=("ndcg_regret_vs_best_single", "mean"))
        .reset_index()
    )
    dataset["method_label"] = dataset["method"].map(METHOD_LABELS)
    contrast_view = contrast_df[
        (contrast_df["baseline"] == "average_single")
        & (contrast_df["metric"].isin(["ndcg@10", "mrr@10"]))
    ].copy()
    for name, frame in {
        "table_overall_quality.tex": overall,
        "table_dataset_summary.tex": dataset,
        "table_planned_contrasts.tex": contrast_view,
    }.items():
        path = tables_dir / name
        path.write_text(frame.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
        files[name] = str(path)
    return files


def make_figures(df: pd.DataFrame, figures_dir: Path) -> dict[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {}
    methods = [method for method in METHOD_ORDER if method in df["method"].unique()]
    datasets = sorted(df["dataset"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    def save(fig, stem: str) -> None:
        paths = []
        for suffix in ["png", "pdf"]:
            path = figures_dir / f"{stem}.{suffix}"
            fig.savefig(path, bbox_inches="tight", dpi=180)
            paths.append(str(path))
        plt.close(fig)
        files[stem] = paths

    quality = df.groupby(["dataset", "method"]).agg(mean_ndcg=("ndcg@10", "mean"), sem_ndcg=("ndcg@10", "sem"), mean_mrr=("mrr@10", "mean"), sem_mrr=("mrr@10", "sem")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    x = np.arange(len(datasets))
    width = 0.82 / max(1, len(methods))
    for idx, method in enumerate(methods):
        subset = quality[quality["method"] == method].set_index("dataset").reindex(datasets)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        axes[0].bar(x + offset, subset["mean_ndcg"], width, yerr=subset["sem_ndcg"].fillna(0), label=METHOD_LABELS[method], color=colors[idx], capsize=2)
        axes[1].bar(x + offset, subset["mean_mrr"], width, yerr=subset["sem_mrr"].fillna(0), label=METHOD_LABELS[method], color=colors[idx], capsize=2)
    for ax, ylabel in zip(axes, ["nDCG@10", "MRR@10"]):
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Dataset-wise ranking quality")
    axes[1].set_title("Dataset-wise reciprocal-rank quality")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    save(fig, "dataset_wise_ranking_quality")

    selected = "validation_weighted_fusion" if "validation_weighted_fusion" in methods else methods[-1]
    heat = (
        df[df["method"] == selected]
        .groupby(["dataset", "budget"])["worst_case_regret_reduction"]
        .mean()
        .unstack("budget")
        .reindex(datasets)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(heat.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(col) for col in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("Candidate budget")
    ax.set_title(f"Worst-case regret reduction: {METHOD_LABELS[selected]}")
    for row in range(len(heat.index)):
        for col in range(len(heat.columns)):
            value = heat.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Reduction ratio")
    save(fig, "worst_case_regret_reduction_heatmap")

    validation = df[df["method"] == "validation_weighted_fusion"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    budgets = sorted(validation["budget"].unique())
    weight_groups = [validation[validation["budget"] == budget]["dense_weight"].dropna().to_numpy(dtype=float) for budget in budgets]
    axes[0].boxplot(weight_groups, labels=[str(budget) for budget in budgets], showmeans=True)
    axes[0].set_xlabel("Candidate budget")
    axes[0].set_ylabel("Selected dense weight")
    axes[0].set_title("Validation-guided fusion weights")
    for method in ["rrf", "zscore_fusion", "minmax_fusion", "validation_weighted_fusion"]:
        if method not in methods:
            continue
        trend = df[df["method"] == method].groupby("budget")["ndcg@10"].mean().reindex(budgets)
        axes[1].plot(budgets, trend, marker="o", label=METHOD_LABELS[method])
    axes[1].set_xlabel("Candidate budget")
    axes[1].set_ylabel("Mean nDCG@10")
    axes[1].set_title("Budget trend")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    save(fig, "fusion_weight_and_budget_behavior")

    pareto = df.groupby("method").agg(mean_ndcg=("ndcg@10", "mean"), mean_latency=("latency_query_ms", "mean")).reindex(methods)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for idx, method in enumerate(methods):
        row = pareto.loc[method]
        ax.scatter(row["mean_latency"], row["mean_ndcg"], s=70, color=colors[idx], label=METHOD_LABELS[method])
    ax.set_xlabel("Mean latency per query (ms)")
    ax.set_ylabel("Mean nDCG@10")
    ax.set_title("Quality-latency Pareto")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    save(fig, "quality_latency_pareto")
    return files


def contribution_gate(df: pd.DataFrame, contrast_df: pd.DataFrame, formal_summary: dict[str, Any] | None) -> dict[str, Any]:
    selected = "validation_weighted_fusion" if "validation_weighted_fusion" in df["method"].unique() else "rrf"
    selected_rows = df[df["method"] == selected]
    dataset_summary = selected_rows.groupby("dataset").agg(
        mean_ndcg_delta=("ndcg@10_delta_vs_average_single", "mean"),
        mean_mrr_delta=("mrr@10_delta_vs_average_single", "mean"),
        mean_regret=("ndcg_regret_vs_best_single", "mean"),
    )
    positive_datasets = int((dataset_summary["mean_ndcg_delta"] > 0.015).sum())
    n_units = int(selected_rows.groupby(UNIT_COLS).ngroups)
    ndcg_contrast = contrast_df[
        (contrast_df["method"] == selected)
        & (contrast_df["baseline"] == "average_single")
        & (contrast_df["metric"] == "ndcg@10")
    ]
    mrr_contrast = contrast_df[
        (contrast_df["method"] == selected)
        & (contrast_df["baseline"] == "average_single")
        & (contrast_df["metric"] == "mrr@10")
    ]
    ndcg_row = ndcg_contrast.iloc[0].to_dict() if not ndcg_contrast.empty else {}
    mrr_row = mrr_contrast.iloc[0].to_dict() if not mrr_contrast.empty else {}
    formal_pass = not formal_summary or formal_summary.get("formal_execution_status") == "PASS" or formal_summary.get("status") == "PASS"
    ndcg_supported = bool(ndcg_row) and float(ndcg_row.get("mean_delta", 0.0)) > 0.015 and float(ndcg_row.get("ci95_low", -1.0)) > 0.0
    mrr_supported = bool(mrr_row) and float(mrr_row.get("mean_delta", 0.0)) > 0.015 and float(mrr_row.get("ci95_low", -1.0)) > 0.0
    p_supported = any(
        row and row.get("p_value_holm") is not None and float(row["p_value_holm"]) < 0.05
        for row in [ndcg_row, mrr_row]
    )
    ready = formal_pass and n_units >= 80 and positive_datasets >= 3 and (ndcg_supported or mrr_supported) and p_supported
    return {
        "state_label": "PUBLIC_PHASE4_READY"
        if ready
        else "PUBLIC_FORMAL_WEAK_SIGNAL_NEEDS_STRENGTHENING",
        "selected_method": selected,
        "paired_unit": "dataset-budget-seed",
        "paired_unit_count": n_units,
        "positive_dataset_count_ndcg_delta_gt_0_015": positive_datasets,
        "ndcg_average_single_contrast": ndcg_row,
        "mrr_average_single_contrast": mrr_row,
        "criteria": {
            "formal_execution_pass": bool(formal_pass),
            "paired_units_sufficient_for_ei_empirical_paper": n_units >= 80,
            "not_single_dataset_dominated": positive_datasets >= 3,
            "bootstrap_ci_supports_positive_delta": ndcg_supported or mrr_supported,
            "holm_corrected_support": p_supported,
        },
        "recommendation": "Manager may decide whether to authorize Phase 5 writing; Phase 5 was not started."
        if ready
        else "Run a strengthened matrix before Phase 5, or get explicit Manager decision to proceed with weaker descriptive evidence.",
    }


def write_report(
    output_dir: Path,
    formal_summary: dict[str, Any] | None,
    aggregate_files: dict[str, str],
    stat_files: dict[str, str],
    table_files: dict[str, str],
    figure_files: dict[str, list[str]],
    gate: dict[str, Any],
) -> Path:
    path = output_dir / "phase4_analysis_report.md"
    note = formal_summary.get("matrix_note") if formal_summary else None
    lines = [
        "# Phase 4 Analysis Report",
        "",
        f"Status: `{gate['state_label']}`",
        "",
        "## Matrix",
        "",
        f"- formal execution status: `{formal_summary.get('formal_execution_status', formal_summary.get('status')) if formal_summary else 'unknown'}`",
        f"- datasets: `{', '.join(formal_summary.get('config', {}).get('datasets', [])) if formal_summary else 'unknown'}`",
        f"- methods: `{', '.join(formal_summary.get('config', {}).get('methods', [])) if formal_summary else 'unknown'}`",
        f"- expected/actual cells: `{formal_summary.get('config', {}).get('expected_cell_count') if formal_summary else 'unknown'} / {formal_summary.get('config', {}).get('actual_cell_count') if formal_summary else 'unknown'}`",
        f"- failures: `{len(formal_summary.get('failures', [])) if formal_summary else 'unknown'}`",
        "",
    ]
    if note:
        lines.extend(["## Dataset Deviation", "", f"- note status: `{note.get('status')}`", f"- decision: `{note.get('decision')}`", f"- reason: {note.get('reason', 'n/a')}", ""])
    lines.extend(
        [
            "## Statistical Unit",
            "",
            "- Paired unit: dataset-budget-seed cell.",
            "- Query-level confidence intervals are not computed because formal records contain aggregate test metrics, not per-query score files.",
            "- Bootstrap CIs are nonparametric CIs over paired dataset-budget-seed deltas.",
            "",
            "## Contribution Gate",
            "",
            f"- selected method: `{gate['selected_method']}`",
            f"- paired units: `{gate['paired_unit_count']}`",
            f"- positive datasets by nDCG delta > 0.015: `{gate['positive_dataset_count_ndcg_delta_gt_0_015']}`",
            f"- criteria: `{json.dumps(_json_safe(gate['criteria']), sort_keys=True)}`",
            f"- recommendation: {gate['recommendation']}",
            "",
            "## Outputs",
            "",
            f"- aggregate CSV files: `{len(aggregate_files)}`",
            f"- statistical CSV files: `{len(stat_files)}`",
            f"- LaTeX tables: `{len(table_files)}`",
            f"- figures: `{len(figure_files)}` stems, PNG and PDF variants",
            "",
            "Phase 5 writing was not started.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-records", required=True)
    parser.add_argument("--formal-summary", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formal_summary = None
    if args.formal_summary and Path(args.formal_summary).exists():
        formal_summary = json.loads(Path(args.formal_summary).read_text(encoding="utf-8"))

    df = pd.read_csv(args.formal_records)
    for column in METRICS + ["latency_query_ms", "dense_weight"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    enriched = add_baselines(df)
    aggregate_files = aggregate_outputs(enriched, output_dir)
    contrast_df, contrast_files = planned_contrasts(enriched, output_dir)
    static_contrast_df, static_contrast_files = static_baseline_contrasts(enriched, output_dir)
    friedman_df, friedman_files = friedman_tests(enriched, output_dir)
    stat_files = {**contrast_files, **static_contrast_files, **friedman_files}
    table_files = write_tables(enriched, contrast_df, output_dir)
    figure_files = make_figures(enriched, Path(args.figures_dir))
    gate = contribution_gate(enriched, contrast_df, formal_summary)
    report_path = write_report(output_dir, formal_summary, aggregate_files, stat_files, table_files, figure_files, gate)
    payload = {
        "status": gate["state_label"],
        "formal_summary": formal_summary,
        "aggregate_files": aggregate_files,
        "statistical_files": stat_files,
        "table_files": table_files,
        "figure_files": figure_files,
        "contribution_gate": gate,
        "static_baseline_contrasts": static_contrast_df.to_dict(orient="records"),
        "friedman_tests": friedman_df.to_dict(orient="records"),
        "report_md": str(report_path),
    }
    write_json(output_dir / "phase4_analysis_report.json", payload)
    print(json.dumps(_json_safe({"status": gate["state_label"], "report_md": str(report_path), "figures": figure_files}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
