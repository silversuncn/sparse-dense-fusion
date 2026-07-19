"""Validate and summarize the public sparse-dense fusion artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    "formal_rows": 672,
    "datasets": {"arguana", "fiqa", "nfcorpus", "scifact"},
    "budgets": {"20", "50", "100", "200"},
    "methods": {
        "bm25",
        "dense",
        "rrf",
        "zscore_fusion",
        "minmax_fusion",
        "validation_weighted_fusion",
    },
    "seeds": {"19", "31", "47", "73", "109", "149", "181"},
    "paired_units": 112,
    "ndcg_delta": 0.05122960449440239,
    "mrr_delta": 0.055090008998786556,
    "holm_p": 6.55211033727735e-19,
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(data_dir: Path) -> dict[str, object]:
    rows = _read_csv(data_dir / "formal_records.csv")
    configs = {(r["dataset"], r["budget"], r["method"], r["seed"]) for r in rows}
    summary = _read_json(data_dir / "formal_summary.json")
    gate = _read_json(data_dir / "formal_gate.json")
    phase4 = _read_json(data_dir / "phase4_analysis_report.json")
    contribution = phase4["contribution_gate"]
    ndcg = contribution["ndcg_average_single_contrast"]
    mrr = contribution["mrr_average_single_contrast"]

    payload = {
        "formal_rows": len(rows),
        "unique_configs": len(configs),
        "datasets": sorted({r["dataset"] for r in rows}),
        "budgets": sorted({r["budget"] for r in rows}, key=int),
        "methods": sorted({r["method"] for r in rows}),
        "seeds": sorted({r["seed"] for r in rows}, key=int),
        "summary_status": summary["status"],
        "expected_cell_count": summary["config"]["expected_cell_count"],
        "actual_cell_count": summary["config"]["actual_cell_count"],
        "failure_count": len(summary["failures"]),
        "gate_pass": bool(gate["pass"]),
        "selected_gate_method": gate["selected_gate_method"],
        "method_reduces_average_regret_cells": gate["method_reduces_average_regret_cells"],
        "paired_units": contribution["paired_unit_count"],
        "positive_dataset_count": contribution["positive_dataset_count_ndcg_delta_gt_0_015"],
        "ndcg_delta_vs_average_single": ndcg["mean_delta"],
        "ndcg_ci95": [ndcg["ci95_low"], ndcg["ci95_high"]],
        "mrr_delta_vs_average_single": mrr["mean_delta"],
        "mrr_ci95": [mrr["ci95_low"], mrr["ci95_high"]],
        "holm_p": ndcg["p_value_holm"],
    }
    return payload


def validate(summary: dict[str, object]) -> None:
    if summary["formal_rows"] != EXPECTED["formal_rows"]:
        raise AssertionError(summary["formal_rows"])
    if summary["unique_configs"] != EXPECTED["formal_rows"]:
        raise AssertionError(summary["unique_configs"])
    for key in ["datasets", "budgets", "methods", "seeds"]:
        if set(summary[key]) != EXPECTED[key]:
            raise AssertionError((key, summary[key]))
    if summary["summary_status"] != "PASS":
        raise AssertionError(summary["summary_status"])
    if summary["expected_cell_count"] != EXPECTED["formal_rows"]:
        raise AssertionError(summary["expected_cell_count"])
    if summary["actual_cell_count"] != EXPECTED["formal_rows"]:
        raise AssertionError(summary["actual_cell_count"])
    if summary["failure_count"] != 0:
        raise AssertionError(summary["failure_count"])
    if not summary["gate_pass"]:
        raise AssertionError("gate failed")
    if summary["selected_gate_method"] != "validation_weighted_fusion":
        raise AssertionError(summary["selected_gate_method"])
    for method, count in summary["method_reduces_average_regret_cells"].items():
        if method in {"rrf", "zscore_fusion", "minmax_fusion", "validation_weighted_fusion"} and count != 112:
            raise AssertionError((method, count))
    if summary["paired_units"] != EXPECTED["paired_units"]:
        raise AssertionError(summary["paired_units"])
    if summary["positive_dataset_count"] != 4:
        raise AssertionError(summary["positive_dataset_count"])
    for key, expected in [("ndcg_delta_vs_average_single", EXPECTED["ndcg_delta"]), ("mrr_delta_vs_average_single", EXPECTED["mrr_delta"]), ("holm_p", EXPECTED["holm_p"])]:
        if abs(float(summary[key]) - expected) > 1e-12:
            raise AssertionError((key, summary[key], expected))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    payload = summarize(Path(args.data_dir))
    validate(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
