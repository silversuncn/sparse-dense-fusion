"""Formal matrix runner for sparse-dense retrieval fusion."""

from __future__ import annotations

import argparse
import json
import traceback
import time
from pathlib import Path

from fusion_pipeline import DEFAULT_METHODS, run_pilot_matrix, set_offline_env, write_csv, write_json, write_jsonl


def _read_json(path: str | None):
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return {"status": "missing", "path": str(target)}
    return json.loads(target.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa", "arguana"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 31, 47, 73, 109, 149, 181])
    parser.add_argument("--max-queries", type=int, default=1_000_000)
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--offline-models", action="store_true")
    parser.add_argument("--allow-fetch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--matrix-note-json", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    exit_path = output_dir / "formal.exitcode"
    started = time.time()
    set_offline_env(args.offline_models)
    try:
        result = run_pilot_matrix(
            data_root=Path(args.data_root),
            datasets=args.datasets,
            budgets=args.budgets,
            methods=args.methods,
            seeds=args.seeds,
            max_queries=args.max_queries,
            dense_model=args.dense_model,
            offline_models=args.offline_models,
            allow_fetch=args.allow_fetch,
            batch_size=args.batch_size,
        )
        result["elapsed_sec"] = time.time() - started
        rows = result.pop("rows")
        expected = int(result["config"]["expected_cell_count"])
        actual = int(result["config"]["actual_cell_count"])
        execution_pass = actual == expected and not result["failures"]
        result["formal_execution_status"] = "PASS" if execution_pass else "FAILED"
        result["pilot_style_quality_gate_status"] = result["status"]
        result["status"] = result["formal_execution_status"]
        result["matrix_note"] = _read_json(args.matrix_note_json)
        result["artifact_files"] = {
            "records_jsonl": str(output_dir / "formal_records.jsonl"),
            "records_csv": str(output_dir / "formal_records.csv"),
            "summary_json": str(output_dir / "formal_summary.json"),
            "gate_json": str(output_dir / "formal_gate.json"),
            "quality_report_json": str(output_dir / "formal_quality_report.json"),
            "pipeline_log": str(log_path),
            "formal_exitcode": str(exit_path),
        }
        quality_report = {
            "formal_execution_status": result["formal_execution_status"],
            "expected_cell_count": expected,
            "actual_cell_count": actual,
            "failure_count": len(result["failures"]),
            "pilot_style_quality_gate_pass": bool(result["gate"]["pass"]),
            "selected_gate_method": result["gate"]["selected_gate_method"],
            "criteria": result["gate"]["criteria"],
            "severe_degradation_datasets": result["gate"]["severe_degradation_datasets"],
            "matrix_note": result["matrix_note"],
        }
        write_jsonl(output_dir / "formal_records.jsonl", rows)
        write_csv(output_dir / "formal_records.csv", rows)
        write_json(output_dir / "formal_summary.json", result)
        write_json(output_dir / "formal_gate.json", result["gate"])
        write_json(output_dir / "formal_quality_report.json", quality_report)
        compact = {
            "status": result["status"],
            "expected_cell_count": expected,
            "actual_cell_count": actual,
            "quality_gate_pass": bool(result["gate"]["pass"]),
            "selected_gate_method": result["gate"]["selected_gate_method"],
            "elapsed_sec": result["elapsed_sec"],
            "failures": result["failures"],
            "datasets": list(args.datasets),
            "methods": list(args.methods),
            "budgets": list(args.budgets),
            "seeds": list(args.seeds),
        }
        log_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exit_path.write_text("0\n" if execution_pass else "1\n", encoding="utf-8")
        print(json.dumps(compact, indent=2, sort_keys=True))
        return 0 if execution_pass else 1
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        }
        log_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        exit_path.write_text("1\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
