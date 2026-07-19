"""Bounded pilot runner for sparse-dense fusion."""

from __future__ import annotations

import argparse
import json
import traceback
import time
from pathlib import Path

from fusion_pipeline import DEFAULT_METHODS, run_pilot_matrix, set_offline_env, write_csv, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 31, 47])
    parser.add_argument("--max-queries", type=int, default=96)
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--offline-models", action="store_true")
    parser.add_argument("--allow-fetch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    exit_path = output_dir / "pilot.exitcode"
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
        write_jsonl(output_dir / "pilot_records.jsonl", rows)
        write_csv(output_dir / "pilot_records.csv", rows)
        result["artifact_files"] = {
            "records_jsonl": str(output_dir / "pilot_records.jsonl"),
            "records_csv": str(output_dir / "pilot_records.csv"),
            "summary_json": str(output_dir / "pilot_summary.json"),
            "gate_json": str(output_dir / "pilot_gate.json"),
            "pipeline_log": str(log_path),
            "pilot_exitcode": str(exit_path),
        }
        write_json(output_dir / "pilot_summary.json", result)
        write_json(output_dir / "pilot_gate.json", result["gate"])
        compact = {
            "status": result["status"],
            "expected_cell_count": result["config"]["expected_cell_count"],
            "actual_cell_count": result["config"]["actual_cell_count"],
            "gate_pass": result["gate"]["pass"],
            "selected_gate_method": result["gate"]["selected_gate_method"],
            "criteria": result["gate"]["criteria"],
            "elapsed_sec": result["elapsed_sec"],
            "failures": result["failures"],
        }
        log_path.write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exit_path.write_text("0\n", encoding="utf-8")
        print(json.dumps(compact, indent=2, sort_keys=True))
        return 0
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
