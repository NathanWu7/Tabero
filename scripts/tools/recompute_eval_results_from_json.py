#!/usr/bin/env python3
"""Recompute TacManip evaluation TXT summaries from saved JSON results.

Why this exists:
- Older runs may have saved per-task `success_rate` correctly, but marked `status="failed"`
  due to a non-zero subprocess return code (often teardown issues).
- The original TXT writer only computed Suite/Overall averages for `status=="completed"`,
  so those runs printed `N/A` even though per-task success rates/metrics existed.

This script reads `evaluation_results/success_rates_*.json` and writes a new TXT file with:
- the same formatting as `scripts/tools/run_task_evaluations.py::save_success_rates_txt`
  (including Suite/Overall summary and the "=== metric (all tasks) ===" ascii table)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class TaskResult:
    task_suite: str
    task_id: int
    task_name: str
    language_instruction: str
    success_rate: Optional[float]  # percent, e.g. 54.0
    successful_experiments: Optional[int]
    total_experiments: Optional[int]
    execution_time: Optional[float]
    status: str

    # Hybrid metrics (may be None)
    avg_squeeze_pred: Optional[float]
    avg_squeeze_meas: Optional[float]
    task_squeeze_max_mean: Optional[float]
    task_squeeze_max_meas_mean: Optional[float]
    task_app_mean_mean: Optional[float]
    task_ap_mean_meas_mean: Optional[float]
    task_app_max_mean: Optional[float]
    task_ap_max_meas_mean: Optional[float]


def _load_results(json_path: Path) -> tuple[dict[str, Any], list[TaskResult]]:
    data = json.loads(json_path.read_text())
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    results_dict = data.get("results", {}) if isinstance(data, dict) else {}

    out: list[TaskResult] = []
    if isinstance(results_dict, dict):
        for _, r in results_dict.items():
            if not isinstance(r, dict):
                continue
            out.append(
                TaskResult(
                    task_suite=str(r.get("task_suite", "")),
                    task_id=int(r.get("task_id", -1)),
                    task_name=str(r.get("task_name", "")),
                    language_instruction=str(r.get("language_instruction", "")),
                    success_rate=r.get("success_rate", None),
                    successful_experiments=r.get("successful_experiments", None),
                    total_experiments=r.get("total_experiments", None),
                    execution_time=r.get("execution_time", None),
                    status=str(r.get("status", "")),
                    avg_squeeze_pred=r.get("avg_squeeze_pred", None),
                    avg_squeeze_meas=r.get("avg_squeeze_meas", None),
                    task_squeeze_max_mean=r.get("task_squeeze_max_mean", None),
                    task_squeeze_max_meas_mean=r.get("task_squeeze_max_meas_mean", None),
                    task_app_mean_mean=r.get("task_app_mean_mean", None),
                    task_ap_mean_meas_mean=r.get("task_ap_mean_meas_mean", None),
                    task_app_max_mean=r.get("task_app_max_mean", None),
                    task_ap_max_meas_mean=r.get("task_ap_max_meas_mean", None),
                )
            )
    return meta, out


def _fmt(value: Optional[float], digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _suite_short(suite: str) -> str:
    if isinstance(suite, str) and suite.startswith("libero_"):
        return suite.split("libero_", 1)[1]
    return suite


def _parse_timestamp(meta_ts: Any) -> str:
    """Match run_task_evaluations.py timestamp formatting: '%Y-%m-%d %H:%M:%S'."""
    if not meta_ts:
        return ""
    ts = str(meta_ts)
    # Accept ISO strings like '2026-01-09T04:06:15.460026' or with 'Z'
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _write_metrics_table_like_run_task_evaluations(f, results: list[TaskResult]) -> None:
    """Write the same table format as run_task_evaluations.py::save_success_rates_txt."""
    rows: list[list[str]] = []
    for r in sorted(results, key=lambda x: (x.task_suite, x.task_id)):
        if r.status != "completed":
            continue

        # success_rate 以 0.x 形式展示
        sr_display = (r.success_rate / 100.0) if r.success_rate is not None else None

        rows.append(
            [
                f"{_suite_short(r.task_suite)} {r.task_id}",
                _fmt(sr_display, 2),
                _fmt(r.avg_squeeze_pred, 4),
                _fmt(r.avg_squeeze_meas, 4),
                _fmt(r.task_squeeze_max_mean, 4),
                _fmt(r.task_squeeze_max_meas_mean, 4),
                _fmt(r.task_app_mean_mean, 4),
                _fmt(r.task_ap_mean_meas_mean, 4),
                _fmt(r.task_app_max_mean, 4),
                _fmt(r.task_ap_max_meas_mean, 4),
            ]
        )

    if not rows:
        return

    headers = [
        "task id",
        "success_rate",
        "squeeze_avg_pred",
        "squeeze_avg_meas",
        "squeeze_max_pred",
        "squeeze_max_meas",
        "ap_avg_pred",
        "ap_avg_meas",
        "ap_max_pred",
        "ap_max_meas",
    ]

    cols = list(zip(headers, *rows))
    col_widths = [max(len(str(cell)) for cell in col) + 2 for col in cols]  # 两侧各空 1 格

    def _hline() -> str:
        return "+" + "+".join("-" * w for w in col_widths) + "+"

    def _format_row(cells: list[str]) -> str:
        padded: list[str] = []
        for cell, width in zip(cells, col_widths):
            cell_str = str(cell)
            space = width - len(cell_str)
            left = space // 2
            right = space - left
            padded.append(" " * left + cell_str + " " * right)
        return "|" + "|".join(padded) + "|"

    f.write("\n=== metric (all tasks) ===\n\n")
    f.write(_hline() + "\n")
    f.write(_format_row(headers) + "\n")
    f.write(_hline() + "\n")
    for row in rows:
        f.write(_format_row(row) + "\n")
    f.write(_hline() + "\n")


def recompute_txt(json_path: Path, output_path: Path) -> None:
    meta, results = _load_results(json_path)
    if not results:
        raise SystemExit(f"No results found in: {json_path}")

    # Re-mark tasks as completed if the evaluation stats were successfully parsed for the full run.
    # This mirrors the newer run_task_evaluations.py behavior.
    num_total_experiments = meta.get("num_total_experiments", None)
    remapped: list[TaskResult] = []
    for r in results:
        parsed_full_eval = (
            (r.success_rate is not None)
            and (r.successful_experiments is not None)
            and (r.total_experiments is not None)
            and (num_total_experiments is not None)
            and (int(r.total_experiments) == int(num_total_experiments))
        )
        status = "completed" if parsed_full_eval else r.status
        remapped.append(TaskResult(**{**r.__dict__, "status": status}))
    results = remapped

    # Group by suite
    suites: dict[str, list[TaskResult]] = {}
    for r in results:
        suites.setdefault(r.task_suite, []).append(r)

    # Suite averages: same logic as run_task_evaluations.py::save_success_rates_txt
    suite_avgs: dict[str, Optional[float]] = {}
    for suite, suite_results in suites.items():
        valid = [r for r in suite_results if r.status == "completed" and r.success_rate is not None]
        suite_avgs[suite] = (sum(r.success_rate for r in valid) / len(valid)) if valid else None

    overall_valid_suite_avgs = [v for v in suite_avgs.values() if v is not None]
    overall_avg = (sum(overall_valid_suite_avgs) / len(overall_valid_suite_avgs)) if overall_valid_suite_avgs else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("POLICY EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Match the exact configuration block style used by run_task_evaluations.py
        f.write("Configuration:\n")
        policy_model = meta.get("policy_model", "openpi")
        control_mode = meta.get("control_mode", "tactile")
        num_total = meta.get("num_total_experiments", "")
        num_steps = meta.get("num_success_steps", "")
        max_steps = ""
        with_steps = meta.get("max_inference_steps_policy", {})
        if isinstance(with_steps, dict):
            max_steps = with_steps.get("default", "")
        ts = _parse_timestamp(meta.get("timestamp", ""))

        f.write(f"  Policy Model: {policy_model}\n")
        if policy_model == "openpi":
            f.write(f"  Control Mode: {control_mode}\n")
        f.write(f"  Experiments per task: {num_total}\n")
        f.write(f"  Success steps required: {num_steps}\n")
        f.write(f"  Max inference steps: {max_steps}\n")
        f.write(f"  Timestamp: {ts}\n")
        f.write("\n" + "=" * 60 + "\n\n")

        for suite in sorted(suites.keys()):
            suite_results = sorted(suites[suite], key=lambda x: x.task_id)
            f.write(f"{suite.upper()}:\n")
            f.write("-" * 40 + "\n")

            for r in suite_results:
                if r.success_rate is None:
                    f.write(f"  Task {r.task_id} ({r.task_name}): N/A ({r.status})\n")
                    continue

                successful = r.successful_experiments if r.successful_experiments is not None else "?"
                total = r.total_experiments if r.total_experiments is not None else "?"
                exec_time = r.execution_time if r.execution_time is not None else 0.0
                f.write(
                    f"  Task {r.task_id} ({r.task_name}): {r.success_rate:.2f}% ({successful}/{total}) [{exec_time:.1f}s]\n"
                )

                # Hybrid metrics block if any metric exists
                metrics_present = any(
                    v is not None
                    for v in (
                        r.avg_squeeze_pred,
                        r.avg_squeeze_meas,
                        r.task_squeeze_max_mean,
                        r.task_squeeze_max_meas_mean,
                        r.task_app_mean_mean,
                        r.task_ap_mean_meas_mean,
                        r.task_app_max_mean,
                        r.task_ap_max_meas_mean,
                    )
                )
                if metrics_present:
                    f.write("    Hybrid metrics (success only, task-level mean over experiments):\n")
                    # Follow run_task_evaluations.py field order and spacing.
                    f.write("      ")
                    if r.avg_squeeze_pred is not None:
                        f.write(f"squeeze_avg_pred={r.avg_squeeze_pred:.4f} ")
                    else:
                        f.write("squeeze_avg_pred=N/A ")
                    if r.avg_squeeze_meas is not None:
                        f.write(f"squeeze_avg_meas={r.avg_squeeze_meas:.4f}")
                    else:
                        f.write("squeeze_avg_meas=N/A")
                    f.write("\n")

                    f.write("      ")
                    if r.task_squeeze_max_mean is not None:
                        f.write(f"squeeze_max_pred={r.task_squeeze_max_mean:.4f} ")
                    else:
                        f.write("squeeze_max_pred=N/A ")
                    if r.task_squeeze_max_meas_mean is not None:
                        f.write(f"squeeze_max_meas={r.task_squeeze_max_meas_mean:.4f}")
                    else:
                        f.write("squeeze_max_meas=N/A")
                    f.write("\n")

                    f.write("      ")
                    if r.task_app_mean_mean is not None:
                        f.write(f"ap_avg_pred={r.task_app_mean_mean:.4f} ")
                    else:
                        f.write("ap_avg_pred=N/A ")
                    if r.task_ap_mean_meas_mean is not None:
                        f.write(f"ap_avg_meas={r.task_ap_mean_meas_mean:.4f}")
                    else:
                        f.write("ap_avg_meas=N/A")
                    f.write("\n")

                    f.write("      ")
                    if r.task_app_max_mean is not None:
                        f.write(f"ap_max_pred={r.task_app_max_mean:.4f} ")
                    else:
                        f.write("ap_max_pred=N/A ")
                    if r.task_ap_max_meas_mean is not None:
                        f.write(f"ap_max_meas={r.task_ap_max_meas_mean:.4f}")
                    else:
                        f.write("ap_max_meas=N/A")
                    f.write("\n")

            f.write("\n")
            avg = suite_avgs.get(suite)
            if avg is None:
                f.write("  Suite Average: N/A (no completed tasks)\n\n")
            else:
                completed_count = len([r for r in suite_results if r.status == "completed"])
                f.write(f"  Suite Average: {avg:.2f}% ({completed_count}/{len(suite_results)} tasks completed)\n\n")

        f.write("=" * 60 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 60 + "\n")
        for suite in sorted(suite_avgs.keys()):
            avg = suite_avgs[suite]
            if avg is None:
                f.write(f"  {suite}: N/A\n")
            else:
                f.write(f"  {suite}: {avg:.2f}%\n")
        f.write("\n")
        if overall_avg is None:
            f.write("  Overall Average: N/A\n")
        else:
            f.write(f"  Overall Average: {overall_avg:.2f}% ({len(overall_valid_suite_avgs)}/{len(suite_avgs)} suites)\n")
        f.write("=" * 60 + "\n")

        _write_metrics_table_like_run_task_evaluations(f, results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, nargs="+", required=True, help="Path(s) to success_rates_*.json")
    ap.add_argument("--output_dir", type=str, default="", help="Optional output directory. Default: alongside input.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    for inp in args.input:
        json_path = Path(inp).expanduser().resolve()
        if not json_path.exists():
            raise SystemExit(f"Not found: {json_path}")
        if out_dir is None:
            output_path = json_path.with_suffix("").with_name(json_path.stem + "_recomputed.txt")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / (json_path.stem + "_recomputed.txt")
        recompute_txt(json_path, output_path)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
