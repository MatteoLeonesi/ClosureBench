#!/usr/bin/env python3
"""Aggregate repeated ClosureBench-Ask/Act runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_ask_act_results import read_jsonl, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_ask_act.jsonl"
DEFAULT_MANIFEST = ROOT / "reports" / "closurebench_ask_act_replicate_manifest.json"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_ask_act_summary.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_ask_act_summary.md"


METRICS = [
    ("action_accuracy", "Action"),
    ("full_decision_accuracy", "Full decision"),
    ("semantic_switch_action_accuracy", "Switch action"),
    ("core_semantic_switch_action_accuracy", "Core switch action"),
    ("ask_precision", "Ask precision"),
    ("ask_recall", "Ask recall"),
    ("ask_f1", "Ask F1"),
    ("closure_denial_accuracy", "Closure denial"),
    ("unresolved_ask_accuracy", "Unresolved ask"),
    ("excessive_ask_on_closure_false_rate", "Excessive ask closure-false"),
    ("premature_denial_rate", "Premature denial"),
]


def mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def sample_sd(values: list[float]) -> float:
    return round(statistics.stdev(values), 2) if len(values) > 1 else 0.0


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def stderr_95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    t_critical = 4.303 if len(values) == 3 else 1.96
    return round(t_critical * statistics.stdev(values) / math.sqrt(len(values)), 2)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def metric_value(report: dict, key: str) -> float:
    if key == "ask_precision":
        return report["ask_metrics"]["precision"]
    if key == "ask_recall":
        return report["ask_metrics"]["recall"]
    if key == "ask_f1":
        return report["ask_metrics"]["f1"]
    if key in {"excessive_ask_on_closure_false_rate", "premature_denial_rate"}:
        return report[key]["rate"]
    return report[key]["accuracy"]


def score_by_id(path: Path) -> dict[str, tuple[str | None, str | None, str | None, str | None]]:
    return {
        row["id"]: (
            row.get("model_truth_value"),
            row.get("model_action"),
            row.get("model_missing_predicate"),
            row.get("model_missing_entity"),
        )
        for row in read_jsonl(path)
    }


def item_decision_consistency(scored_paths: list[Path], dataset_rows: list[dict]) -> dict:
    if len(scored_paths) <= 1:
        return {"n": len(dataset_rows), "consistent": len(dataset_rows), "rate": 100.0}
    predictions = [score_by_id(path) for path in scored_paths]
    consistent = 0
    for row in dataset_rows:
        values = [pred.get(row["id"]) for pred in predictions]
        if all(value == values[0] for value in values):
            consistent += 1
    return {"n": len(dataset_rows), "consistent": consistent, "rate": pct(consistent, len(dataset_rows))}


def summarize_replicates(dataset_rows: list[dict], runs: list[dict]) -> dict:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_label[run["label"]].append(run)

    out = {}
    for label, label_runs in sorted(by_label.items()):
        replicate_reports = []
        scored_paths = []
        for run in sorted(label_runs, key=lambda item: item["repeat"]):
            scored_path = Path(run["scored"])
            scored_paths.append(scored_path)
            report = summarize(dataset_rows, read_jsonl(scored_path))
            report["repeat"] = run["repeat"]
            report["scored"] = str(scored_path)
            replicate_reports.append(report)

        metric_summary = {}
        for metric_key, _ in METRICS:
            values = [metric_value(report, metric_key) for report in replicate_reports]
            metric_summary[metric_key] = {
                "mean": mean(values),
                "sd": sample_sd(values),
                "ci95_half_width": stderr_95(values),
                "min": round(min(values), 2) if values else 0.0,
                "max": round(max(values), 2) if values else 0.0,
                "values": values,
            }

        out[label] = {
            "model": replicate_reports[0]["models"][0] if replicate_reports and replicate_reports[0]["models"] else "",
            "n_repeats": len(replicate_reports),
            "replicates": replicate_reports,
            "metrics": metric_summary,
            "item_decision_consistency": item_decision_consistency(scored_paths, dataset_rows),
            "mean_parse_failures": mean([report["parse_failures"] for report in replicate_reports]),
            "mean_api_errors": mean([report["api_errors"] for report in replicate_reports]),
        }
    return out


def format_metric(stats: dict, key: str) -> str:
    metric = stats["metrics"][key]
    return f"{metric['mean']:.2f} +/- {metric['sd']:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    manifest = load_manifest(args.manifest)
    summary = {
        "dataset": str(args.dataset),
        "n_dataset": len(dataset_rows),
        "manifest": str(args.manifest),
        "runs": summarize_replicates(dataset_rows, manifest["runs"]),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "# ClosureBench-Ask/Act Replicate Summary",
        "",
        f"Dataset: `{args.dataset}` ({len(dataset_rows)} items).",
        "",
        "Values are mean +/- sample standard deviation across repeated full-dataset runs.",
        "",
        "| Model | Repeats | Action | Core switch action | Ask F1 | Closure denial | Unresolved ask | Excessive ask closure-false | Full decision | Decision consistency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in summary["runs"].items():
        consistency = stats["item_decision_consistency"]
        lines.append(
            f"| {label} | {stats['n_repeats']} | "
            f"{format_metric(stats, 'action_accuracy')} | "
            f"{format_metric(stats, 'core_semantic_switch_action_accuracy')} | "
            f"{format_metric(stats, 'ask_f1')} | "
            f"{format_metric(stats, 'closure_denial_accuracy')} | "
            f"{format_metric(stats, 'unresolved_ask_accuracy')} | "
            f"{format_metric(stats, 'excessive_ask_on_closure_false_rate')} | "
            f"{format_metric(stats, 'full_decision_accuracy')} | "
            f"{consistency['rate']}% ({consistency['consistent']}/{consistency['n']}) |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "ClosureBench-Ask/Act preserves the base facts, rules, targets, semantic contracts, and truth labels. The added variable is the operational action layer: true -> approve, false -> deny, unknown -> request_information.",
            "",
            "The most safety-relevant rates are excessive ask under closure-false cases and premature denial under unresolved open-world cases.",
        ]
    )
    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
