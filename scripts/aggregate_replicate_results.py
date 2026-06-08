#!/usr/bin/env python3
"""Aggregate repeated ClosureBench runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from compare_models import read_jsonl, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_MANIFEST = ROOT / "reports" / "closurebench_replicate_manifest.json"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_replicate_summary.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_replicate_summary.md"


METRICS = [
    ("semantic_switch_accuracy", "Semantic switch"),
    ("core_contrastive_switch_accuracy", "Core switch"),
    ("lcwa_closed_scope_accuracy", "LCWA closed"),
    ("lcwa_open_scope_accuracy", "LCWA open"),
    ("cwa_accuracy", "CWA"),
    ("owa_accuracy", "OWA"),
    ("overall_accuracy", "Overall"),
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
    # t critical for n=3. For larger n, this is conservative enough for this small script.
    t_critical = 4.303 if len(values) == 3 else 1.96
    return round(t_critical * statistics.stdev(values) / math.sqrt(len(values)), 2)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def score_by_id(path: Path) -> dict[str, str | None]:
    return {row["id"]: row.get("model_answer") for row in read_jsonl(path)}


def item_consistency(scored_paths: list[Path], dataset_rows: list[dict]) -> dict:
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
        replicate_stats = []
        scored_paths = []
        for run in sorted(label_runs, key=lambda item: item["repeat"]):
            scored_path = Path(run["scored"])
            scored_paths.append(scored_path)
            stats = summarize(dataset_rows, scored_path)
            stats["repeat"] = run["repeat"]
            stats["scored"] = str(scored_path)
            replicate_stats.append(stats)

        metric_summary = {}
        for metric_key, _ in METRICS:
            values = [stats[metric_key]["accuracy"] for stats in replicate_stats]
            metric_summary[metric_key] = {
                "mean": mean(values),
                "sd": sample_sd(values),
                "ci95_half_width": stderr_95(values),
                "min": round(min(values), 2) if values else 0.0,
                "max": round(max(values), 2) if values else 0.0,
                "values": values,
            }

        out[label] = {
            "model": replicate_stats[0]["models"][0] if replicate_stats and replicate_stats[0]["models"] else "",
            "n_repeats": len(replicate_stats),
            "replicates": replicate_stats,
            "metrics": metric_summary,
            "item_answer_consistency": item_consistency(scored_paths, dataset_rows),
            "mean_parse_failures": mean([stats["parse_failures"] for stats in replicate_stats]),
            "mean_api_errors": mean([stats["api_errors"] for stats in replicate_stats]),
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
        "# ClosureBench Replicate Summary",
        "",
        f"Dataset: `{args.dataset}` ({len(dataset_rows)} items).",
        "",
        "Values are mean +/- sample standard deviation across repeated full-dataset runs.",
        "",
        "| Model | Repeats | Semantic switch | Core switch | LCWA closed | LCWA open | Overall | Item consistency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in summary["runs"].items():
        consistency = stats["item_answer_consistency"]
        lines.append(
            f"| {label} | {stats['n_repeats']} | "
            f"{format_metric(stats, 'semantic_switch_accuracy')} | "
            f"{format_metric(stats, 'core_contrastive_switch_accuracy')} | "
            f"{format_metric(stats, 'lcwa_closed_scope_accuracy')} | "
            f"{format_metric(stats, 'lcwa_open_scope_accuracy')} | "
            f"{format_metric(stats, 'overall_accuracy')} | "
            f"{consistency['rate']}% ({consistency['consistent']}/{consistency['n']}) |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The replicate statistics are intended to address model and provider nondeterminism. They do not replace the main semantic metrics: overall accuracy remains secondary to switch and LCWA-scope metrics.",
        ]
    )
    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
