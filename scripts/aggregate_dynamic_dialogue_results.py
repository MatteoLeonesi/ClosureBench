#!/usr/bin/env python3
"""Aggregate repeated ClosureBench-Dyn runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_dynamic_dialogue_results import read_jsonl, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_dynamic_dialogue.jsonl"
DEFAULT_MANIFEST = ROOT / "reports" / "closurebench_dynamic_dialogue_replicate_manifest.json"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_dynamic_dialogue_summary.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_dynamic_dialogue_summary.md"


METRICS = [
    ("dialogue_all_answer_accuracy", "Dialogue all-answer"),
    ("dialogue_all_turn_accuracy", "Dialogue all-turn full"),
    ("answer_accuracy", "Turn answer"),
    ("revision_accuracy", "Revision"),
    ("persistence_accuracy", "Persistence"),
    ("completeness_addition_accuracy", "Completeness addition"),
    ("reopening_accuracy", "Reopening"),
    ("narrowing_persistence_accuracy", "Narrowing persistence"),
    ("changed_flag_accuracy", "Changed flag"),
    ("applied_update_accuracy", "Applied update"),
    ("closure_inertia_rate", "Closure inertia"),
    ("closure_retention_failure_rate", "Retention failure"),
    ("anchoring_error_rate", "Anchoring error"),
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
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(report: dict, key: str) -> float:
    metric = report[key]
    if "rate" in metric:
        return metric["rate"]
    return metric["accuracy"]


def score_by_id(path: Path) -> dict[str, tuple[str | None, bool | None, str | None]]:
    return {
        row["id"]: (
            row.get("model_answer"),
            row.get("changed_from_previous"),
            row.get("applied_update"),
        )
        for row in read_jsonl(path)
    }


def turn_ids(dialogues: list[dict]) -> list[str]:
    ids = []
    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            ids.append(f"{dialogue['id']}__turn_{turn['turn_index']}")
    return ids


def turn_decision_consistency(scored_paths: list[Path], dialogues: list[dict]) -> dict:
    ids = turn_ids(dialogues)
    if len(scored_paths) <= 1:
        return {"n": len(ids), "consistent": len(ids), "rate": 100.0}
    predictions = [score_by_id(path) for path in scored_paths]
    consistent = 0
    for row_id in ids:
        values = [pred.get(row_id) for pred in predictions]
        if all(value == values[0] for value in values):
            consistent += 1
    return {"n": len(ids), "consistent": consistent, "rate": pct(consistent, len(ids))}


def summarize_replicates(dialogues: list[dict], runs: list[dict]) -> dict:
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
            report = summarize(dialogues, read_jsonl(scored_path))
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
            "turn_decision_consistency": turn_decision_consistency(scored_paths, dialogues),
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

    dialogues = read_jsonl(args.dataset)
    manifest = load_manifest(args.manifest)
    summary = {
        "dataset": str(args.dataset),
        "n_dialogues": len(dialogues),
        "n_turns": sum(len(dialogue["turns"]) for dialogue in dialogues),
        "manifest": str(args.manifest),
        "runs": summarize_replicates(dialogues, manifest["runs"]),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# ClosureBench-Dyn Replicate Summary",
        "",
        f"Dataset: `{args.dataset}` ({summary['n_dialogues']} dialogues, {summary['n_turns']} turns).",
        "",
        "Values are mean +/- sample standard deviation across repeated full-dataset runs.",
        "",
        "| Model | Repeats | All-answer dialogues | Revision | Persistence | Retention failure | Reopening | Closure inertia | Changed flag | Full-turn dialogues | Turn consistency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in summary["runs"].items():
        consistency = stats["turn_decision_consistency"]
        lines.append(
            f"| {label} | {stats['n_repeats']} | "
            f"{format_metric(stats, 'dialogue_all_answer_accuracy')} | "
            f"{format_metric(stats, 'revision_accuracy')} | "
            f"{format_metric(stats, 'persistence_accuracy')} | "
            f"{format_metric(stats, 'closure_retention_failure_rate')} | "
            f"{format_metric(stats, 'reopening_accuracy')} | "
            f"{format_metric(stats, 'closure_inertia_rate')} | "
            f"{format_metric(stats, 'changed_flag_accuracy')} | "
            f"{format_metric(stats, 'dialogue_all_turn_accuracy')} | "
            f"{consistency['rate']}% ({consistency['consistent']}/{consistency['n']}) |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "ClosureBench-Dyn keeps the facts, rules, target, and glossary fixed across a three-turn dialogue. Only the set of complete predicates changes.",
            "",
            "Revision accuracy measures transition turns where the gold answer changes. Persistence accuracy measures transition turns where the gold answer should stay the same.",
            "",
            "Retention failure counts local-narrowing persistence turns where the model answers `unknown` even though the target predicate remains complete and the gold answer remains `false`. Closure inertia counts reopen turns where the model keeps `false` after the completeness basis for that false conclusion has been withdrawn.",
        ]
    )
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
