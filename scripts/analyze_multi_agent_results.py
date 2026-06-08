#!/usr/bin/env python3
"""Analyze ClosureBench-MA scored outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_multi_agent.jsonl"
DEFAULT_SCORED = ROOT / "results" / "closurebench_multi_agent_scored.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_multi_agent_report.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_multi_agent_report.md"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def metric(rows: list[dict], key: str) -> dict:
    correct = sum(1 for row in rows if row.get(key))
    return {"correct": correct, "n": len(rows), "accuracy": pct(correct, len(rows))}


def rate(rows: list[dict], predicate) -> dict:
    count = sum(1 for row in rows if predicate(row))
    return {"count": count, "n": len(rows), "rate": pct(count, len(rows))}


def grouped_metric(rows: list[dict], group_key: str, metric_key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return {name: metric(items, metric_key) for name, items in sorted(grouped.items())}


def grouped_triple_metric(rows: list[dict], group_key: str) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    return {
        name: {
            "n": len(items),
            "truth": metric(items, "truth_correct")["accuracy"],
            "source": metric(items, "source_correct")["accuracy"],
            "closure": metric(items, "closure_correct")["accuracy"],
            "full": metric(items, "correct")["accuracy"],
        }
        for name, items in sorted(grouped.items())
    }


def switch_accuracy(rows: list[dict], key: str, subset: str | None = None) -> dict:
    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if subset is not None and row.get("subset") != subset:
            continue
        by_base[row["base_id"]].append(row)
    triples = [items for items in by_base.values() if len({item["semantics"] for item in items}) == 3]
    correct = sum(1 for items in triples if all(item.get(key) for item in items))
    return {"correct": correct, "n": len(triples), "accuracy": pct(correct, len(triples))}


def classify_error(row: dict) -> str:
    if row.get("error"):
        return "api_error"
    if row.get("model_truth_value") is None and row.get("model_closure_handling") is None:
        return "parse_failure"
    if row.get("correct"):
        return "correct"
    if row["gold_closure_handling"] == "open_absence" and (
        row.get("model_closure_handling") == "closed_absence" or row.get("model_truth_value") == "false"
    ):
        return "closure_leakage_open_treated_closed"
    if row["gold_closure_handling"] == "closed_absence" and (
        row.get("model_closure_handling") == "open_absence" or row.get("model_truth_value") == "unknown"
    ):
        return "closure_loss_closed_treated_open"
    if row.get("truth_correct") and not row.get("source_correct"):
        return "truth_correct_source_wrong"
    if row.get("truth_correct") and not row.get("closure_correct"):
        return "truth_correct_closure_wrong"
    if not row.get("truth_correct"):
        return "truth_wrong"
    return "other_attribution_error"


def source_matches(model_source: str | None, gold_source: str, gold_closure_handling: str) -> bool:
    if model_source == gold_source:
        return True
    return gold_closure_handling == "open_absence" and model_source in {"agent_b", "neither"}


def summarize(dataset_rows: list[dict], scored_rows: list[dict]) -> dict:
    dataset_by_id = {row["id"]: row for row in dataset_rows}
    rows = []
    for scored in scored_rows:
        if scored["id"] in dataset_by_id:
            merged = dict(dataset_by_id[scored["id"]])
            merged.update(scored)
            dataset = dataset_by_id[scored["id"]]
            merged["gold_truth_value"] = dataset["gold_truth_value"]
            merged["gold_source_used"] = dataset["gold_source_used"]
            merged["gold_closure_handling"] = dataset["gold_closure_handling"]
            merged["truth_correct"] = merged.get("model_truth_value") == merged["gold_truth_value"]
            merged["source_correct"] = source_matches(
                merged.get("model_source_used"),
                merged["gold_source_used"],
                merged["gold_closure_handling"],
            )
            merged["closure_correct"] = merged.get("model_closure_handling") == merged["gold_closure_handling"]
            merged["correct"] = merged["truth_correct"] and merged["source_correct"] and merged["closure_correct"]
            rows.append(merged)
    for row in rows:
        row["error_type"] = classify_error(row)

    closed_absence = [row for row in rows if row["gold_closure_handling"] == "closed_absence"]
    open_absence = [row for row in rows if row["gold_closure_handling"] == "open_absence"]

    return {
        "n_dataset": len(dataset_rows),
        "n_scored": len(rows),
        "coverage": pct(len(rows), len(dataset_rows)),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "api_errors": sum(1 for row in rows if row.get("error")),
        "parse_failures": sum(
            1
            for row in rows
            if row.get("model_truth_value") is None
            or row.get("model_source_used") is None
            or row.get("model_closure_handling") is None
        ),
        "json_valid": {
            "n": sum(1 for row in rows if row.get("json_valid")),
            "rate": pct(sum(1 for row in rows if row.get("json_valid")), len(rows)),
        },
        "truth_accuracy": metric(rows, "truth_correct"),
        "source_accuracy": metric(rows, "source_correct"),
        "closure_accuracy": metric(rows, "closure_correct"),
        "full_accuracy": metric(rows, "correct"),
        "semantic_switch_full_accuracy": switch_accuracy(rows, "correct"),
        "core_switch_full_accuracy": switch_accuracy(rows, "correct", "core_contrastive"),
        "closed_absence_accuracy": metric(closed_absence, "correct"),
        "open_absence_accuracy": metric(open_absence, "correct"),
        "closure_leakage_rate": rate(
            open_absence,
            lambda row: row.get("model_closure_handling") == "closed_absence" or row.get("model_truth_value") == "false",
        ),
        "closure_loss_rate": rate(
            closed_absence,
            lambda row: row.get("model_closure_handling") == "open_absence" or row.get("model_truth_value") == "unknown",
        ),
        "agent_a_source_accuracy": metric([row for row in rows if row["gold_source_used"] == "agent_a"], "source_correct"),
        "agent_b_source_accuracy": metric([row for row in rows if row["gold_source_used"] == "agent_b"], "source_correct"),
        "by_semantics": grouped_triple_metric(rows, "semantics"),
        "by_closure_handling": grouped_triple_metric(rows, "gold_closure_handling"),
        "by_source_used": grouped_metric(rows, "gold_source_used", "source_correct"),
        "by_family": grouped_metric(rows, "family", "correct"),
        "error_types": dict(Counter(row["error_type"] for row in rows)),
    }


def write_markdown(path: Path, report: dict) -> None:
    model = ", ".join(report["models"]) if report["models"] else "none"
    lines = [
        "# ClosureBench-MA Report",
        "",
        f"Model: {model}.",
        f"Scored items: {report['n_scored']}/{report['n_dataset']} ({report['coverage']}%).",
        f"Full accuracy: {report['full_accuracy']['accuracy']}% "
        f"({report['full_accuracy']['correct']}/{report['full_accuracy']['n']}).",
        f"Truth accuracy: {report['truth_accuracy']['accuracy']}% "
        f"({report['truth_accuracy']['correct']}/{report['truth_accuracy']['n']}).",
        f"Source accuracy: {report['source_accuracy']['accuracy']}% "
        f"({report['source_accuracy']['correct']}/{report['source_accuracy']['n']}).",
        f"Closure accuracy: {report['closure_accuracy']['accuracy']}% "
        f"({report['closure_accuracy']['correct']}/{report['closure_accuracy']['n']}).",
        f"Semantic-switch full accuracy: {report['semantic_switch_full_accuracy']['accuracy']}% "
        f"({report['semantic_switch_full_accuracy']['correct']}/{report['semantic_switch_full_accuracy']['n']}).",
        f"Core-switch full accuracy: {report['core_switch_full_accuracy']['accuracy']}% "
        f"({report['core_switch_full_accuracy']['correct']}/{report['core_switch_full_accuracy']['n']}).",
        f"Closed-absence accuracy: {report['closed_absence_accuracy']['accuracy']}% "
        f"({report['closed_absence_accuracy']['correct']}/{report['closed_absence_accuracy']['n']}).",
        f"Open-absence accuracy: {report['open_absence_accuracy']['accuracy']}% "
        f"({report['open_absence_accuracy']['correct']}/{report['open_absence_accuracy']['n']}).",
        f"Closure leakage rate: {report['closure_leakage_rate']['rate']}% "
        f"({report['closure_leakage_rate']['count']}/{report['closure_leakage_rate']['n']}).",
        f"Closure loss rate: {report['closure_loss_rate']['rate']}% "
        f"({report['closure_loss_rate']['count']}/{report['closure_loss_rate']['n']}).",
        f"Strict JSON-valid response rate: {report['json_valid']['rate']}%.",
        "",
        "## By Semantics",
        "",
        "| Semantics | Truth | Source | Closure | Full | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["by_semantics"].items():
        lines.append(f"| {name} | {values['truth']} | {values['source']} | {values['closure']} | {values['full']} | {values['n']} |")

    lines.extend(["", "## By Closure Handling", "", "| Gold handling | Truth | Source | Closure | Full | n |", "|---|---:|---:|---:|---:|---:|"])
    for name, values in report["by_closure_handling"].items():
        lines.append(f"| {name} | {values['truth']} | {values['source']} | {values['closure']} | {values['full']} | {values['n']} |")

    lines.extend(["", "## Error Types", "", "| Error type | Count |", "|---|---:|"])
    for name, count in sorted(report["error_types"].items()):
        lines.append(f"| {name} | {count} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    report = summarize(read_jsonl(args.dataset), read_jsonl(args.scored))
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.report_md, report)
    print(json.dumps(report["full_accuracy"], indent=2))
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
