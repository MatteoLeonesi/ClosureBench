#!/usr/bin/env python3
"""Analyze ClosureBench-Ask/Act results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_ask_act.jsonl"
DEFAULT_SCORED = ROOT / "results" / "closurebench_ask_act_scored.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_ask_act_report.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_ask_act_report.md"

REQUEST = "request_information"
CLOSURE_FALSE_REASONS = {"global_closed_world_underivable", "local_closed_world_underivable"}
UNRESOLVED_REASONS = {"open_world_underivable", "local_open_world_underivable"}
FALSE_ACTION_MARKERS = (
    "should be false",
    "truth_value should be false",
    "truth value should be false",
    "should deny",
    "action deny",
    "action should be deny",
)
UNKNOWN_ACTION_MARKERS = (
    "should be unknown",
    "truth_value is unknown",
    "truth value is unknown",
    "should be request_information",
    "action should be request_information",
    "action request_information",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 2)


def metric(rows: list[dict], key: str) -> dict:
    correct = sum(1 for row in rows if row.get(key))
    return {"correct": correct, "n": len(rows), "accuracy": pct(correct, len(rows))}


def rate(rows: list[dict], predicate) -> dict:
    count = sum(1 for row in rows if predicate(row))
    return {"count": count, "n": len(rows), "rate": pct(count, len(rows))}


def grouped_metric(rows: list[dict], group_key: str, metric_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {name: metric(items, metric_key) for name, items in sorted(groups.items())}


def grouped_dual_metric(rows: list[dict], group_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {
        name: {
            "n": len(items),
            "truth_accuracy": metric(items, "truth_correct")["accuracy"],
            "action_accuracy": metric(items, "action_correct")["accuracy"],
            "full_accuracy": metric(items, "correct")["accuracy"],
        }
        for name, items in sorted(groups.items())
    }


def classify_error(row: dict) -> str:
    if row.get("error"):
        return "api_error"
    if row.get("model_truth_value") is None and row.get("model_action") is None:
        return "parse_failure"
    if row.get("correct"):
        return "correct"
    if row.get("action_correct") and not row.get("truth_correct"):
        return "corrupt_action_success_truth_wrong"
    if row.get("truth_correct") and not row.get("action_correct"):
        return "truth_correct_action_wrong"
    if row["gold_action"] == REQUEST and row.get("model_action") == "deny":
        return "premature_denial_from_unresolved_absence"
    if row["gold_action"] == REQUEST and row.get("model_action") == "approve":
        return "premature_approval_from_unresolved_absence"
    if row["gold_action"] != REQUEST and row.get("model_action") == REQUEST:
        if row.get("gold_reason_type") in CLOSURE_FALSE_REASONS:
            return "excessive_ask_under_closure_false"
        return "excessive_ask_on_determined_case"
    if row["gold_action"] == REQUEST and row.get("model_action") == REQUEST and not row.get("targeted_info_correct"):
        return "untargeted_or_wrong_information_request"
    if row.get("gold_reason_type") in CLOSURE_FALSE_REASONS and row.get("model_action") != "deny":
        return "missed_closure_denial"
    return "other_decision_error"


def contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def semantic_switch_accuracy(rows: list[dict], metric_key: str, subset: str | None = None) -> dict:
    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if subset is not None and row.get("subset") != subset:
            continue
        by_base[row["base_id"]].append(row)
    complete = [items for items in by_base.values() if len({item["semantics"] for item in items}) == 3]
    correct = sum(1 for items in complete if all(item.get(metric_key) for item in items))
    return {"correct": correct, "n": len(complete), "accuracy": pct(correct, len(complete))}


def summarize(dataset_rows: list[dict], scored_rows: list[dict]) -> dict:
    dataset_by_id = {row["id"]: row for row in dataset_rows}
    rows = []
    for scored in scored_rows:
        if scored["id"] not in dataset_by_id:
            continue
        merged = dict(dataset_by_id[scored["id"]])
        merged.update(scored)
        rows.append(merged)

    for row in rows:
        row["error_type"] = classify_error(row)

    gold_ask = [row for row in rows if row["gold_action"] == REQUEST]
    model_ask = [row for row in rows if row.get("model_action") == REQUEST]
    true_positive_ask = sum(1 for row in rows if row["gold_action"] == REQUEST and row.get("model_action") == REQUEST)
    ask_precision_value = pct(true_positive_ask, len(model_ask))
    ask_recall_value = pct(true_positive_ask, len(gold_ask))

    determined = [row for row in rows if row["gold_action"] != REQUEST]
    closure_false = [row for row in rows if row.get("gold_reason_type") in CLOSURE_FALSE_REASONS]
    unresolved = [row for row in rows if row.get("gold_reason_type") in UNRESOLVED_REASONS]
    action_correct_truth_wrong = [row for row in rows if row.get("action_correct") and not row.get("truth_correct")]
    closure_false_errors = [row for row in closure_false if not row.get("correct")]
    unresolved_errors = [row for row in unresolved if not row.get("correct")]

    return {
        "n_dataset": len(dataset_rows),
        "n_scored": len(rows),
        "coverage": pct(len(rows), len(dataset_rows)),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "api_errors": sum(1 for row in rows if row.get("error")),
        "parse_failures": sum(1 for row in rows if row.get("model_truth_value") is None or row.get("model_action") is None),
        "json_valid": {
            "n": sum(1 for row in rows if row.get("json_valid")),
            "rate": pct(sum(1 for row in rows if row.get("json_valid")), len(rows)),
        },
        "truth_accuracy": metric(rows, "truth_correct"),
        "action_accuracy": metric(rows, "action_correct"),
        "full_decision_accuracy": metric(rows, "correct"),
        "semantic_switch_action_accuracy": semantic_switch_accuracy(rows, "action_correct"),
        "semantic_switch_full_accuracy": semantic_switch_accuracy(rows, "correct"),
        "core_semantic_switch_action_accuracy": semantic_switch_accuracy(rows, "action_correct", "core_contrastive"),
        "core_semantic_switch_full_accuracy": semantic_switch_accuracy(rows, "correct", "core_contrastive"),
        "ask_metrics": {
            "true_positive": true_positive_ask,
            "model_asks": len(model_ask),
            "gold_asks": len(gold_ask),
            "precision": ask_precision_value,
            "recall": ask_recall_value,
            "f1": f1(ask_precision_value, ask_recall_value),
        },
        "targeted_information_accuracy": metric(gold_ask, "targeted_info_correct"),
        "closure_denial_accuracy": metric(closure_false, "action_correct"),
        "unresolved_ask_accuracy": metric(unresolved, "action_correct"),
        "internal_action_consistency": metric(rows, "internal_action_consistent"),
        "premature_denial_rate": rate(gold_ask, lambda row: row.get("model_action") == "deny"),
        "premature_approval_rate": rate(
            [row for row in rows if row["gold_action"] != "approve"],
            lambda row: row.get("model_action") == "approve",
        ),
        "excessive_ask_rate": rate(determined, lambda row: row.get("model_action") == REQUEST),
        "excessive_ask_on_closure_false_rate": rate(closure_false, lambda row: row.get("model_action") == REQUEST),
        "corrupt_action_success_rate": {
            "count": len(action_correct_truth_wrong),
            "n": len(rows),
            "rate": pct(len(action_correct_truth_wrong), len(rows)),
        },
        "latent_self_correction_markers": {
            "closure_false_errors_with_false_or_deny_marker": {
                "count": sum(
                    1 for row in closure_false_errors if contains_any_marker(row.get("output_text", ""), FALSE_ACTION_MARKERS)
                ),
                "n": len(closure_false_errors),
                "rate": pct(
                    sum(
                        1
                        for row in closure_false_errors
                        if contains_any_marker(row.get("output_text", ""), FALSE_ACTION_MARKERS)
                    ),
                    len(closure_false_errors),
                ),
            },
            "unresolved_errors_with_unknown_or_request_marker": {
                "count": sum(
                    1 for row in unresolved_errors if contains_any_marker(row.get("output_text", ""), UNKNOWN_ACTION_MARKERS)
                ),
                "n": len(unresolved_errors),
                "rate": pct(
                    sum(
                        1
                        for row in unresolved_errors
                        if contains_any_marker(row.get("output_text", ""), UNKNOWN_ACTION_MARKERS)
                    ),
                    len(unresolved_errors),
                ),
            },
        },
        "by_semantics": grouped_dual_metric(rows, "semantics"),
        "by_gold_action": grouped_dual_metric(rows, "gold_action"),
        "by_gold_reason_type": grouped_dual_metric(rows, "gold_reason_type"),
        "by_family": grouped_metric(rows, "family", "action_correct"),
        "error_types": dict(Counter(row["error_type"] for row in rows)),
    }


def write_markdown(path: Path, report: dict) -> None:
    model = ", ".join(report["models"]) if report["models"] else "none"
    lines = [
        "# ClosureBench-Ask/Act Report",
        "",
        f"Model: {model}.",
        f"Scored items: {report['n_scored']}/{report['n_dataset']} ({report['coverage']}%).",
        f"Action accuracy: {report['action_accuracy']['accuracy']}% "
        f"({report['action_accuracy']['correct']}/{report['action_accuracy']['n']}).",
        f"Truth accuracy: {report['truth_accuracy']['accuracy']}% "
        f"({report['truth_accuracy']['correct']}/{report['truth_accuracy']['n']}).",
        f"Full decision accuracy: {report['full_decision_accuracy']['accuracy']}% "
        f"({report['full_decision_accuracy']['correct']}/{report['full_decision_accuracy']['n']}).",
        f"Semantic-switch action accuracy: {report['semantic_switch_action_accuracy']['accuracy']}% "
        f"({report['semantic_switch_action_accuracy']['correct']}/{report['semantic_switch_action_accuracy']['n']}).",
        f"Core semantic-switch action accuracy: {report['core_semantic_switch_action_accuracy']['accuracy']}% "
        f"({report['core_semantic_switch_action_accuracy']['correct']}/{report['core_semantic_switch_action_accuracy']['n']}).",
        f"Ask precision/recall/F1: {report['ask_metrics']['precision']} / "
        f"{report['ask_metrics']['recall']} / {report['ask_metrics']['f1']}.",
        f"Targeted information accuracy: {report['targeted_information_accuracy']['accuracy']}% "
        f"({report['targeted_information_accuracy']['correct']}/{report['targeted_information_accuracy']['n']}).",
        f"Closure-denial accuracy: {report['closure_denial_accuracy']['accuracy']}% "
        f"({report['closure_denial_accuracy']['correct']}/{report['closure_denial_accuracy']['n']}).",
        f"Unresolved-ask accuracy: {report['unresolved_ask_accuracy']['accuracy']}% "
        f"({report['unresolved_ask_accuracy']['correct']}/{report['unresolved_ask_accuracy']['n']}).",
        f"Premature denial rate: {report['premature_denial_rate']['rate']}% "
        f"({report['premature_denial_rate']['count']}/{report['premature_denial_rate']['n']}).",
        f"Excessive ask rate: {report['excessive_ask_rate']['rate']}% "
        f"({report['excessive_ask_rate']['count']}/{report['excessive_ask_rate']['n']}).",
        f"Excessive ask on closure-false cases: {report['excessive_ask_on_closure_false_rate']['rate']}% "
        f"({report['excessive_ask_on_closure_false_rate']['count']}/{report['excessive_ask_on_closure_false_rate']['n']}).",
        f"Strict JSON-valid response rate: {report['json_valid']['rate']}%.",
        f"Closure-false errors with false/deny self-correction markers: "
        f"{report['latent_self_correction_markers']['closure_false_errors_with_false_or_deny_marker']['rate']}% "
        f"({report['latent_self_correction_markers']['closure_false_errors_with_false_or_deny_marker']['count']}/"
        f"{report['latent_self_correction_markers']['closure_false_errors_with_false_or_deny_marker']['n']}).",
        f"Unresolved errors with unknown/request self-correction markers: "
        f"{report['latent_self_correction_markers']['unresolved_errors_with_unknown_or_request_marker']['rate']}% "
        f"({report['latent_self_correction_markers']['unresolved_errors_with_unknown_or_request_marker']['count']}/"
        f"{report['latent_self_correction_markers']['unresolved_errors_with_unknown_or_request_marker']['n']}).",
        "",
        "## By Semantic Contract",
        "",
        "| Contract | Truth | Action | Full | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in report["by_semantics"].items():
        lines.append(
            f"| {name} | {stats['truth_accuracy']}% | {stats['action_accuracy']}% | "
            f"{stats['full_accuracy']}% | {stats['n']} |"
        )

    lines.extend(["", "## By Gold Action", "", "| Gold action | Truth | Action | Full | n |", "|---|---:|---:|---:|---:|"])
    for name, stats in report["by_gold_action"].items():
        lines.append(
            f"| {name} | {stats['truth_accuracy']}% | {stats['action_accuracy']}% | "
            f"{stats['full_accuracy']}% | {stats['n']} |"
        )

    lines.extend(["", "## By Gold Reason Type", "", "| Reason type | Truth | Action | Full | n |", "|---|---:|---:|---:|---:|"])
    for name, stats in report["by_gold_reason_type"].items():
        lines.append(
            f"| {name} | {stats['truth_accuracy']}% | {stats['action_accuracy']}% | "
            f"{stats['full_accuracy']}% | {stats['n']} |"
        )

    lines.extend(["", "## Error Types", "", "| Error type | Count |", "|---|---:|"])
    for error_type, count in sorted(report["error_types"].items()):
        lines.append(f"| {error_type} | {count} |")

    lines.extend(["", "## Interpretation", ""])
    if report["excessive_ask_on_closure_false_rate"]["count"]:
        lines.append(
            "The model often requests more information even when CWA/LCWA closure already makes the target false. "
            "This is an action-level version of the LCWA closed-scope failure: the model treats a determined denial as unresolved."
        )
    elif report["premature_denial_rate"]["count"]:
        lines.append(
            "The model sometimes denies under unresolved open-world uncertainty. This is the unsafe opposite failure: "
            "it acts as if absence were false when the contract leaves it unknown."
        )
    else:
        lines.append("No dominant ask/act failure mode was observed in the scored run.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scored", action="append", type=Path, default=None)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    scored_paths = args.scored or [DEFAULT_SCORED]
    scored_rows = []
    for path in scored_paths:
        scored_rows.extend(read_jsonl(path))

    report = summarize(dataset_rows, scored_rows)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    write_markdown(args.report_md, report)

    print(json.dumps(report["action_accuracy"], indent=2))
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
