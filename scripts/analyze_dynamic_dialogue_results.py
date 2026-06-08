#!/usr/bin/env python3
"""Analyze ClosureBench dynamic dialogue results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_dynamic_dialogue.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_dynamic_dialogue_report.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_dynamic_dialogue_report.md"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def flatten_dataset(dialogues: list[dict]) -> dict[str, dict]:
    rows = {}
    for dialogue in dialogues:
        for turn in dialogue["turns"]:
            row_id = f"{dialogue['id']}__turn_{turn['turn_index']}"
            rows[row_id] = {
                "id": row_id,
                "dialogue_id": dialogue["id"],
                "base_id": dialogue["base_id"],
                "domain": dialogue["domain"],
                "family": dialogue["family"],
                "dialogue_type": dialogue["dialogue_type"],
                "gold_pattern": dialogue["gold_pattern"],
                "turn_index": turn["turn_index"],
                "update_operation": turn["update_operation"],
                "complete_predicates_after_update": turn["complete_predicates_after_update"],
                "gold_answer": turn["gold_answer"],
                "expected_changed_from_previous": turn["expected_changed_from_previous"],
                "expected_applied_update": turn.get("expected_applied_update"),
            }
    return rows


def accuracy(rows: list[dict], key: str = "answer_correct") -> dict:
    correct = sum(1 for row in rows if row.get(key))
    return {"correct": correct, "n": len(rows), "accuracy": pct(correct, len(rows))}


def available_accuracy(rows: list[dict], key: str) -> dict:
    available = [row for row in rows if key in row and row.get(key) is not None]
    return accuracy(available, key)


def grouped_accuracy(rows: list[dict], group_key: str, metric_key: str = "answer_correct") -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    return {name: accuracy(items, metric_key) for name, items in sorted(groups.items())}


def enrich_transition_rows(rows: list[dict]) -> list[dict]:
    by_dialogue: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dialogue[row["dialogue_id"]].append(row)

    enriched = []
    for items in by_dialogue.values():
        previous = None
        for row in sorted(items, key=lambda item: item["turn_index"]):
            row = dict(row)
            if previous is None:
                row["model_answer_changed"] = None
                row["anchoring_error"] = False
            else:
                if row.get("model_answer") is None or previous.get("model_answer") is None:
                    row["model_answer_changed"] = None
                else:
                    row["model_answer_changed"] = row["model_answer"] != previous["model_answer"]
                row["anchoring_error"] = bool(
                    row.get("expected_changed_from_previous") is True
                    and (
                        row.get("model_answer") == previous.get("model_answer")
                        or row.get("changed_from_previous") is False
                    )
                )
            enriched.append(row)
            previous = row
    return sorted(enriched, key=lambda row: (row["dialogue_id"], row["turn_index"]))


def classify_error(row: dict) -> str:
    if row.get("error"):
        return "api_error"
    if row.get("model_answer") is None:
        return "parse_failure"
    if row.get("answer_correct") and row.get("changed_correct"):
        return "correct"
    if row.get("answer_correct") and not row.get("changed_correct"):
        return "transition_flag_error"
    turn = row["turn_index"]
    gold = row["gold_answer"]
    pred = row.get("model_answer")
    if turn == 2 and gold == "false" and pred == "unknown":
        return "missed_completeness_addition"
    if turn == 3 and gold == "unknown" and pred == "false":
        return "failed_reopening_overclosed"
    if turn == 3 and gold == "false" and pred == "unknown":
        return "lost_relevant_local_closure"
    if row.get("anchoring_error"):
        return "anchoring_error"
    return "other_answer_error"


def summarize(dialogues: list[dict], scored_rows: list[dict]) -> dict:
    dataset_by_id = flatten_dataset(dialogues)
    rows = []
    for scored in scored_rows:
        if scored["id"] not in dataset_by_id:
            continue
        merged = dict(dataset_by_id[scored["id"]])
        merged.update(scored)
        rows.append(merged)
    rows = enrich_transition_rows(rows)
    for row in rows:
        row["error_type"] = classify_error(row)

    by_dialogue: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dialogue[row["dialogue_id"]].append(row)
    complete_dialogues = [items for items in by_dialogue.values() if len(items) == 3]
    all_answer = sum(1 for items in complete_dialogues if all(row["answer_correct"] for row in items))
    all_full = sum(1 for items in complete_dialogues if all(row["correct"] for row in items))

    turn2 = [row for row in rows if row["turn_index"] == 2]
    turn3 = [row for row in rows if row["turn_index"] == 3]
    reopen = [row for row in turn3 if row["dialogue_type"] == "reopen"]
    persist = [row for row in turn3 if row["dialogue_type"] == "persist_after_narrowing"]
    transitions = [row for row in rows if row["turn_index"] > 1]
    revisions = [row for row in transitions if row["expected_changed_from_previous"] is True]
    persistence = [row for row in transitions if row["expected_changed_from_previous"] is False]
    anchoring_errors = sum(1 for row in revisions if row["anchoring_error"])
    closure_inertia_errors = sum(1 for row in reopen if row.get("model_answer") == "false")
    closure_retention_errors = sum(1 for row in persist if row.get("model_answer") == "unknown")

    return {
        "n_dialogues": len(dialogues),
        "n_turns": len(dataset_by_id),
        "n_scored": len(rows),
        "coverage": pct(len(rows), len(dataset_by_id)),
        "models": sorted({row.get("model") for row in rows if row.get("model")}),
        "api_errors": sum(1 for row in rows if row.get("error")),
        "parse_failures": sum(1 for row in rows if row.get("model_answer") is None and not row.get("error")),
        "json_valid": {
            "n": sum(1 for row in rows if row.get("json_valid")),
            "rate": pct(sum(1 for row in rows if row.get("json_valid")), len(rows)),
        },
        "answer_accuracy": accuracy(rows, "answer_correct"),
        "full_turn_accuracy": accuracy(rows, "correct"),
        "dialogue_all_answer_accuracy": {"correct": all_answer, "n": len(complete_dialogues), "accuracy": pct(all_answer, len(complete_dialogues))},
        "dialogue_all_turn_accuracy": {"correct": all_full, "n": len(complete_dialogues), "accuracy": pct(all_full, len(complete_dialogues))},
        "revision_accuracy": accuracy(revisions, "answer_correct"),
        "persistence_accuracy": accuracy(persistence, "answer_correct"),
        "completeness_addition_accuracy": accuracy(turn2, "answer_correct"),
        "reopening_accuracy": accuracy(reopen, "answer_correct"),
        "narrowing_persistence_accuracy": accuracy(persist, "answer_correct"),
        "changed_flag_accuracy": accuracy(transitions, "changed_correct"),
        "applied_update_accuracy": available_accuracy(rows, "applied_update_correct"),
        "anchoring_error_rate": {"errors": anchoring_errors, "n": len(revisions), "rate": pct(anchoring_errors, len(revisions))},
        "closure_inertia_rate": {"errors": closure_inertia_errors, "n": len(reopen), "rate": pct(closure_inertia_errors, len(reopen))},
        "closure_retention_failure_rate": {
            "errors": closure_retention_errors,
            "n": len(persist),
            "rate": pct(closure_retention_errors, len(persist)),
        },
        "by_turn": grouped_accuracy(rows, "turn_index", "answer_correct"),
        "by_dialogue_type": grouped_accuracy(rows, "dialogue_type", "answer_correct"),
        "by_update_operation": grouped_accuracy(rows, "update_operation", "answer_correct"),
        "error_types": dict(Counter(row["error_type"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scored", action="append", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    dialogues = read_jsonl(args.dataset)
    scored_rows = []
    for path in args.scored:
        scored_rows.extend(read_jsonl(path))
    report = summarize(dialogues, scored_rows)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    model = ", ".join(report["models"]) if report["models"] else "none"
    lines = [
        "# ClosureBench Dynamic Dialogue Report",
        "",
        f"Model: {model}.",
        f"Scored turns: {report['n_scored']}/{report['n_turns']} ({report['coverage']}%).",
        f"Dialogue all-turn accuracy: {report['dialogue_all_turn_accuracy']['accuracy']}% "
        f"({report['dialogue_all_turn_accuracy']['correct']}/{report['dialogue_all_turn_accuracy']['n']}).",
        f"Dialogue all-answer accuracy: {report['dialogue_all_answer_accuracy']['accuracy']}% "
        f"({report['dialogue_all_answer_accuracy']['correct']}/{report['dialogue_all_answer_accuracy']['n']}).",
        f"Answer accuracy: {report['answer_accuracy']['accuracy']}% "
        f"({report['answer_accuracy']['correct']}/{report['answer_accuracy']['n']}).",
        f"Revision accuracy: {report['revision_accuracy']['accuracy']}% "
        f"({report['revision_accuracy']['correct']}/{report['revision_accuracy']['n']}).",
        f"Persistence accuracy: {report['persistence_accuracy']['accuracy']}% "
        f"({report['persistence_accuracy']['correct']}/{report['persistence_accuracy']['n']}).",
        f"Completeness addition accuracy: {report['completeness_addition_accuracy']['accuracy']}% "
        f"({report['completeness_addition_accuracy']['correct']}/{report['completeness_addition_accuracy']['n']}).",
        f"Reopening accuracy: {report['reopening_accuracy']['accuracy']}% "
        f"({report['reopening_accuracy']['correct']}/{report['reopening_accuracy']['n']}).",
        f"Narrowing persistence accuracy: {report['narrowing_persistence_accuracy']['accuracy']}% "
        f"({report['narrowing_persistence_accuracy']['correct']}/{report['narrowing_persistence_accuracy']['n']}).",
        f"Closure inertia rate: {report['closure_inertia_rate']['rate']}% "
        f"({report['closure_inertia_rate']['errors']}/{report['closure_inertia_rate']['n']}).",
        f"Closure retention failure rate: {report['closure_retention_failure_rate']['rate']}% "
        f"({report['closure_retention_failure_rate']['errors']}/{report['closure_retention_failure_rate']['n']}).",
        f"Anchoring error rate: {report['anchoring_error_rate']['rate']}% "
        f"({report['anchoring_error_rate']['errors']}/{report['anchoring_error_rate']['n']}).",
        f"Changed-flag accuracy: {report['changed_flag_accuracy']['accuracy']}% "
        f"({report['changed_flag_accuracy']['correct']}/{report['changed_flag_accuracy']['n']}).",
        f"Applied-update accuracy: {report['applied_update_accuracy']['accuracy']}% "
        f"({report['applied_update_accuracy']['correct']}/{report['applied_update_accuracy']['n']}).",
        "",
        "## By Turn",
        "",
        "| Turn | Accuracy |",
        "|---:|---:|",
    ]
    for turn, stats in report["by_turn"].items():
        lines.append(f"| {turn} | {stats['accuracy']}% ({stats['correct']}/{stats['n']}) |")
    lines.extend(["", "## Error Types", "", "| Error type | Count |", "|---|---:|"])
    for error_type, count in sorted(report["error_types"].items()):
        lines.append(f"| {error_type} | {count} |")

    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
