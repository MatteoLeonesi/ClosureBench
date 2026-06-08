#!/usr/bin/env python3
"""Analyze closure-contract benchmark results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_SCORED = ROOT / "results" / "closurebench_deepseek_v4_flash_scored.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_deepseek_v4_flash_report.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_deepseek_v4_flash_report.md"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def grouped_accuracy(rows: list[dict], key: str) -> dict:
    out = {}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    for name, items in sorted(groups.items()):
        out[name] = {
            "n": len(items),
            "correct": sum(1 for item in items if item["correct"]),
            "accuracy": pct(sum(1 for item in items if item["correct"]), len(items)),
        }
    return out


def grouped_accuracy_two_keys(rows: list[dict], key_a: str, key_b: str) -> dict:
    out = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row[key_a]), str(row[key_b]))].append(row)
    for (name_a, name_b), items in sorted(groups.items()):
        out.setdefault(name_a, {})[name_b] = {
            "n": len(items),
            "correct": sum(1 for item in items if item["correct"]),
            "accuracy": pct(sum(1 for item in items if item["correct"]), len(items)),
        }
    return out


def is_lcwa_open_scope(row: dict) -> bool:
    if row["semantics"] != "lcwa":
        return False
    if row.get("gold_reason_type") == "local_open_world_underivable":
        return True
    return row.get("expected_pattern") == "owa_unknown_cwa_false_lcwa_unknown"


def is_lcwa_closed_scope(row: dict) -> bool:
    if row["semantics"] != "lcwa":
        return False
    if row.get("gold_reason_type") == "local_closed_world_underivable":
        return True
    return row.get("expected_pattern") == "owa_unknown_cwa_false_lcwa_false"


def classify_error(row: dict) -> str:
    gold = row["gold_answer"]
    pred = row["model_answer"]
    sem = row["semantics"]
    family = row["family"]
    if pred is None:
        return "unparseable_or_api_error"
    if pred == gold:
        return "correct"
    if gold == "unknown" and pred == "false":
        if sem == "lcwa":
            return "closure_scope_error_false_for_open_predicate"
        return "closed_world_bias"
    if gold == "false" and pred == "unknown":
        if sem == "lcwa":
            return "missed_local_closure"
        return "open_world_bias"
    if gold != "true" and pred == "true":
        return "hallucinated_true"
    if gold == "true" and pred != "true":
        return "missed_true_entailment"
    if family == "explicit_negative_fact":
        return "explicit_negation_error"
    return "other_error"


def baseline_predictions(row: dict) -> dict[str, str]:
    if row.get("gold_reason_type") == "entailed_by_fact_or_rule":
        return {"always_cwa": "true", "always_owa": "true", "ignore_lcwa": "true"}
    if row.get("gold_reason_type") == "explicit_negative_fact":
        return {"always_cwa": "false", "always_owa": "false", "ignore_lcwa": "false"}
    family = row["family"]
    if family in {"known_positive_fact", "derived_true_closed_conclusion"}:
        return {"always_cwa": "true", "always_owa": "true", "ignore_lcwa": "true"}
    if family == "explicit_negative_fact":
        return {"always_cwa": "false", "always_owa": "false", "ignore_lcwa": "false"}
    always_cwa = "false"
    always_owa = "unknown"
    ignore_lcwa = "unknown" if row["semantics"] != "cwa" else "false"
    return {"always_cwa": always_cwa, "always_owa": always_owa, "ignore_lcwa": ignore_lcwa}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    scored_rows = read_jsonl(args.scored)
    dataset_by_id = {row["id"]: row for row in dataset_rows}

    rows = []
    for row in scored_rows:
        if row["id"] not in dataset_by_id:
            continue
        merged = dict(dataset_by_id[row["id"]])
        merged.update(row)
        merged["error_type"] = classify_error(merged)
        rows.append(merged)

    n = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    json_valid = sum(1 for row in rows if row["json_valid"])
    parsed_answers = sum(1 for row in rows if row["model_answer"] is not None)
    api_errors = sum(1 for row in rows if row["error"])

    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)
    complete_base_groups = [items for items in by_base.values() if len(items) == 3]
    switch_correct = sum(1 for items in complete_base_groups if all(item["correct"] for item in items))
    core_base_groups = [
        items
        for items in complete_base_groups
        if items and items[0].get("subset") == "core_contrastive"
    ]
    core_switch_correct = sum(1 for items in core_base_groups if all(item["correct"] for item in items))

    unknown_rows = [row for row in rows if row["gold_answer"] == "unknown"]
    false_rows = [row for row in rows if row["gold_answer"] == "false"]
    true_rows = [row for row in rows if row["gold_answer"] == "true"]
    lcwa_rows = [row for row in rows if row["semantics"] == "lcwa"]
    lcwa_open_rows = [row for row in lcwa_rows if is_lcwa_open_scope(row)]
    lcwa_closed_rows = [row for row in lcwa_rows if is_lcwa_closed_scope(row)]

    baselines = {}
    for baseline in ("always_cwa", "always_owa", "ignore_lcwa"):
        b_correct = 0
        for row in rows:
            pred = baseline_predictions(row)[baseline]
            b_correct += pred == row["gold_answer"]
        baselines[baseline] = {"correct": b_correct, "accuracy": pct(b_correct, n), "n": n}

    report = {
        "n_scored": n,
        "n_dataset": len(dataset_rows),
        "coverage": pct(n, len(dataset_rows)),
        "api_errors": api_errors,
        "json_valid": {"n": json_valid, "rate": pct(json_valid, n)},
        "answer_parse": {"n": parsed_answers, "rate": pct(parsed_answers, n)},
        "overall_accuracy": {"correct": correct, "n": n, "accuracy": pct(correct, n)},
        "by_semantics": grouped_accuracy(rows, "semantics"),
        "by_subset": grouped_accuracy(rows, "subset") if rows and "subset" in rows[0] else {},
        "by_family": grouped_accuracy(rows, "family"),
        "by_family_and_semantics": grouped_accuracy_two_keys(rows, "family", "semantics"),
        "by_gold_answer": grouped_accuracy(rows, "gold_answer"),
        "by_gold_reason_type": grouped_accuracy(rows, "gold_reason_type") if rows and "gold_reason_type" in rows[0] else {},
        "semantic_switch_accuracy": {
            "complete_base_groups": len(complete_base_groups),
            "correct_groups": switch_correct,
            "accuracy": pct(switch_correct, len(complete_base_groups)),
        },
        "core_contrastive_switch_accuracy": {
            "complete_base_groups": len(core_base_groups),
            "correct_groups": core_switch_correct,
            "accuracy": pct(core_switch_correct, len(core_base_groups)),
        },
        "unknown_accuracy": {
            "correct": sum(1 for row in unknown_rows if row["correct"]),
            "n": len(unknown_rows),
            "accuracy": pct(sum(1 for row in unknown_rows if row["correct"]), len(unknown_rows)),
        },
        "false_accuracy": {
            "correct": sum(1 for row in false_rows if row["correct"]),
            "n": len(false_rows),
            "accuracy": pct(sum(1 for row in false_rows if row["correct"]), len(false_rows)),
        },
        "true_accuracy": {
            "correct": sum(1 for row in true_rows if row["correct"]),
            "n": len(true_rows),
            "accuracy": pct(sum(1 for row in true_rows if row["correct"]), len(true_rows)),
        },
        "lcwa_open_scope_accuracy": {
            "correct": sum(1 for row in lcwa_open_rows if row["correct"]),
            "n": len(lcwa_open_rows),
            "accuracy": pct(sum(1 for row in lcwa_open_rows if row["correct"]), len(lcwa_open_rows)),
        },
        "lcwa_closed_scope_accuracy": {
            "correct": sum(1 for row in lcwa_closed_rows if row["correct"]),
            "n": len(lcwa_closed_rows),
            "accuracy": pct(sum(1 for row in lcwa_closed_rows if row["correct"]), len(lcwa_closed_rows)),
        },
        "semantic_bias_index": {
            "owa_unknown_predicted_false_rate": pct(
                sum(1 for row in rows if row["semantics"] == "owa" and row["gold_answer"] == "unknown" and row["model_answer"] == "false"),
                sum(1 for row in rows if row["semantics"] == "owa" and row["gold_answer"] == "unknown"),
            ),
            "lcwa_open_predicted_false_rate": pct(
                sum(1 for row in lcwa_open_rows if row["model_answer"] == "false"),
                len(lcwa_open_rows),
            ),
            "lcwa_closed_predicted_unknown_rate": pct(
                sum(1 for row in lcwa_closed_rows if row["model_answer"] == "unknown"),
                len(lcwa_closed_rows),
            ),
        },
        "error_types": Counter(row["error_type"] for row in rows),
        "baselines": baselines,
        "models": sorted({row["model"] for row in rows}),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=dict)

    lines = [
        "# Closure-Contract Report",
        "",
        f"Model: {', '.join(report['models'])}.",
        f"Scored items: {n}/{len(dataset_rows)} ({report['coverage']}%).",
        f"Semantic switch accuracy: {report['semantic_switch_accuracy']['accuracy']}% "
        f"({switch_correct}/{len(complete_base_groups)} base groups).",
        f"Core contrastive switch accuracy: {report['core_contrastive_switch_accuracy']['accuracy']}% "
        f"({core_switch_correct}/{len(core_base_groups)} base groups).",
        f"LCWA closed-scope accuracy: {report['lcwa_closed_scope_accuracy']['accuracy']}% "
        f"({report['lcwa_closed_scope_accuracy']['correct']}/{report['lcwa_closed_scope_accuracy']['n']}).",
        f"Overall accuracy: {report['overall_accuracy']['accuracy']}% ({correct}/{n}).",
        f"Answer parse rate: {report['answer_parse']['rate']}%.",
        f"Strict JSON-valid response rate: {report['json_valid']['rate']}%.",
        "",
        "## Accuracy by Semantic Contract",
        "",
    ]
    for name, stats in report["by_semantics"].items():
        lines.append(f"- {name}: {stats['accuracy']}% ({stats['correct']}/{stats['n']})")

    if report["by_subset"]:
        lines.extend(["", "## Accuracy by Subset", ""])
        for name, stats in report["by_subset"].items():
            lines.append(f"- {name}: {stats['accuracy']}% ({stats['correct']}/{stats['n']})")

    lines.extend(["", "## Accuracy by Family", ""])
    for name, stats in report["by_family"].items():
        lines.append(f"- {name}: {stats['accuracy']}% ({stats['correct']}/{stats['n']})")

    lines.extend(["", "## Accuracy by Family and Contract", ""])
    lines.append("| Family | OWA | CWA | LCWA |")
    lines.append("|---|---:|---:|---:|")
    for family, sem_stats in report["by_family_and_semantics"].items():
        cells = []
        for sem in ("owa", "cwa", "lcwa"):
            stats = sem_stats.get(sem, {"accuracy": 0.0, "correct": 0, "n": 0})
            cells.append(f"{stats['accuracy']}% ({stats['correct']}/{stats['n']})")
        lines.append(f"| {family} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines.extend(["", "## Key Diagnostics", ""])
    lines.append(
        f"- Unknown accuracy: {report['unknown_accuracy']['accuracy']}% "
        f"({report['unknown_accuracy']['correct']}/{report['unknown_accuracy']['n']})"
    )
    lines.append(
        f"- LCWA open-scope accuracy: {report['lcwa_open_scope_accuracy']['accuracy']}% "
        f"({report['lcwa_open_scope_accuracy']['correct']}/{report['lcwa_open_scope_accuracy']['n']})"
    )
    lines.append(
        f"- LCWA closed-scope accuracy: {report['lcwa_closed_scope_accuracy']['accuracy']}% "
        f"({report['lcwa_closed_scope_accuracy']['correct']}/{report['lcwa_closed_scope_accuracy']['n']})"
    )
    lines.append(
        f"- OWA unknown predicted false rate: "
        f"{report['semantic_bias_index']['owa_unknown_predicted_false_rate']}%"
    )
    lines.append(
        f"- LCWA open predicates predicted false rate: "
        f"{report['semantic_bias_index']['lcwa_open_predicted_false_rate']}%"
    )
    lines.append(
        f"- LCWA closed predicates predicted unknown rate: "
        f"{report['semantic_bias_index']['lcwa_closed_predicted_unknown_rate']}%"
    )

    lines.extend(["", "## Error Types", ""])
    for name, count in sorted(report["error_types"].items()):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Baselines", ""])
    for name, stats in report["baselines"].items():
        lines.append(f"- {name}: {stats['accuracy']}% ({stats['correct']}/{stats['n']})")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The core measure is not ordinary item accuracy. The stricter measure is semantic switch accuracy: "
            "all three variants of the same base knowledge scenario must be answered correctly after only the "
            "closure contract changes.",
        ]
    )
    if correct == n:
        lines.extend(
            [
                "",
                "No semantic errors were observed after successful answer parsing on this run.",
            ]
        )
    elif (
        report["error_types"].get("closure_scope_error_false_for_open_predicate", 0) > 0
        and report["error_types"].get("missed_local_closure", 0) > 0
    ):
        lines.extend(
            [
                "",
                "The main observed failure mode is unstable local-closure scope control. The model sometimes "
                "over-applies closure to LCWA-open predicates, predicting false where the contract requires "
                "unknown, and sometimes under-applies closure to LCWA-closed predicates, predicting unknown "
                "where the contract requires false.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The main observed failure mode is asymmetric. The model preserves unknowns under OWA and under "
                "LCWA-open predicates, but often predicts unknown when CWA or LCWA-closed predicates require false. "
                "This is a closure-contract compliance failure rather than a generic entailment failure.",
            ]
        )

    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(json.dumps(report["overall_accuracy"], indent=2))
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
