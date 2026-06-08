#!/usr/bin/env python3
"""Compare closure-contract runs with switch-first metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_REPORT_JSON = ROOT / "reports" / "model_comparison.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "model_comparison.md"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return round(100 * num / den, 2)


def parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Runs must be provided as label=/path/to/scored.jsonl")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty")
    return label, Path(path)


def is_lcwa_closed_scope(row: dict) -> bool:
    if row["semantics"] != "lcwa":
        return False
    if row.get("gold_reason_type") == "local_closed_world_underivable":
        return True
    return row.get("expected_pattern") == "owa_unknown_cwa_false_lcwa_false"


def is_lcwa_open_scope(row: dict) -> bool:
    if row["semantics"] != "lcwa":
        return False
    if row.get("gold_reason_type") == "local_open_world_underivable":
        return True
    return row.get("expected_pattern") == "owa_unknown_cwa_false_lcwa_unknown"


def summarize(dataset_rows: list[dict], scored_path: Path) -> dict:
    dataset_by_id = {row["id"]: row for row in dataset_rows}
    scored_rows = read_jsonl(scored_path)
    rows = []
    for row in scored_rows:
        if row["id"] not in dataset_by_id:
            continue
        merged = dict(dataset_by_id[row["id"]])
        merged.update(row)
        rows.append(merged)

    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)
    complete_groups = [items for items in by_base.values() if len(items) == 3]
    switch_correct = sum(1 for items in complete_groups if all(item["correct"] for item in items))
    core_groups = [
        items
        for items in complete_groups
        if items and items[0].get("subset") == "core_contrastive"
    ]
    core_switch_correct = sum(1 for items in core_groups if all(item["correct"] for item in items))

    lcwa_closed = [row for row in rows if is_lcwa_closed_scope(row)]
    lcwa_open = [row for row in rows if is_lcwa_open_scope(row)]
    cwa = [row for row in rows if row["semantics"] == "cwa"]
    owa = [row for row in rows if row["semantics"] == "owa"]
    false_rows = [row for row in rows if row["gold_answer"] == "false"]
    unknown_rows = [row for row in rows if row["gold_answer"] == "unknown"]

    def accuracy(items: list[dict]) -> dict:
        correct = sum(1 for item in items if item["correct"])
        return {"correct": correct, "n": len(items), "accuracy": pct(correct, len(items))}

    return {
        "n_scored": len(rows),
        "n_dataset": len(dataset_rows),
        "coverage": pct(len(rows), len(dataset_rows)),
        "models": sorted({row["model"] for row in rows}),
        "api_errors": sum(1 for row in rows if row.get("error")),
        "parse_failures": sum(1 for row in rows if row.get("model_answer") is None),
        "semantic_switch_accuracy": {
            "correct": switch_correct,
            "n": len(complete_groups),
            "accuracy": pct(switch_correct, len(complete_groups)),
        },
        "core_contrastive_switch_accuracy": {
            "correct": core_switch_correct,
            "n": len(core_groups),
            "accuracy": pct(core_switch_correct, len(core_groups)),
        },
        "lcwa_closed_scope_accuracy": accuracy(lcwa_closed),
        "lcwa_open_scope_accuracy": accuracy(lcwa_open),
        "cwa_accuracy": accuracy(cwa),
        "owa_accuracy": accuracy(owa),
        "false_accuracy": accuracy(false_rows),
        "unknown_accuracy": accuracy(unknown_rows),
        "overall_accuracy": accuracy(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", action="append", required=True, type=parse_run_arg)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    comparison = {
        "dataset": str(args.dataset),
        "n_dataset": len(dataset_rows),
        "runs": {
            label: summarize(dataset_rows, path)
            for label, path in args.run
        },
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)

    lines = [
        "# Closure-Contract Model Comparison",
        "",
        f"Dataset: `{args.dataset}` ({len(dataset_rows)} items).",
        "",
        "Primary metrics are semantic switch accuracy and LCWA closed-scope accuracy. Overall accuracy is reported only as a secondary diagnostic.",
        "",
        "| Run | Model | Coverage | Semantic Switch | Core Switch | LCWA Closed | LCWA Open | CWA | OWA | Overall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in comparison["runs"].items():
        model = ", ".join(stats["models"])
        lines.append(
            f"| {label} | {model} | {stats['coverage']}% | "
            f"{stats['semantic_switch_accuracy']['accuracy']}% "
            f"({stats['semantic_switch_accuracy']['correct']}/{stats['semantic_switch_accuracy']['n']}) | "
            f"{stats['core_contrastive_switch_accuracy']['accuracy']}% "
            f"({stats['core_contrastive_switch_accuracy']['correct']}/{stats['core_contrastive_switch_accuracy']['n']}) | "
            f"{stats['lcwa_closed_scope_accuracy']['accuracy']}% "
            f"({stats['lcwa_closed_scope_accuracy']['correct']}/{stats['lcwa_closed_scope_accuracy']['n']}) | "
            f"{stats['lcwa_open_scope_accuracy']['accuracy']}% "
            f"({stats['lcwa_open_scope_accuracy']['correct']}/{stats['lcwa_open_scope_accuracy']['n']}) | "
            f"{stats['cwa_accuracy']['accuracy']}% | "
            f"{stats['owa_accuracy']['accuracy']}% | "
            f"{stats['overall_accuracy']['accuracy']}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A model passes a base scenario only if it answers all three semantic variants correctly after the facts and rules stay fixed and only the closure contract changes.",
        ]
    )
    with args.report_md.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
