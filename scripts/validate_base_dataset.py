#!/usr/bin/env python3
"""Validate the Closure-Contract dataset invariants."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "closurebench_validation.md"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    errors: list[str] = []
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        fail(errors, "Duplicate item ids found.")

    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)
        if row["gold_answer"] not in {"true", "false", "unknown"}:
            fail(errors, f"{row['id']} has invalid gold answer {row['gold_answer']!r}.")
        if row["semantics"] == "owa" and row["closed_predicates"]:
            fail(errors, f"{row['id']} has closed predicates under OWA.")
        if row["semantics"] == "cwa" and set(row["closed_predicates"]) != set(row["vocabulary_predicates"]):
            fail(errors, f"{row['id']} does not close every vocabulary predicate under CWA.")
        if row["semantics"] == "lcwa" and row["closed_predicates"] == row["vocabulary_predicates"]:
            fail(errors, f"{row['id']} LCWA should not be identical to global CWA.")
        if "Important closure rule:" not in row["prompt"]:
            fail(errors, f"{row['id']} prompt is missing explicit closure instruction.")

    switch_patterns = Counter()
    for base_id, group in by_base.items():
        if len(group) != 3:
            fail(errors, f"{base_id} does not have exactly three semantic variants.")
            continue
        sems = {row["semantics"] for row in group}
        if sems != {"owa", "cwa", "lcwa"}:
            fail(errors, f"{base_id} has semantic variants {sorted(sems)}.")
        facts = {tuple(row["symbolic"]["positive_atoms"]) for row in group}
        negatives = {tuple(row["symbolic"]["negative_atoms"]) for row in group}
        rules = {json.dumps(row["symbolic"]["rules"], sort_keys=True) for row in group}
        targets = {row["query_atom"] for row in group}
        if len(facts) != 1 or len(negatives) != 1 or len(rules) != 1 or len(targets) != 1:
            fail(errors, f"{base_id} changes facts, rules, negatives, or target across semantic variants.")
        answers = {row["semantics"]: row["gold_answer"] for row in group}
        switch_patterns[(answers["owa"], answers["cwa"], answers["lcwa"])] += 1

    lines = [
        "# Closure-Contract Validation",
        "",
        f"Dataset: `{args.dataset}`",
        f"Items: {len(rows)}",
        f"Base scenarios: {len(by_base)}",
        f"Validation status: {'PASS' if not errors else 'FAIL'}",
        "",
        "## Distributions",
        "",
        f"- Semantics: {dict(Counter(row['semantics'] for row in rows))}",
        f"- Gold answers: {dict(Counter(row['gold_answer'] for row in rows))}",
        f"- Subsets: {dict(Counter(row['subset'] for row in rows))}",
        f"- Splits: {dict(Counter(row['split'] for row in rows))}",
        f"- Gold reason types: {dict(Counter(row['gold_reason_type'] for row in rows))}",
        "",
        "## Semantic Switch Patterns",
        "",
    ]
    for pattern, count in sorted(switch_patterns.items()):
        lines.append(f"- {pattern}: {count}")

    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors[:100])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Validation status: {'PASS' if not errors else 'FAIL'}")
    print(f"Wrote {args.report}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

