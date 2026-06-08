#!/usr/bin/env python3
"""Validate ClosureBench dynamic multi-turn dialogue structure."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_dynamic_dialogue.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "closurebench_dynamic_dialogue_validation.md"

EXPECTED_OPERATIONS = [
    "SET_COMPLETE_PREDICATES",
    "ADD_COMPLETE_PREDICATES",
    "REPLACE_COMPLETE_PREDICATES",
]
EXPECTED_UPDATES = ["open", "global_complete", "local_complete"]
EXPECTED_PATTERNS = {"unknown|false|false", "unknown|false|unknown"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_dialogue(dialogue: dict, errors: list[str]) -> None:
    dialogue_id = dialogue.get("id", "<missing-id>")
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        fail(errors, f"{dialogue_id}: expected exactly 3 turns")
        return

    gold_answers = []
    for index, turn in enumerate(turns, start=1):
        prefix = f"{dialogue_id} turn {index}"
        if turn.get("turn_index") != index:
            fail(errors, f"{prefix}: wrong turn_index {turn.get('turn_index')!r}")
        if turn.get("update_operation") != EXPECTED_OPERATIONS[index - 1]:
            fail(errors, f"{prefix}: wrong update_operation {turn.get('update_operation')!r}")
        if turn.get("expected_applied_update") != EXPECTED_UPDATES[index - 1]:
            fail(errors, f"{prefix}: wrong expected_applied_update {turn.get('expected_applied_update')!r}")
        if turn.get("gold_answer") not in {"true", "false", "unknown"}:
            fail(errors, f"{prefix}: invalid gold_answer {turn.get('gold_answer')!r}")
        gold_answers.append(turn.get("gold_answer"))

        prompt = turn.get("prompt", "")
        banned_labels = ["OWA", "CWA", "LCWA", "open-world", "closed-world", "local-closed-world"]
        for label in banned_labels:
            if label.lower() in prompt.lower():
                fail(errors, f"{prefix}: prompt contains explicit semantic label {label!r}")

    pattern = "|".join(str(answer) for answer in gold_answers)
    if dialogue.get("gold_pattern") != pattern:
        fail(errors, f"{dialogue_id}: gold_pattern {dialogue.get('gold_pattern')!r} != turn pattern {pattern!r}")
    if pattern not in EXPECTED_PATTERNS:
        fail(errors, f"{dialogue_id}: unsupported gold trajectory {pattern!r}")

    expected_changed = [None, gold_answers[1] != gold_answers[0], gold_answers[2] != gold_answers[1]]
    for index, turn in enumerate(turns):
        if turn.get("expected_changed_from_previous") != expected_changed[index]:
            fail(
                errors,
                f"{dialogue_id} turn {index + 1}: expected_changed_from_previous "
                f"{turn.get('expected_changed_from_previous')!r} != {expected_changed[index]!r}",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-dialogues", type=int, default=100)
    args = parser.parse_args()

    dialogues = read_jsonl(args.dataset)
    errors: list[str] = []
    if len(dialogues) != args.expected_dialogues:
        fail(errors, f"expected {args.expected_dialogues} dialogues; found {len(dialogues)}")

    ids = [dialogue.get("id") for dialogue in dialogues]
    duplicate_ids = [item for item, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        fail(errors, f"duplicate dialogue ids: {', '.join(map(str, duplicate_ids[:10]))}")

    for dialogue in dialogues:
        validate_dialogue(dialogue, errors)

    pattern_counts = Counter(dialogue.get("gold_pattern") for dialogue in dialogues)
    type_counts = Counter(dialogue.get("dialogue_type") for dialogue in dialogues)
    domain_counts = Counter(dialogue.get("domain") for dialogue in dialogues)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ClosureBench-Dyn Validation",
        "",
        f"Dataset: `{args.dataset}`.",
        f"Dialogues: {len(dialogues)}.",
        f"Turns: {sum(len(dialogue.get('turns', [])) for dialogue in dialogues)}.",
        f"Errors: {len(errors)}.",
        "",
        "## Gold Trajectories",
        "",
        "| Pattern | Count |",
        "|---|---:|",
    ]
    for pattern, count in sorted(pattern_counts.items()):
        lines.append(f"| `{pattern}` | {count} |")
    lines.extend(["", "## Dialogue Types", "", "| Type | Count |", "|---|---:|"])
    for dialogue_type, count in sorted(type_counts.items()):
        lines.append(f"| `{dialogue_type}` | {count} |")
    lines.extend(["", "## Domains", "", "| Domain | Count |", "|---|---:|"])
    for domain, count in sorted(domain_counts.items()):
        lines.append(f"| `{domain}` | {count} |")
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors[:100])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.report}")
    if errors:
        print(f"Validation failed with {len(errors)} errors.", file=sys.stderr)
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
