#!/usr/bin/env python3
"""Generate a dynamic-state dialogue extension for ClosureBench.

This extension avoids explicit OWA/CWA/LCWA labels in the turns. Instead, each
dialogue maintains a formal set of complete predicates and updates that set over
time. The model must preserve the facts/rules, apply only the state update, and
revise or persist its answer accordingly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "closurebench_dynamic_dialogue.jsonl"
DEFAULT_METADATA = ROOT / "data" / "closurebench_dynamic_dialogue_metadata.json"
DEFAULT_DESIGN = ROOT / "reports" / "closurebench_dynamic_dialogue_design.md"

SEMANTIC_ORDER = ("owa", "cwa", "lcwa")
PATTERN_PERSIST = "unknown|false|false"
PATTERN_REOPEN = "unknown|false|unknown"
TARGET_PATTERNS = (PATTERN_PERSIST, PATTERN_REOPEN)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_by_base(rows: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        grouped[row["base_id"]][row["semantics"]] = row
    return grouped


def pattern_for(group: dict[str, dict]) -> str:
    return "|".join(group[semantics]["gold_answer"] for semantics in SEMANTIC_ORDER)


def bullet(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def predicate_list(predicates: list[str]) -> str:
    if not predicates:
        return "none"
    return ", ".join(predicates)


def glossary_text(glossary: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {glossary[key]}" for key in sorted(glossary))


def rules_text(rules: list[str]) -> str:
    if not rules:
        return "- none"
    return "\n".join(f"- {rule}" for rule in rules)


def common_reasoning_contract() -> str:
    return "\n".join(
        [
            "Reasoning contract:",
            "- Positive facts are true.",
            "- Negative facts make the corresponding target atom false.",
            "- A rule derives its conclusion only when every antecedent is true.",
            "- First apply all rules until no new facts can be derived.",
            "- Then evaluate unstated and underivable atoms using the current completeness state.",
            "- If a predicate is complete, an unstated and underivable atom using that predicate is false.",
            "- If a predicate is not complete, an unstated and underivable atom using that predicate is unknown.",
            "- A false answer derived from completeness is not a permanent negative fact; it can be withdrawn if the completeness state changes.",
        ]
    )


def output_contract(turn_index: int, expected_update: str) -> str:
    changed_rule = (
        "For turn 1, changed_from_previous must be null."
        if turn_index == 1
        else "For this turn, changed_from_previous must be true iff your current answer differs from your immediately previous answer."
    )
    return (
        "Return exactly one compact JSON object with keys answer, changed_from_previous, applied_update, and rationale. "
        "The answer value must be one of true, false, unknown. "
        "The changed_from_previous value must be true, false, or null. "
        "The applied_update value must be one of open, global_complete, local_complete. "
        f"For this turn, applied_update must be exactly {expected_update}. "
        f"{changed_rule}"
    )


def kb_text(row: dict) -> str:
    return "\n".join(
        [
            "Knowledge base:",
            "",
            "Predicate glossary:",
            glossary_text(row["predicate_glossary"]),
            "",
            "Positive facts:",
            bullet(row["facts_positive"]),
            "",
            "Negative facts:",
            bullet(row["facts_negative"]),
            "",
            "Rules:",
            rules_text(row["rules_natural"]),
            "",
            f"Target statement: {row['target_statement']}",
            f"Target atom: {row['query_atom']}",
        ]
    )


def build_turn_prompt(turn_index: int, turn: dict, row: dict) -> str:
    complete = predicate_list(turn["complete_predicates_after_update"])
    update_lines = [f"- Operation: {turn['update_operation']}"]
    if turn["update_operation"] == "SET_COMPLETE_PREDICATES":
        update_lines.append(f"- New complete predicates: {complete}")
    elif turn["update_operation"] == "ADD_COMPLETE_PREDICATES":
        update_lines.append(f"- Predicates to add as complete: {predicate_list(turn['update_predicates'])}")
        update_lines.append(f"- Complete predicates after this update: {complete}")
    elif turn["update_operation"] == "REPLACE_COMPLETE_PREDICATES":
        update_lines.append("- Discard the previous complete-predicate set.")
        update_lines.append(f"- Replacement complete predicates: {complete}")
    else:
        raise ValueError(f"Unknown update operation: {turn['update_operation']}")

    if turn_index == 1:
        header = [
            "This is turn 1 of a 3-turn dynamic completeness dialogue.",
            "You are given an initial knowledge base and an initial completeness state.",
            "Do not use outside knowledge.",
        ]
        body = [
            *header,
            output_contract(turn_index, turn["expected_applied_update"]),
            "",
            common_reasoning_contract(),
            "",
            kb_text(row),
            "",
            "Completeness-state update for this turn:",
            *update_lines,
        ]
        return "\n".join(body)

    return "\n".join(
        [
            f"This is turn {turn_index} of a 3-turn dynamic completeness dialogue.",
            "Keep the exact same knowledge base, facts, rules, glossary, and target from turn 1.",
            "Apply only the completeness-state update below.",
            "Recompute the answer from the updated state. Do not treat a previous false/unknown conclusion as a new fact.",
            output_contract(turn_index, turn["expected_applied_update"]),
            "",
            common_reasoning_contract(),
            "",
            "Completeness-state update for this turn:",
            *update_lines,
            "",
            f"Target statement: {row['target_statement']}",
            f"Target atom: {row['query_atom']}",
        ]
    )


def build_dialogue(index: int, base_id: str, group: dict[str, dict]) -> dict:
    owa = group["owa"]
    cwa = group["cwa"]
    lcwa = group["lcwa"]
    pattern = pattern_for(group)

    turn_specs = [
        {
            "turn_index": 1,
            "source_semantics": "owa",
            "update_operation": "SET_COMPLETE_PREDICATES",
            "update_predicates": [],
            "complete_predicates_after_update": [],
            "gold_answer": owa["gold_answer"],
            "expected_changed_from_previous": None,
            "expected_applied_update": "open",
        },
        {
            "turn_index": 2,
            "source_semantics": "cwa",
            "update_operation": "ADD_COMPLETE_PREDICATES",
            "update_predicates": cwa["closed_predicates"],
            "complete_predicates_after_update": cwa["closed_predicates"],
            "gold_answer": cwa["gold_answer"],
            "expected_changed_from_previous": cwa["gold_answer"] != owa["gold_answer"],
            "expected_applied_update": "global_complete",
        },
        {
            "turn_index": 3,
            "source_semantics": "lcwa",
            "update_operation": "REPLACE_COMPLETE_PREDICATES",
            "update_predicates": lcwa["closed_predicates"],
            "complete_predicates_after_update": lcwa["closed_predicates"],
            "gold_answer": lcwa["gold_answer"],
            "expected_changed_from_previous": lcwa["gold_answer"] != cwa["gold_answer"],
            "expected_applied_update": "local_complete",
        },
    ]
    turns = []
    source_rows = {"owa": owa, "cwa": cwa, "lcwa": lcwa}
    for spec in turn_specs:
        source_row = source_rows[spec["source_semantics"]]
        turn = dict(spec)
        turn["prompt"] = build_turn_prompt(spec["turn_index"], spec, source_row)
        turns.append(turn)

    return {
        "id": f"dynamic_dialogue_v3_{index:04d}",
        "base_id": base_id,
        "split": owa["split"],
        "domain": owa["domain"],
        "family": owa["family"],
        "subset": owa["subset"],
        "gold_pattern": pattern,
        "dialogue_type": "reopen" if pattern == PATTERN_REOPEN else "persist_after_narrowing",
        "target_statement": owa["target_statement"],
        "query_atom": owa["query_atom"],
        "facts_positive": owa["facts_positive"],
        "facts_negative": owa["facts_negative"],
        "rules_natural": owa["rules_natural"],
        "predicate_glossary": owa["predicate_glossary"],
        "vocabulary_predicates": owa["vocabulary_predicates"],
        "symbolic": owa["symbolic"],
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--n-per-pattern", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    grouped = group_by_base(rows)
    candidates: dict[str, list[tuple[str, dict[str, dict]]]] = {pattern: [] for pattern in TARGET_PATTERNS}
    for base_id, group in grouped.items():
        if set(group) != set(SEMANTIC_ORDER):
            continue
        if group["owa"].get("subset") != "core_contrastive":
            continue
        pattern = pattern_for(group)
        if pattern in candidates:
            candidates[pattern].append((base_id, group))

    rng = random.Random(args.seed)
    selected: list[tuple[str, dict[str, dict]]] = []
    for pattern in TARGET_PATTERNS:
        pool = candidates[pattern]
        if len(pool) < args.n_per_pattern:
            raise ValueError(f"Need {args.n_per_pattern} candidates for {pattern}; found {len(pool)}")
        rng.shuffle(pool)
        selected.extend(pool[: args.n_per_pattern])
    selected.sort(key=lambda item: item[0])

    dialogues = [build_dialogue(index, base_id, group) for index, (base_id, group) in enumerate(selected)]
    write_jsonl(args.output, dialogues)

    pattern_counts = Counter(dialogue["gold_pattern"] for dialogue in dialogues)
    type_counts = Counter(dialogue["dialogue_type"] for dialogue in dialogues)
    domain_counts = Counter(dialogue["domain"] for dialogue in dialogues)
    family_counts = Counter(dialogue["family"] for dialogue in dialogues)
    metadata = {
        "source_dataset": str(args.input),
        "dataset": str(args.output),
        "n_dialogues": len(dialogues),
        "n_turns": len(dialogues) * 3,
        "turns_per_dialogue": 3,
        "seed": args.seed,
        "gold_patterns": dict(pattern_counts),
        "dialogue_types": dict(type_counts),
        "domains": dict(sorted(domain_counts.items())),
        "families": dict(sorted(family_counts.items())),
        "turn_update_sequence": [
            "SET_COMPLETE_PREDICATES([])",
            "ADD_COMPLETE_PREDICATES(vocabulary)",
            "REPLACE_COMPLETE_PREDICATES(local_subset)",
        ],
        "primary_metrics": [
            "dialogue_all_answer_accuracy",
            "revision_accuracy",
            "persistence_accuracy",
            "dialogue_all_turn_accuracy",
            "completeness_addition_accuracy",
            "reopening_accuracy",
            "narrowing_persistence_accuracy",
            "closure_inertia_rate",
            "closure_retention_failure_rate",
            "changed_flag_accuracy",
            "applied_update_accuracy",
            "anchoring_error_rate",
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    args.design.parent.mkdir(parents=True, exist_ok=True)
    design = [
        "# ClosureBench Dynamic Dialogue Extension",
        "",
        "This extension tests state updates rather than explicit semantic labels.",
        "Each dialogue keeps the same facts, rules, target, and glossary across three turns.",
        "Only the set of complete predicates changes.",
        "",
        "Turn sequence:",
        "",
        "1. `SET_COMPLETE_PREDICATES([])`: no predicate is complete; missing underivable atoms are unknown.",
        "2. `ADD_COMPLETE_PREDICATES(vocabulary)`: all vocabulary predicates become complete; missing underivable atoms are false.",
        "3. `REPLACE_COMPLETE_PREDICATES(local_subset)`: the global complete set is replaced by a local subset.",
        "",
        "Pattern distribution:",
        "",
        "| Gold pattern | Dialogues | Expected behavior |",
        "|---|---:|---|",
        f"| `unknown/false/false` | {pattern_counts[PATTERN_PERSIST]} | revise on turn 2, persist on turn 3 |",
        f"| `unknown/false/unknown` | {pattern_counts[PATTERN_REOPEN]} | revise on turn 2, reopen on turn 3 |",
        "",
        "Primary metrics:",
        "",
        "- Completeness addition accuracy: turn-2 accuracy after adding complete predicates.",
        "- Revision accuracy: transition-turn accuracy when the gold answer must change.",
        "- Persistence accuracy: transition-turn accuracy when the gold answer must stay the same.",
        "- Reopening accuracy: turn-3 accuracy when replacing global completeness withdraws the basis for false.",
        "- Narrowing persistence accuracy: turn-3 accuracy when local completeness still supports false.",
        "- Closure inertia rate: reopen turns where the model keeps false after the completeness basis has been withdrawn.",
        "- Closure retention failure rate: local-narrowing persistence turns where the model answers unknown even though local completeness still supports false.",
        "- Changed-flag accuracy: whether the model correctly reports answer changes across turns.",
        "- Applied-update accuracy: whether the model emits the correct dynamic update label for each turn.",
        "- Anchoring error rate: expected-revision turns where the model keeps the previous answer or says no change occurred.",
    ]
    with args.design.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(design) + "\n")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")
    print(f"Wrote {args.design}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
