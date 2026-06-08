#!/usr/bin/env python3
"""Generate ClosureBench-Ask/Act from the base benchmark dataset.

The extension keeps the same facts, rules, target atom, and semantic contracts
as the base benchmark, but changes the task from passive truth evaluation to an
operational decision:

- approve when the target is true;
- deny when the target is false;
- request_information when the target is unknown.

For unknown targets, the dataset also stores a symbolic missing atom that the
model should ask about.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from generate_base_dataset import Atom, Rule, forward_chain, parse_atom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "closurebench_ask_act.jsonl"
DEFAULT_METADATA = ROOT / "data" / "closurebench_ask_act_metadata.json"
DEFAULT_DESIGN = ROOT / "reports" / "closurebench_ask_act_design.md"

ACTION_FOR_TRUTH = {
    "true": "approve",
    "false": "deny",
    "unknown": "request_information",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def bullet(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def glossary_text(glossary: dict[str, str]) -> str:
    return "\n".join(f"- {predicate}: {glossary[predicate]}" for predicate in sorted(glossary))


def rules_text(rules: list[str]) -> str:
    if not rules:
        return "- none"
    return "\n".join(f"- {rule}" for rule in rules)


def predicate_list(predicates: list[str]) -> str:
    if not predicates:
        return "none"
    return ", ".join(predicates)


def build_rules(raw_rules: list[dict]) -> list[Rule]:
    return [
        Rule(
            antecedents=tuple(parse_atom(key) for key in raw["antecedents"]),
            conclusion=parse_atom(raw["conclusion"]),
            text=raw["text"],
        )
        for raw in raw_rules
    ]


def missing_atom_for_unknown(item: dict) -> Atom:
    """Return the most local missing atom to ask about for an unknown target."""

    symbolic = item["symbolic"]
    positive = {parse_atom(key) for key in symbolic["positive_atoms"]}
    negative = {parse_atom(key) for key in symbolic["negative_atoms"]}
    rules = build_rules(symbolic["rules"])
    query = parse_atom(symbolic["query_atom"])
    closure = forward_chain(positive, rules)

    candidates: list[Atom] = []
    for rule in rules:
        if rule.conclusion != query:
            continue
        missing = [
            atom
            for atom in rule.antecedents
            if atom not in closure and atom not in negative
        ]
        candidates.extend(missing)

    if candidates:
        return sorted(candidates, key=lambda atom: atom.key())[0]
    return query


def question_for_atom(atom: Atom, glossary: dict[str, str]) -> str:
    phrase = glossary.get(atom.predicate, atom.predicate.replace("_", " "))
    return f"Is it true that {atom.entity} {phrase}?"


def gold_fields(item: dict) -> dict:
    truth = item["gold_answer"]
    action = ACTION_FOR_TRUTH[truth]
    if action == "request_information":
        missing = missing_atom_for_unknown(item)
        missing_predicate = missing.predicate
        missing_entity = missing.entity
        question = question_for_atom(missing, item["predicate_glossary"])
        missing_atom = missing.key()
    else:
        missing_predicate = None
        missing_entity = None
        question = None
        missing_atom = None

    return {
        "gold_truth_value": truth,
        "gold_action": action,
        "gold_missing_predicate": missing_predicate,
        "gold_missing_entity": missing_entity,
        "gold_missing_atom": missing_atom,
        "gold_question_to_ask": question,
    }


def build_prompt(item: dict) -> str:
    closed = predicate_list(item["closed_predicates"])
    return (
        "You are evaluating an operational decision under a formal knowledge-base semantic contract.\n"
        "Return exactly one JSON object with keys truth_value, action, missing_predicate, missing_entity, "
        "question_to_ask, and rationale.\n\n"
        "Allowed values:\n"
        "- truth_value must be one of: true, false, unknown.\n"
        "- action must be one of: approve, deny, request_information.\n"
        "- If action is request_information, missing_predicate must be the exact predicate name to check, "
        "missing_entity must be the exact entity to check, and question_to_ask must ask for that missing atom.\n"
        "- If action is approve or deny, missing_predicate, missing_entity, and question_to_ask must be null.\n\n"
        "Action mapping:\n"
        "- approve iff the target statement is semantically true.\n"
        "- deny iff the target statement is semantically false.\n"
        "- request_information iff the target statement is semantically unknown.\n\n"
        "Formal rule semantics:\n"
        "- Positive facts are true.\n"
        "- Negative facts make the corresponding target atom false.\n"
        "- A rule derives its conclusion only when every antecedent is true.\n"
        "- First apply all rules until no new facts can be derived.\n"
        "- Then apply the semantic contract to unstated and underivable atoms.\n"
        "- For a complete predicate P, if P(entity) is not explicitly stated and cannot be derived, "
        "then P(entity) is false.\n"
        "- For an open predicate P, if P(entity) is not explicitly stated and cannot be derived, "
        "then P(entity) is unknown.\n\n"
        f"Semantic contract: {item['semantic_contract']['name']}.\n"
        f"{item['semantic_contract']['instruction']}\n"
        f"Complete predicates for this item: {closed}.\n\n"
        "Predicate glossary:\n"
        f"{glossary_text(item['predicate_glossary'])}\n\n"
        "Positive facts:\n"
        f"{bullet(item['facts_positive'])}\n\n"
        "Negative facts:\n"
        f"{bullet(item['facts_negative'])}\n\n"
        "Rules:\n"
        f"{rules_text(item['rules_natural'])}\n\n"
        f"Target atom: {item['query_atom']}.\n"
        f"Target statement: {item['target_statement']}\n\n"
        "Decide whether to approve the target statement, deny it, or request more information."
    )


def build_ask_act_item(item: dict) -> dict:
    gold = gold_fields(item)
    out = {
        "id": f"{item['id']}__ask_act",
        "source_id": item["id"],
        "base_id": item["base_id"],
        "split": item["split"],
        "domain": item["domain"],
        "family": item["family"],
        "subset": item["subset"],
        "semantics": item["semantics"],
        "semantic_contract": item["semantic_contract"],
        "closed_predicates": item["closed_predicates"],
        "facts_positive": item["facts_positive"],
        "facts_negative": item["facts_negative"],
        "rules_natural": item["rules_natural"],
        "predicate_glossary": item["predicate_glossary"],
        "vocabulary_predicates": item["vocabulary_predicates"],
        "query_atom": item["query_atom"],
        "target_statement": item["target_statement"],
        "gold_reason_type": item["gold_reason_type"],
        "symbolic": item["symbolic"],
        **gold,
    }
    out["prompt"] = build_prompt(out)
    return out


def validate_items(items: list[dict]) -> None:
    for item in items:
        expected_action = ACTION_FOR_TRUTH[item["gold_truth_value"]]
        if item["gold_action"] != expected_action:
            raise ValueError(f"{item['id']} has inconsistent action mapping")
        if item["gold_action"] == "request_information":
            if not item["gold_missing_predicate"] or not item["gold_missing_entity"]:
                raise ValueError(f"{item['id']} request item has no missing target")
        elif item["gold_missing_predicate"] is not None or item["gold_missing_entity"] is not None:
            raise ValueError(f"{item['id']} non-request item has missing target fields")

    grouped: dict[str, set[str]] = defaultdict(set)
    for item in items:
        grouped[item["base_id"]].add(item["semantics"])
    incomplete = [base_id for base_id, semantics in grouped.items() if semantics != {"owa", "cwa", "lcwa"}]
    if incomplete:
        raise ValueError(f"Found incomplete semantic triples, first: {incomplete[0]}")


def write_design(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ClosureBench-Ask/Act Extension",
        "",
        "ClosureBench-Ask/Act converts each base truth-label item into an operational decision item.",
        "The facts, rules, target atom, semantic contract, and symbolic verifier are unchanged.",
        "",
        "Gold mapping:",
        "",
        "| Truth value | Action | Missing fields |",
        "|---|---|---|",
        "| `true` | `approve` | null |",
        "| `false` | `deny` | null |",
        "| `unknown` | `request_information` | exact missing predicate/entity |",
        "",
        "For unknown derived targets, the missing field is the first underived rule antecedent rather than the target predicate itself.",
        "For unknown direct targets, the missing field is the target atom.",
        "",
        "Dataset summary:",
        "",
        f"- Items: {metadata['num_items']}",
        f"- Base scenarios: {metadata['num_base_scenarios']}",
        f"- Request-information items: {metadata['gold_actions'].get('request_information', 0)}",
        f"- Deny items: {metadata['gold_actions'].get('deny', 0)}",
        f"- Approve items: {metadata['gold_actions'].get('approve', 0)}",
        "",
        "Primary metrics:",
        "",
        "- Action accuracy.",
        "- Truth-value accuracy.",
        "- Full decision accuracy: truth, action, and targeted missing fields are correct.",
        "- Ask precision, recall, and F1.",
        "- Targeted information accuracy on gold request-information cases.",
        "- Premature denial rate on unresolved open-world cases.",
        "- Excessive ask rate on semantically determined cases.",
        "- Closure-denial accuracy on CWA/LCWA-closed false cases.",
        "- Semantic-switch action accuracy across OWA/CWA/LCWA triples.",
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    args = parser.parse_args()

    source_items = read_jsonl(args.input)
    ask_act_items = [build_ask_act_item(item) for item in source_items]
    validate_items(ask_act_items)
    write_jsonl(args.output, ask_act_items)

    grouped = defaultdict(set)
    for item in ask_act_items:
        grouped[item["base_id"]].add(item["semantics"])

    metadata = {
        "source_dataset": str(args.input),
        "dataset": str(args.output),
        "version": "closurebench_ask_act",
        "num_items": len(ask_act_items),
        "num_base_scenarios": len(grouped),
        "semantics": dict(Counter(item["semantics"] for item in ask_act_items)),
        "domains": dict(Counter(item["domain"] for item in ask_act_items)),
        "families": dict(Counter(item["family"] for item in ask_act_items)),
        "subsets": dict(Counter(item["subset"] for item in ask_act_items)),
        "gold_truth_values": dict(Counter(item["gold_truth_value"] for item in ask_act_items)),
        "gold_actions": dict(Counter(item["gold_action"] for item in ask_act_items)),
        "gold_reason_types": dict(Counter(item["gold_reason_type"] for item in ask_act_items)),
        "primary_metrics": [
            "action_accuracy",
            "truth_accuracy",
            "full_decision_accuracy",
            "ask_precision_recall_f1",
            "targeted_information_accuracy",
            "premature_denial_rate",
            "excessive_ask_rate",
            "closure_denial_accuracy",
            "semantic_switch_action_accuracy",
        ],
        "notes": [
            "This file is derived deterministically from ClosureBench.",
            "Unknown direct targets ask for the target atom.",
            "Unknown derived targets ask for the missing rule antecedent.",
            "Approve/deny cases require null missing fields to avoid unnecessary escalation.",
        ],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    write_design(args.design, metadata)

    print(f"Wrote {len(ask_act_items)} items to {args.output}")
    print(f"Wrote metadata to {args.metadata}")
    print(f"Wrote design to {args.design}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
