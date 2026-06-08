#!/usr/bin/env python3
"""Generate ClosureBench-MA, a source-scoped multi-agent extension.

ClosureBench-MA keeps the base symbolic facts, rules, target atoms, and
OWA/CWA/LCWA labels, but presents evidence as reports from two source agents.
The coordinator must preserve which source/predicate is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from generate_base_dataset import Rule, forward_chain, parse_atom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "closurebench_multi_agent.jsonl"
DEFAULT_METADATA = ROOT / "data" / "closurebench_multi_agent_metadata.json"
DEFAULT_DESIGN = ROOT / "reports" / "closurebench_multi_agent_design.md"

SOURCE_VALUES = {"agent_a", "agent_b", "both", "neither"}
CLOSURE_HANDLINGS = {"entailed_or_explicit", "explicit_negative", "closed_absence", "open_absence"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def atom_key_source(atom_key: str, complete_predicates: set[str]) -> str:
    predicate, _ = atom_key.split("::", 1)
    return "agent_a" if predicate in complete_predicates else "agent_b"


def combine_sources(sources: set[str]) -> str:
    clean = {source for source in sources if source in {"agent_a", "agent_b"}}
    if clean == {"agent_a"}:
        return "agent_a"
    if clean == {"agent_b"}:
        return "agent_b"
    if clean == {"agent_a", "agent_b"}:
        return "both"
    return "neither"


def build_rules(raw_rules: list[dict]) -> list[Rule]:
    return [
        Rule(
            antecedents=tuple(parse_atom(key) for key in raw["antecedents"]),
            conclusion=parse_atom(raw["conclusion"]),
            text=raw["text"],
        )
        for raw in raw_rules
    ]


def source_for_true_query(item: dict, complete_predicates: set[str]) -> str:
    symbolic = item["symbolic"]
    query = parse_atom(symbolic["query_atom"])
    positive = {parse_atom(key) for key in symbolic["positive_atoms"]}
    rules = build_rules(symbolic["rules"])
    closure = forward_chain(positive, rules)

    if query.key() in symbolic["positive_atoms"]:
        return atom_key_source(query.key(), complete_predicates)

    for rule in rules:
        if rule.conclusion == query and all(atom in closure for atom in rule.antecedents):
            return combine_sources({atom_key_source(atom.key(), complete_predicates) for atom in rule.antecedents})

    return "neither"


def gold_fields(item: dict) -> dict:
    complete_predicates = set(item["closed_predicates"])
    truth = item["gold_answer"]
    reason = item["gold_reason_type"]

    if reason == "entailed_by_fact_or_rule":
        source_used = source_for_true_query(item, complete_predicates)
        closure_handling = "entailed_or_explicit"
    elif reason == "explicit_negative_fact":
        source_used = atom_key_source(item["query_atom"], complete_predicates)
        closure_handling = "explicit_negative"
    elif reason in {"global_closed_world_underivable", "local_closed_world_underivable"}:
        source_used = "agent_a"
        closure_handling = "closed_absence"
    elif reason in {"open_world_underivable", "local_open_world_underivable"}:
        source_used = "agent_b"
        closure_handling = "open_absence"
    else:
        raise ValueError(f"Unhandled reason type: {reason}")

    return {
        "gold_truth_value": truth,
        "gold_source_used": source_used,
        "gold_closure_handling": closure_handling,
    }


def bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- none"


def glossary_text(glossary: dict[str, str]) -> str:
    return "\n".join(f"- {predicate}: {glossary[predicate]}" for predicate in sorted(glossary))


def rules_text(rules: list[str]) -> str:
    return "\n".join(f"- {rule}" for rule in rules) if rules else "- none"


def source_reports(item: dict) -> dict:
    complete_predicates = set(item["closed_predicates"])
    reports = {
        "agent_a": {"positive": [], "negative": []},
        "agent_b": {"positive": [], "negative": []},
    }
    for fact, atom_key in zip(item["facts_positive"], item["symbolic"]["positive_atoms"], strict=True):
        reports[atom_key_source(atom_key, complete_predicates)]["positive"].append(fact)
    for fact, atom_key in zip(item["facts_negative"], item["symbolic"]["negative_atoms"], strict=True):
        reports[atom_key_source(atom_key, complete_predicates)]["negative"].append(fact)
    return reports


def complete_predicates_text(predicates: list[str]) -> str:
    return ", ".join(predicates) if predicates else "none"


def build_prompt(item: dict) -> str:
    reports = item["source_reports"]
    complete = complete_predicates_text(item["agent_a_complete_predicates"])
    open_predicates = complete_predicates_text(item["agent_b_open_predicates"])
    return (
        "You are the coordinator in a multi-agent workflow. Two source agents reported evidence from different "
        "systems. Your job is to answer the target statement while preserving source-scoped closure contracts.\n"
        "Return exactly one JSON object with keys truth_value, source_used, closure_handling, and rationale.\n\n"
        "Allowed values:\n"
        "- truth_value: true, false, unknown.\n"
        "- source_used: agent_a, agent_b, both, neither.\n"
        "- closure_handling: entailed_or_explicit, explicit_negative, closed_absence, open_absence.\n\n"
        "Source attribution rule:\n"
        "- Use agent_a when an Agent A complete-source fact, negative fact, or closed absence determines the answer.\n"
        "- Use agent_b when an Agent B open-source fact, negative fact, or open absence determines the answer.\n"
        "- Use both when a rule-derived answer depends on facts from both source agents.\n"
        "- Use neither only if no source report or source-scoped absence is relevant.\n\n"
        "Source-scoped closure contract:\n"
        "- Agent A is an authoritative system of record only for the listed complete predicates.\n"
        "- For Agent A complete predicates, after applying rules, absence of a report means false.\n"
        "- Agent B is an incomplete/open source. Absence from Agent B never means false by itself.\n"
        "- Do not transfer Agent A completeness to Agent B-only/open predicates.\n"
        "- Do not ignore Agent A completeness for predicates it explicitly covers.\n\n"
        f"Agent A complete predicates: {complete}.\n"
        f"Agent B open/incomplete predicates: {open_predicates}.\n\n"
        "Predicate glossary:\n"
        f"{glossary_text(item['predicate_glossary'])}\n\n"
        "Coordinator policy rules:\n"
        f"{rules_text(item['rules_natural'])}\n\n"
        "Agent A report:\n"
        "Positive facts:\n"
        f"{bullet(reports['agent_a']['positive'])}\n"
        "Negative facts:\n"
        f"{bullet(reports['agent_a']['negative'])}\n\n"
        "Agent B report:\n"
        "Positive facts:\n"
        f"{bullet(reports['agent_b']['positive'])}\n"
        "Negative facts:\n"
        f"{bullet(reports['agent_b']['negative'])}\n\n"
        f"Target atom: {item['query_atom']}.\n"
        f"Target statement: {item['target_statement']}\n\n"
        "Decide the truth value, identify which source justified the decision, and classify how closure was handled."
    )


def select_base_ids(rows: list[dict], max_base_scenarios: int) -> set[str]:
    by_base: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_base[row["base_id"]].append(row)

    bases_by_subset: dict[str, list[str]] = defaultdict(list)
    for base_id, items in by_base.items():
        subsets = {item["subset"] for item in items}
        if len(subsets) != 1:
            raise ValueError(f"Mixed subset for {base_id}")
        subset = next(iter(subsets))
        bases_by_subset[subset].append(base_id)

    target_counts = {
        "core_contrastive": int(max_base_scenarios * 0.60),
        "control_entailed": int(max_base_scenarios * 0.20),
    }
    target_counts["control_explicit_negative"] = max_base_scenarios - sum(target_counts.values())

    selected: set[str] = set()
    for subset, count in target_counts.items():
        candidates = sorted(
            bases_by_subset[subset],
            key=lambda base_id: hashlib.sha256(f"closurebench-ma::{base_id}".encode()).hexdigest(),
        )
        selected.update(candidates[:count])
    return selected


def build_item(item: dict) -> dict:
    agent_a_complete = list(item["closed_predicates"])
    agent_b_open = [predicate for predicate in item["vocabulary_predicates"] if predicate not in set(agent_a_complete)]
    out = {
        "id": f"{item['id']}__multi_agent",
        "source_id": item["id"],
        "base_id": item["base_id"],
        "split": item["split"],
        "domain": item["domain"],
        "family": item["family"],
        "subset": item["subset"],
        "semantics": item["semantics"],
        "semantic_contract": item["semantic_contract"],
        "closed_predicates": item["closed_predicates"],
        "agent_a_complete_predicates": agent_a_complete,
        "agent_b_open_predicates": agent_b_open,
        "facts_positive": item["facts_positive"],
        "facts_negative": item["facts_negative"],
        "rules_natural": item["rules_natural"],
        "predicate_glossary": item["predicate_glossary"],
        "vocabulary_predicates": item["vocabulary_predicates"],
        "query_atom": item["query_atom"],
        "target_statement": item["target_statement"],
        "gold_reason_type": item["gold_reason_type"],
        "symbolic": item["symbolic"],
        "source_reports": source_reports(item),
        **gold_fields(item),
    }
    out["prompt"] = build_prompt(out)
    return out


def validate(items: list[dict]) -> None:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item["gold_source_used"] not in SOURCE_VALUES:
            raise ValueError(f"Bad source label in {item['id']}")
        if item["gold_closure_handling"] not in CLOSURE_HANDLINGS:
            raise ValueError(f"Bad closure label in {item['id']}")
        grouped[item["base_id"]].add(item["semantics"])
    incomplete = [base_id for base_id, semantics in grouped.items() if semantics != {"owa", "cwa", "lcwa"}]
    if incomplete:
        raise ValueError(f"Incomplete semantic triples, first: {incomplete[0]}")


def write_design(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ClosureBench-MA Design",
        "",
        "ClosureBench-MA is a source-scoped multi-agent extension of ClosureBench.",
        "It keeps the same symbolic target labels but distributes facts across two source agents.",
        "",
        "- Agent A is authoritative only for explicitly listed complete predicates.",
        "- Agent B is incomplete/open; absence from Agent B remains unknown.",
        "- The coordinator must not globalize or lose source-level completeness.",
        "",
        "Output labels:",
        "",
        "| Field | Values |",
        "|---|---|",
        "| `truth_value` | `true`, `false`, `unknown` |",
        "| `source_used` | `agent_a`, `agent_b`, `both`, `neither` |",
        "| `closure_handling` | `entailed_or_explicit`, `explicit_negative`, `closed_absence`, `open_absence` |",
        "",
        "Primary metrics:",
        "",
        "- Full decision accuracy: truth, source, and closure handling all correct.",
        "- Source attribution accuracy.",
        "- Closure handling accuracy.",
        "- Core distributed switch accuracy.",
        "- Closure leakage rate: open absence treated as closed/false.",
        "- Closure loss rate: closed absence treated as open/unknown.",
        "",
        "For `open_absence`, `agent_b` and `neither` are both accepted for source attribution: the open source determines that absence is unresolved, but there is no positive source evidence.",
        "",
        "Dataset summary:",
        "",
        f"- Items: {metadata['num_items']}",
        f"- Base scenarios: {metadata['num_base_scenarios']}",
        f"- Semantic counts: {metadata['semantics']}",
        f"- Subset counts: {metadata['subsets']}",
        f"- Gold closure handling: {metadata['gold_closure_handling']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--max-base-scenarios", type=int, default=120)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    selected = select_base_ids(rows, args.max_base_scenarios)
    items = [build_item(row) for row in rows if row["base_id"] in selected]
    items.sort(key=lambda row: row["id"])
    validate(items)

    metadata = {
        "name": "ClosureBench-MA",
        "source_dataset": str(args.input),
        "num_items": len(items),
        "num_base_scenarios": len({item["base_id"] for item in items}),
        "semantics": dict(Counter(item["semantics"] for item in items)),
        "subsets": dict(Counter(item["subset"] for item in items)),
        "gold_truth_values": dict(Counter(item["gold_truth_value"] for item in items)),
        "gold_source_used": dict(Counter(item["gold_source_used"] for item in items)),
        "gold_closure_handling": dict(Counter(item["gold_closure_handling"] for item in items)),
        "domains": dict(Counter(item["domain"] for item in items)),
    }

    write_jsonl(args.output, items)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_design(args.design, metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
