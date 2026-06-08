#!/usr/bin/env python3
"""Generate the Closure-Contract benchmark dataset.

Builds a dataset with explicit formal semantics, symbolic atoms/rules, gold
reason types, and train/dev/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_JSONL = DATA_DIR / "closurebench_base.jsonl"
DEFAULT_META = DATA_DIR / "closurebench_base_metadata.json"


DOMAINS = [
    {
        "name": "hospital_access",
        "entities": ["Ava", "Ben", "Cora", "Dylan", "Elena", "Farid"],
        "base": ("completed_training", "has completed safety training"),
        "closed": ("has_clearance", "has security clearance"),
        "open": ("assigned_to_trial", "is assigned to the clinical trial"),
        "derived_closed": ("may_enter_lab", "may enter Lab A"),
        "derived_open": ("needs_supervisor_review", "needs supervisor review"),
        "rule_closed": "If a staff member has security clearance and has completed safety training, then that staff member may enter Lab A.",
        "rule_open": "If a staff member is assigned to the clinical trial, then that staff member needs supervisor review.",
    },
    {
        "name": "university_registration",
        "entities": ["Mia", "Noah", "Lena", "Omar", "Iris", "Theo"],
        "base": ("passed_exam", "passed the qualifying exam"),
        "closed": ("has_advising_clearance", "has advising clearance"),
        "open": ("has_external_scholarship", "has an external scholarship"),
        "derived_closed": ("may_register", "may register for the seminar"),
        "derived_open": ("needs_manual_audit", "needs a manual audit"),
        "rule_closed": "If a student passed the qualifying exam and has advising clearance, then that student may register for the seminar.",
        "rule_open": "If a student has an external scholarship, then that student needs a manual audit.",
    },
    {
        "name": "manufacturing_quality",
        "entities": ["Part A", "Part B", "Part C", "Part D", "Part E", "Part F"],
        "base": ("passed_visual_check", "passed the visual check"),
        "closed": ("has_release_approval", "has release approval"),
        "open": ("has_supplier_note", "has a supplier note"),
        "derived_closed": ("may_ship", "may ship to customers"),
        "derived_open": ("needs_engineer_review", "needs engineer review"),
        "rule_closed": "If a part passed the visual check and has release approval, then that part may ship to customers.",
        "rule_open": "If a part has a supplier note, then that part needs engineer review.",
    },
    {
        "name": "finance_controls",
        "entities": ["Account 12", "Account 27", "Account 35", "Account 48", "Account 59", "Account 61"],
        "base": ("passed_kyc", "passed KYC checks"),
        "closed": ("has_compliance_clearance", "has compliance clearance"),
        "open": ("has_manual_exception", "has a manual exception"),
        "derived_closed": ("may_receive_wire", "may receive a wire transfer"),
        "derived_open": ("needs_compliance_review", "needs compliance review"),
        "rule_closed": "If an account passed KYC checks and has compliance clearance, then that account may receive a wire transfer.",
        "rule_open": "If an account has a manual exception, then that account needs compliance review.",
    },
    {
        "name": "robotics_operations",
        "entities": ["Rover 1", "Rover 2", "Rover 3", "Rover 4", "Rover 5", "Rover 6"],
        "base": ("passed_diagnostics", "passed diagnostics"),
        "closed": ("has_shift_clearance", "has shift clearance"),
        "open": ("has_field_note", "has a field note"),
        "derived_closed": ("may_start_shift", "may start its shift"),
        "derived_open": ("needs_operator_review", "needs operator review"),
        "rule_closed": "If a robot passed diagnostics and has shift clearance, then that robot may start its shift.",
        "rule_open": "If a robot has a field note, then that robot needs operator review.",
    },
    {
        "name": "library_services",
        "entities": ["Reader A", "Reader B", "Reader C", "Reader D", "Reader E", "Reader F"],
        "base": ("has_active_membership", "has an active membership"),
        "closed": ("has_borrowing_clearance", "has borrowing clearance"),
        "open": ("has_special_request", "has a special request"),
        "derived_closed": ("may_borrow_archive_item", "may borrow an archive item"),
        "derived_open": ("needs_librarian_review", "needs librarian review"),
        "rule_closed": "If a reader has an active membership and has borrowing clearance, then that reader may borrow an archive item.",
        "rule_open": "If a reader has a special request, then that reader needs librarian review.",
    },
    {
        "name": "cloud_deployment",
        "entities": ["Service A", "Service B", "Service C", "Service D", "Service E", "Service F"],
        "base": ("passed_unit_tests", "passed unit tests"),
        "closed": ("has_deployment_approval", "has deployment approval"),
        "open": ("has_incident_note", "has an incident note"),
        "derived_closed": ("may_deploy", "may deploy to production"),
        "derived_open": ("needs_sre_review", "needs SRE review"),
        "rule_closed": "If a service passed unit tests and has deployment approval, then that service may deploy to production.",
        "rule_open": "If a service has an incident note, then that service needs SRE review.",
    },
    {
        "name": "procurement_review",
        "entities": ["Vendor A", "Vendor B", "Vendor C", "Vendor D", "Vendor E", "Vendor F"],
        "base": ("submitted_tax_form", "submitted a tax form"),
        "closed": ("has_vendor_approval", "has vendor approval"),
        "open": ("has_risk_note", "has a risk note"),
        "derived_closed": ("may_receive_purchase_order", "may receive a purchase order"),
        "derived_open": ("needs_procurement_review", "needs procurement review"),
        "rule_closed": "If a vendor submitted a tax form and has vendor approval, then that vendor may receive a purchase order.",
        "rule_open": "If a vendor has a risk note, then that vendor needs procurement review.",
    },
]


SEMANTICS = {
    "owa": {
        "name": "Open-world assumption",
        "instruction": "Predicates are open by default. A target atom that is not explicitly stated, explicitly negated, or derivable from the rules is unknown.",
    },
    "cwa": {
        "name": "Closed-world assumption",
        "instruction": "All predicates in the vocabulary are complete. After applying the rules, every unstated and underivable atom is false.",
    },
    "lcwa": {
        "name": "Locally closed-world assumption",
        "instruction": "Only the predicates listed as complete are closed. After applying the rules, an unstated and underivable atom whose predicate is complete is false; an unstated and underivable atom whose predicate is not complete is unknown.",
    },
}


FAMILIES = [
    "closed_missing_direct",
    "open_missing_direct",
    "closed_missing_with_open_distractor",
    "open_missing_with_closed_distractor",
    "closed_derived_missing_antecedent",
    "open_derived_missing_antecedent",
    "entailed_closed_conclusion",
    "entailed_open_conclusion",
    "explicit_negative_closed",
    "explicit_negative_open",
]


@dataclass(frozen=True)
class Atom:
    predicate: str
    entity: str

    def key(self) -> str:
        return f"{self.predicate}::{self.entity}"


@dataclass(frozen=True)
class Rule:
    antecedents: tuple[Atom, ...]
    conclusion: Atom
    text: str


def atom_sentence(atom: Atom, glossary: dict[str, str]) -> str:
    return f"{atom.entity} {glossary[atom.predicate]}."


def neg_sentence(atom: Atom, glossary: dict[str, str]) -> str:
    phrase = glossary[atom.predicate]
    if phrase.startswith("has "):
        return f"{atom.entity} does not {phrase}."
    if phrase.startswith("is "):
        return f"{atom.entity} is not {phrase.removeprefix('is ')}."
    if phrase.startswith("needs "):
        return f"{atom.entity} does not {phrase}."
    if phrase.startswith("may "):
        return f"{atom.entity} may not {phrase.removeprefix('may ')}."
    return f"It is false that {atom.entity} {phrase}."


def parse_atom(key: str) -> Atom:
    predicate, entity = key.split("::", 1)
    return Atom(predicate, entity)


def forward_chain(positive: set[Atom], rules: list[Rule]) -> set[Atom]:
    closure = set(positive)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.conclusion not in closure and all(atom in closure for atom in rule.antecedents):
                closure.add(rule.conclusion)
                changed = True
    return closure


def split_for_base(base_id: str) -> str:
    bucket = int(hashlib.sha256(base_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def gold_for(
    query: Atom,
    positive: set[Atom],
    negative: set[Atom],
    rules: list[Rule],
    semantics: str,
    closed_predicates: set[str],
) -> tuple[str, str, list[str]]:
    closure = forward_chain(positive, rules)
    if query in closure:
        return "true", "entailed_by_fact_or_rule", sorted(atom.key() for atom in closure)
    if query in negative:
        return "false", "explicit_negative_fact", sorted(atom.key() for atom in closure)
    if semantics == "owa":
        return "unknown", "open_world_underivable", sorted(atom.key() for atom in closure)
    if semantics == "cwa":
        return "false", "global_closed_world_underivable", sorted(atom.key() for atom in closure)
    if query.predicate in closed_predicates:
        return "false", "local_closed_world_underivable", sorted(atom.key() for atom in closure)
    return "unknown", "local_open_world_underivable", sorted(atom.key() for atom in closure)


def build_prompt(item: dict) -> str:
    positive_facts = "\n".join(f"- {fact}" for fact in item["facts_positive"]) or "- None"
    negative_facts = "\n".join(f"- {fact}" for fact in item["facts_negative"]) or "- None"
    rules = "\n".join(f"- {rule}" for rule in item["rules_natural"]) or "- None"
    glossary = "\n".join(
        f"- {predicate}: {text}" for predicate, text in sorted(item["predicate_glossary"].items())
    )
    closed = ", ".join(item["closed_predicates"]) if item["closed_predicates"] else "none"
    return (
        "You are evaluating a formal knowledge-base statement.\n"
        "Return exactly one JSON object with keys answer and rationale.\n"
        "The answer must be one of: true, false, unknown.\n\n"
        "Formal rule semantics:\n"
        "- Positive facts are true.\n"
        "- Negative facts make the corresponding target atom false.\n"
        "- A rule derives its conclusion only when every antecedent is true.\n"
        "- First apply all rules until no new facts can be derived.\n"
        "- Then apply the semantic contract to unstated and underivable atoms.\n\n"
        f"Semantic contract: {item['semantic_contract']['name']}.\n"
        f"{item['semantic_contract']['instruction']}\n"
        f"Complete predicates for this item: {closed}.\n\n"
        "Important closure rule:\n"
        "For a complete predicate P, if P(entity) is not explicitly stated and cannot be derived after applying the rules, then P(entity) is false. This does not require an explicit negative fact.\n\n"
        "Predicate glossary:\n"
        f"{glossary}\n\n"
        "Positive facts:\n"
        f"{positive_facts}\n\n"
        "Negative facts:\n"
        f"{negative_facts}\n\n"
        "Rules:\n"
        f"{rules}\n\n"
        f"Target atom: {item['query_atom']}.\n"
        f"Target statement: {item['target_statement']}"
    )


def make_base(domain: dict, family: str, index: int) -> dict:
    entities = domain["entities"]
    e0 = entities[index % len(entities)]
    e1 = entities[(index + 1) % len(entities)]
    e2 = entities[(index + 2) % len(entities)]

    base_pred, base_text = domain["base"]
    closed_pred, closed_text = domain["closed"]
    open_pred, open_text = domain["open"]
    dclosed_pred, dclosed_text = domain["derived_closed"]
    dopen_pred, dopen_text = domain["derived_open"]

    glossary = {
        base_pred: base_text,
        closed_pred: closed_text,
        open_pred: open_text,
        dclosed_pred: dclosed_text,
        dopen_pred: dopen_text,
    }
    positive: set[Atom] = set()
    negative: set[Atom] = set()
    rules: list[Rule] = []
    query: Atom
    lcwa_closed: set[str]
    subset = "core_contrastive"

    def add_closed_rule(entity: str) -> None:
        rules.append(
            Rule(
                antecedents=(Atom(base_pred, entity), Atom(closed_pred, entity)),
                conclusion=Atom(dclosed_pred, entity),
                text=domain["rule_closed"],
            )
        )

    def add_open_rule(entity: str) -> None:
        rules.append(
            Rule(
                antecedents=(Atom(open_pred, entity),),
                conclusion=Atom(dopen_pred, entity),
                text=domain["rule_open"],
            )
        )

    if family == "closed_missing_direct":
        positive.add(Atom(base_pred, e0))
        query = Atom(closed_pred, e0)
        lcwa_closed = {closed_pred}
    elif family == "open_missing_direct":
        positive.add(Atom(base_pred, e0))
        query = Atom(open_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
    elif family == "closed_missing_with_open_distractor":
        positive.update({Atom(base_pred, e0), Atom(open_pred, e1)})
        query = Atom(closed_pred, e0)
        lcwa_closed = {closed_pred}
    elif family == "open_missing_with_closed_distractor":
        positive.update({Atom(base_pred, e1), Atom(closed_pred, e1)})
        query = Atom(open_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
    elif family == "closed_derived_missing_antecedent":
        positive.add(Atom(base_pred, e0))
        add_closed_rule(e0)
        query = Atom(dclosed_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
    elif family == "open_derived_missing_antecedent":
        positive.add(Atom(base_pred, e0))
        add_open_rule(e0)
        query = Atom(dopen_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
    elif family == "entailed_closed_conclusion":
        positive.update({Atom(base_pred, e0), Atom(closed_pred, e0)})
        add_closed_rule(e0)
        query = Atom(dclosed_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
        subset = "control_entailed"
    elif family == "entailed_open_conclusion":
        positive.add(Atom(open_pred, e0))
        add_open_rule(e0)
        query = Atom(dopen_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
        subset = "control_entailed"
    elif family == "explicit_negative_closed":
        positive.add(Atom(base_pred, e2))
        negative.add(Atom(closed_pred, e0))
        query = Atom(closed_pred, e0)
        lcwa_closed = {closed_pred}
        subset = "control_explicit_negative"
    elif family == "explicit_negative_open":
        positive.add(Atom(base_pred, e2))
        negative.add(Atom(open_pred, e0))
        query = Atom(open_pred, e0)
        lcwa_closed = {closed_pred, dclosed_pred}
        subset = "control_explicit_negative"
    else:
        raise ValueError(f"unknown family: {family}")

    return {
        "base_id": f"{domain['name']}_{index:04d}_{family}",
        "domain": domain["name"],
        "family": family,
        "subset": subset,
        "positive_atoms": sorted(atom.key() for atom in positive),
        "negative_atoms": sorted(atom.key() for atom in negative),
        "rules": [
            {
                "antecedents": [atom.key() for atom in rule.antecedents],
                "conclusion": rule.conclusion.key(),
                "text": rule.text,
            }
            for rule in rules
        ],
        "query_atom": query.key(),
        "target_statement": atom_sentence(query, glossary),
        "predicate_glossary": glossary,
        "vocabulary_predicates": sorted(glossary),
        "lcwa_closed_predicates": sorted(lcwa_closed),
    }


def instantiate(base: dict, semantics: str) -> dict:
    positive = {parse_atom(key) for key in base["positive_atoms"]}
    negative = {parse_atom(key) for key in base["negative_atoms"]}
    rules = [
        Rule(
            antecedents=tuple(parse_atom(key) for key in raw["antecedents"]),
            conclusion=parse_atom(raw["conclusion"]),
            text=raw["text"],
        )
        for raw in base["rules"]
    ]
    query = parse_atom(base["query_atom"])
    if semantics == "owa":
        closed_predicates: set[str] = set()
    elif semantics == "cwa":
        closed_predicates = set(base["vocabulary_predicates"])
    else:
        closed_predicates = set(base["lcwa_closed_predicates"])

    gold, reason_type, closure_atoms = gold_for(query, positive, negative, rules, semantics, closed_predicates)

    item = {
        "id": f"{base['base_id']}__{semantics}",
        "base_id": base["base_id"],
        "split": split_for_base(base["base_id"]),
        "domain": base["domain"],
        "family": base["family"],
        "subset": base["subset"],
        "semantics": semantics,
        "semantic_contract": SEMANTICS[semantics],
        "closed_predicates": sorted(closed_predicates),
        "facts_positive": [atom_sentence(parse_atom(key), base["predicate_glossary"]) for key in base["positive_atoms"]],
        "facts_negative": [neg_sentence(parse_atom(key), base["predicate_glossary"]) for key in base["negative_atoms"]],
        "rules_natural": [raw["text"] for raw in base["rules"]],
        "predicate_glossary": base["predicate_glossary"],
        "vocabulary_predicates": base["vocabulary_predicates"],
        "query_atom": base["query_atom"],
        "target_statement": base["target_statement"],
        "gold_answer": gold,
        "gold_reason_type": reason_type,
        "symbolic": {
            "positive_atoms": base["positive_atoms"],
            "negative_atoms": base["negative_atoms"],
            "rules": base["rules"],
            "query_atom": base["query_atom"],
            "closure_atoms": closure_atoms,
        },
    }
    item["prompt"] = build_prompt(item)
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--per-domain-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    bases = []
    index = 0
    for domain in DOMAINS:
        for family in FAMILIES:
            for _ in range(args.per_domain_family):
                bases.append(make_base(domain, family, index))
                index += 1

    items = []
    for base in bases:
        for semantics in ("owa", "cwa", "lcwa"):
            items.append(instantiate(base, semantics))

    random.Random(args.seed).shuffle(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_base = defaultdict(dict)
    for item in items:
        by_base[item["base_id"]][item["semantics"]] = item["gold_answer"]
    pattern_counts = Counter(tuple(group[sem] for sem in ("owa", "cwa", "lcwa")) for group in by_base.values())
    meta = {
        "dataset": args.out.name,
        "version": "closurebench-v1",
        "num_items": len(items),
        "num_base_scenarios": len(bases),
        "domains": Counter(item["domain"] for item in items),
        "families": Counter(item["family"] for item in items),
        "subsets": Counter(item["subset"] for item in items),
        "splits": Counter(item["split"] for item in items),
        "semantics": Counter(item["semantics"] for item in items),
        "gold_answers": Counter(item["gold_answer"] for item in items),
        "gold_reason_types": Counter(item["gold_reason_type"] for item in items),
        "semantic_switch_patterns": {"|".join(key): value for key, value in sorted(pattern_counts.items())},
        "primary_metrics": [
            "semantic_switch_accuracy",
            "lcwa_closed_scope_accuracy",
            "lcwa_open_scope_accuracy",
            "cwa_accuracy",
            "owa_accuracy",
        ],
        "notes": [
            "Each base scenario has OWA, CWA, and LCWA variants with identical facts, rules, and target atom.",
            "Each item contains natural-language text plus symbolic atoms/rules and a gold reason type.",
            "Closure is explicit: closed predicates make unstated and underivable atoms false after rule closure.",
            "Controls are retained but marked separately from core contrastive cases.",
        ],
    }
    with args.metadata.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False, default=dict)

    print(f"Wrote {len(items)} items to {args.out}")
    print(f"Wrote metadata to {args.metadata}")


if __name__ == "__main__":
    main()

