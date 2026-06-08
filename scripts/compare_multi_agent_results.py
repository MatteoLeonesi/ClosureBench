#!/usr/bin/env python3
"""Compare ClosureBench-MA model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_multi_agent_results import read_jsonl, summarize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_multi_agent.jsonl"
DEFAULT_MANIFEST = ROOT / "reports" / "closurebench_multi_agent_manifest.json"
DEFAULT_REPORT_JSON = ROOT / "reports" / "closurebench_multi_agent_summary.json"
DEFAULT_REPORT_MD = ROOT / "reports" / "closurebench_multi_agent_summary.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for run in manifest["runs"]:
        report = summarize(dataset_rows, read_jsonl(Path(run["scored"])))
        rows.append(
            {
                "label": run["label"],
                "model": run["model"],
                "repeat": run.get("repeat"),
                "temperature": run.get("temperature"),
                "report": report,
            }
        )

    comparison = {"dataset": str(args.dataset), "n_dataset": len(dataset_rows), "runs": rows}
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# ClosureBench-MA Model Comparison",
        "",
        f"Dataset: `{args.dataset}` ({len(dataset_rows)} items).",
        "",
        "| Model | Repeat | Full | Truth | Source | Closure | Core switch full | Closed absence | Open absence | Leakage | Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report = row["report"]
        repeat = row["repeat"] if row["repeat"] is not None else 1
        lines.append(
            f"| `{row['label']}` | "
            f"{repeat} | "
            f"{report['full_accuracy']['accuracy']:.2f}% | "
            f"{report['truth_accuracy']['accuracy']:.2f}% | "
            f"{report['source_accuracy']['accuracy']:.2f}% | "
            f"{report['closure_accuracy']['accuracy']:.2f}% | "
            f"{report['core_switch_full_accuracy']['accuracy']:.2f}% | "
            f"{report['closed_absence_accuracy']['accuracy']:.2f}% | "
            f"{report['open_absence_accuracy']['accuracy']:.2f}% | "
            f"{report['closure_leakage_rate']['rate']:.2f}% | "
            f"{report['closure_loss_rate']['rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Closure leakage means an open/incomplete source absence was treated as closed/false.",
            "Closure loss means an authoritative complete-source absence was treated as open/unknown.",
        ]
    )
    args.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_md}")


if __name__ == "__main__":
    main()
