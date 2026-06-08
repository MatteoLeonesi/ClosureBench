#!/usr/bin/env python3
"""Run repeated ClosureBench-Dyn evaluations for configured models."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_dynamic_dialogue.jsonl"
DEFAULT_MODELS = ROOT / "configs" / "closurebench_dynamic_dialogue_models.example.json"
DEFAULT_MANIFEST = ROOT / "reports" / "closurebench_dynamic_dialogue_replicate_manifest.json"
SHARD_DIR = ROOT / "tmp" / "closurebench_dynamic_dialogue_replicate_shards"
SHARD_RESULTS_DIR = ROOT / "results" / "dynamic_dialogue_replicate_shards"
FINAL_RESULTS_DIR = ROOT / "results" / "dynamic_dialogue_replicates"
FINAL_REPORTS_DIR = ROOT / "reports" / "dynamic_dialogue_replicates"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def tag_fragment(run_tag: str) -> str:
    safe = sanitize_label(run_tag)
    return f"{safe}_" if safe else ""


def load_models(path: Path, only: list[str] | None) -> list[dict]:
    models = json.loads(path.read_text(encoding="utf-8"))
    if only:
        wanted = set(only)
        models = [model for model in models if model["label"] in wanted]
        missing = wanted.difference(model["label"] for model in models)
        if missing:
            raise ValueError(f"Missing labels in config: {', '.join(sorted(missing))}")
    return models


def split_shards(dataset: Path, label: str, repeat: int, n_shards: int, run_tag: str) -> list[Path]:
    rows = read_jsonl(dataset)
    paths = []
    safe = sanitize_label(label)
    safe_tag = sanitize_label(run_tag)
    for shard_index in range(n_shards):
        shard_rows = [row for index, row in enumerate(rows) if index % n_shards == shard_index]
        path = SHARD_DIR / safe_tag / safe / f"repeat_{repeat}" / f"shard_{shard_index}.jsonl"
        write_jsonl(path, shard_rows)
        paths.append(path)
    return paths


def turn_order(dataset: Path) -> list[str]:
    ordered = []
    for dialogue in read_jsonl(dataset):
        for turn in dialogue["turns"]:
            ordered.append(f"{dialogue['id']}__turn_{turn['turn_index']}")
    return ordered


def turn_count(dataset: Path) -> int:
    return len(turn_order(dataset))


def merge_shards(dataset: Path, shard_paths: list[Path], output_path: Path) -> None:
    rows_by_id = {}
    for path in shard_paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            existing = rows_by_id.get(row["id"])
            if (
                existing is None
                or (existing.get("error") is not None and row.get("error") is None)
                or (
                    existing.get("error") is None
                    and row.get("error") is None
                    and row.get("completed_at", "") >= existing.get("completed_at", "")
                )
            ):
                rows_by_id[row["id"]] = row
    write_jsonl(output_path, [rows_by_id[row_id] for row_id in turn_order(dataset) if row_id in rows_by_id])


def final_paths(label: str, repeat: int, run_tag: str) -> tuple[Path, Path]:
    safe = sanitize_label(label)
    stem = f"closurebench_dynamic_dialogue_{tag_fragment(run_tag)}{safe}_repeat_{repeat}"
    return FINAL_RESULTS_DIR / f"{stem}_outputs.jsonl", FINAL_RESULTS_DIR / f"{stem}_scored.jsonl"


def complete(scored_path: Path, expected_turns: int) -> bool:
    if not scored_path.exists():
        return False
    ok_ids = {row["id"] for row in read_jsonl(scored_path) if row.get("error") is None}
    return len(ok_ids) == expected_turns


def build_command(model: dict, shard_path: Path, outputs_path: Path, scored_path: Path, temperature: float) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_dynamic_dialogue_experiment.py"),
        "--dataset",
        str(shard_path),
        "--results",
        str(outputs_path),
        "--scored",
        str(scored_path),
        "--model",
        model["model"],
        "--temperature",
        str(temperature),
        "--timeout",
        str(model.get("timeout", 180)),
        "--max-tokens",
        str(model.get("max_tokens", 1024)),
        "--retries",
        str(model.get("retries", 4)),
        "--sleep",
        str(model.get("sleep", 0.02)),
    ]
    if model.get("base_url"):
        cmd.extend(["--base-url", model["base_url"]])
    if model.get("chat_url"):
        cmd.extend(["--chat-url", model["chat_url"]])
    if model.get("thinking"):
        cmd.extend(["--thinking", model["thinking"]])
    if model.get("response_format") is False:
        cmd.append("--no-response-format")
    if model.get("insecure_skip_verify"):
        cmd.append("--insecure-skip-verify")
    return cmd


def run_model_repeat(
    model: dict,
    dataset: Path,
    repeat: int,
    n_shards: int,
    expected_turns: int,
    temperature: float,
    run_tag: str,
) -> dict:
    label = model["label"]
    env_key = model["api_key_env"]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} is not set for model label {label}")

    final_outputs, final_scored = final_paths(label, repeat, run_tag)
    if complete(final_scored, expected_turns):
        print(f"[{label} r{repeat}] already complete", flush=True)
    else:
        shard_paths = split_shards(dataset, label, repeat, n_shards, run_tag)
        processes = []
        safe = sanitize_label(label)
        for shard_index, shard_path in enumerate(shard_paths):
            stem = f"{tag_fragment(run_tag)}{safe}_repeat_{repeat}_shard_{shard_index}"
            shard_outputs = SHARD_RESULTS_DIR / f"{stem}_outputs.jsonl"
            shard_scored = SHARD_RESULTS_DIR / f"{stem}_scored.jsonl"
            env = os.environ.copy()
            env["API_KEY"] = api_key
            cmd = build_command(model, shard_path, shard_outputs, shard_scored, temperature)
            print(f"[{label} r{repeat}] starting shard {shard_index + 1}/{n_shards}", flush=True)
            processes.append((subprocess.Popen(cmd, cwd=ROOT, env=env), shard_outputs, shard_scored))

        failures = []
        for process, _, shard_scored in processes:
            code = process.wait()
            if code != 0:
                failures.append((shard_scored, code))
        if failures:
            detail = ", ".join(f"{path}: exit {code}" for path, code in failures)
            raise RuntimeError(f"{label} repeat {repeat} failed: {detail}")

        merge_shards(dataset, [item[1] for item in processes], final_outputs)
        merge_shards(dataset, [item[2] for item in processes], final_scored)
        if not complete(final_scored, expected_turns):
            raise RuntimeError(f"{label} repeat {repeat} did not complete all {expected_turns} turns without API errors")

    report_json = FINAL_REPORTS_DIR / f"{final_scored.stem}_report.json"
    report_md = FINAL_REPORTS_DIR / f"{final_scored.stem}_report.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_dynamic_dialogue_results.py"),
            "--dataset",
            str(dataset),
            "--scored",
            str(final_scored),
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ],
        cwd=ROOT,
        check=True,
    )
    return {
        "label": label,
        "model": model["model"],
        "repeat": repeat,
        "temperature": temperature,
        "outputs": str(final_outputs),
        "scored": str(final_scored),
        "report_json": str(report_json),
        "report_md": str(report_md),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0")))
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--only", action="append", help="Run only this model label. Can be repeated.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.shards < 1:
        raise ValueError("--shards must be >= 1")

    expected_turns = turn_count(args.dataset)
    manifest = {
        "dataset": str(args.dataset),
        "dataset_turns": expected_turns,
        "repeats": args.repeats,
        "shards": args.shards,
        "temperature": args.temperature,
        "run_tag": args.run_tag,
        "runs": [],
    }
    for model in load_models(args.models, args.only):
        for repeat in range(1, args.repeats + 1):
            manifest["runs"].append(
                run_model_repeat(
                    model,
                    args.dataset,
                    repeat,
                    args.shards,
                    expected_turns,
                    args.temperature,
                    args.run_tag,
                )
            )
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    aggregate_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_dynamic_dialogue_results.py"),
        "--dataset",
        str(args.dataset),
        "--manifest",
        str(args.manifest),
    ]
    if args.run_tag:
        aggregate_cmd.extend(
            [
                "--report-json",
                str(ROOT / "reports" / f"closurebench_dynamic_dialogue_{tag_fragment(args.run_tag)}summary.json"),
                "--report-md",
                str(ROOT / "reports" / f"closurebench_dynamic_dialogue_{tag_fragment(args.run_tag)}summary.md"),
            ]
        )
    subprocess.run(aggregate_cmd, cwd=ROOT, check=True)
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
