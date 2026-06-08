#!/usr/bin/env python3
"""Run repeated ClosureBench evaluations for all configured models."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_MODELS = ROOT / "configs" / "closurebench_replicate_models.example.json"
DEFAULT_MANIFEST = ROOT / "reports" / "closurebench_replicate_manifest.json"
SHARD_DIR = ROOT / "tmp" / "closurebench_replicate_shards"
SHARD_RESULTS_DIR = ROOT / "results" / "replicate_shards"
FINAL_RESULTS_DIR = ROOT / "results" / "replicates"
FINAL_REPORTS_DIR = ROOT / "reports" / "replicates"


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
    with path.open("r", encoding="utf-8") as fh:
        models = json.load(fh)
    if only:
        wanted = set(only)
        models = [model for model in models if model["label"] in wanted]
        missing = wanted.difference(model["label"] for model in models)
        if missing:
            raise ValueError(f"Missing labels in config: {', '.join(sorted(missing))}")
    return models


def split_shards(dataset: Path, label: str, repeat: int, n_shards: int, run_tag: str) -> list[Path]:
    rows = read_jsonl(dataset)
    shard_paths = []
    safe_label = sanitize_label(label)
    safe_tag = sanitize_label(run_tag)
    for shard_index in range(n_shards):
        shard_rows = [row for idx, row in enumerate(rows) if idx % n_shards == shard_index]
        shard_path = SHARD_DIR / safe_tag / safe_label / f"repeat_{repeat}" / f"shard_{shard_index}.jsonl"
        write_jsonl(shard_path, shard_rows)
        shard_paths.append(shard_path)
    return shard_paths


def item_order(dataset: Path) -> list[str]:
    return [row["id"] for row in read_jsonl(dataset)]


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
    ordered_rows = [rows_by_id[row_id] for row_id in item_order(dataset) if row_id in rows_by_id]
    write_jsonl(output_path, ordered_rows)


def build_runner_command(model: dict, shard_path: Path, results_path: Path, scored_path: Path, temperature: float) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_experiment.py"),
        "--dataset",
        str(shard_path),
        "--results",
        str(results_path),
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
    if model.get("reasoning_effort"):
        cmd.extend(["--reasoning-effort", model["reasoning_effort"]])
    if model.get("response_format") is False:
        cmd.append("--no-response-format")
    if model.get("insecure_skip_verify"):
        cmd.append("--insecure-skip-verify")
    return cmd


def final_paths(label: str, repeat: int, run_tag: str) -> tuple[Path, Path]:
    safe_label = sanitize_label(label)
    stem = f"closurebench_{tag_fragment(run_tag)}{safe_label}_repeat_{repeat}"
    return FINAL_RESULTS_DIR / f"{stem}_outputs.jsonl", FINAL_RESULTS_DIR / f"{stem}_scored.jsonl"


def is_complete(scored_path: Path, dataset_size: int) -> bool:
    if not scored_path.exists():
        return False
    successful_ids = {
        row["id"]
        for row in read_jsonl(scored_path)
        if row.get("error") is None
    }
    return len(successful_ids) == dataset_size


def run_model_repeat(
    model: dict,
    dataset: Path,
    repeat: int,
    n_shards: int,
    dataset_size: int,
    temperature: float,
    run_tag: str,
) -> dict:
    label = model["label"]
    env_key = model["api_key_env"]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(f"{env_key} is not set for model label {label}")

    final_outputs, final_scored = final_paths(label, repeat, run_tag)
    if is_complete(final_scored, dataset_size):
        print(f"[{label} r{repeat}] already complete", flush=True)
    else:
        shard_paths = split_shards(dataset, label, repeat, n_shards, run_tag)
        processes = []
        safe_label = sanitize_label(label)
        for shard_index, shard_path in enumerate(shard_paths):
            shard_stem = f"{tag_fragment(run_tag)}{safe_label}_repeat_{repeat}_shard_{shard_index}"
            shard_outputs = SHARD_RESULTS_DIR / f"{shard_stem}_outputs.jsonl"
            shard_scored = SHARD_RESULTS_DIR / f"{shard_stem}_scored.jsonl"
            cmd = build_runner_command(model, shard_path, shard_outputs, shard_scored, temperature)
            env = os.environ.copy()
            env["API_KEY"] = api_key
            print(f"[{label} r{repeat}] starting shard {shard_index + 1}/{n_shards}", flush=True)
            processes.append((subprocess.Popen(cmd, cwd=ROOT, env=env), shard_outputs, shard_scored))

        failures = []
        for process, _, shard_scored in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((shard_scored, return_code))
        if failures:
            details = ", ".join(f"{path}: exit {code}" for path, code in failures)
            raise RuntimeError(f"{label} repeat {repeat} failed: {details}")

        merge_shards(dataset, [item[1] for item in processes], final_outputs)
        merge_shards(dataset, [item[2] for item in processes], final_scored)

    report_json = FINAL_REPORTS_DIR / f"{final_scored.stem}_report.json"
    report_md = FINAL_REPORTS_DIR / f"{final_scored.stem}_report.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_results.py"),
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

    dataset_size = len(read_jsonl(args.dataset))
    models = load_models(args.models, args.only)
    manifest = {
        "dataset": str(args.dataset),
        "dataset_size": dataset_size,
        "repeats": args.repeats,
        "shards": args.shards,
        "temperature": args.temperature,
        "run_tag": args.run_tag,
        "runs": [],
    }

    for model in models:
        for repeat in range(1, args.repeats + 1):
            manifest["runs"].append(
                run_model_repeat(
                    model,
                    args.dataset,
                    repeat,
                    args.shards,
                    dataset_size,
                    args.temperature,
                    args.run_tag,
                )
            )
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            with args.manifest.open("w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, ensure_ascii=False)

    aggregate_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "aggregate_replicate_results.py"),
        "--dataset",
        str(args.dataset),
        "--manifest",
        str(args.manifest),
    ]
    if args.run_tag:
        aggregate_cmd.extend(
            [
                "--report-json",
                str(ROOT / "reports" / f"closurebench_{tag_fragment(args.run_tag)}replicate_summary.json"),
                "--report-md",
                str(ROOT / "reports" / f"closurebench_{tag_fragment(args.run_tag)}replicate_summary.md"),
            ]
        )
    subprocess.run(aggregate_cmd, cwd=ROOT, check=True)
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
