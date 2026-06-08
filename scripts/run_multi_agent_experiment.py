#!/usr/bin/env python3
"""Run ClosureBench-MA prompts against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "closurebench_multi_agent.jsonl"
DEFAULT_RESULTS = ROOT / "results" / "closurebench_multi_agent_outputs.jsonl"
DEFAULT_SCORED = ROOT / "results" / "closurebench_multi_agent_scored.jsonl"

SYSTEM_PROMPT = (
    "You are a careful coordinator in a multi-agent system. Use only the source-agent reports, rules, "
    "and source-scoped closure contract. Return exactly one compact JSON object."
)

TRUTH_VALUES = {"true", "false", "unknown"}
SOURCE_VALUES = {"agent_a", "agent_b", "both", "neither"}
CLOSURE_HANDLINGS = {"entailed_or_explicit", "explicit_negative", "closed_absence", "open_absence"}

TRUTH_FIELD_RE = re.compile(
    r'"(?:truth_value|answer)"\s*:\s*(?:"(?P<quoted>true|false|unknown)"|(?P<bare>true|false|unknown))',
    re.IGNORECASE,
)
SOURCE_FIELD_RE = re.compile(
    r'"(?:source_used|source|agent)"\s*:\s*"?(?P<source>agent[_ -]?a|agent[_ -]?b|both|neither|none)"?',
    re.IGNORECASE,
)
CLOSURE_FIELD_RE = re.compile(
    r'"(?:closure_handling|closure|handling)"\s*:\s*"?(?P<closure>entailed[_ -]?or[_ -]?explicit|explicit[_ -]?negative|closed[_ -]?absence|open[_ -]?absence)"?',
    re.IGNORECASE,
)
RETRY_IN_RE = re.compile(r"retry in\s+(?P<seconds>[0-9.]+)s", re.IGNORECASE)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_json_text(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    return raw


def normalize_truth(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized if normalized in TRUTH_VALUES else None


def normalize_source(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized in {"none", "no_source", "no_agent"}:
        return "neither"
    return normalized if normalized in SOURCE_VALUES else None


def normalize_closure(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized if normalized in CLOSURE_HANDLINGS else None


def parse_output(text: str) -> dict:
    raw = normalize_json_text(text)
    parsed: dict | None = None
    strict_json = False
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            parsed = loaded
            strict_json = True
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        truth_value = normalize_truth(parsed.get("truth_value", parsed.get("answer")))
        source_used = normalize_source(parsed.get("source_used", parsed.get("source", parsed.get("agent"))))
        closure_handling = normalize_closure(parsed.get("closure_handling", parsed.get("closure", parsed.get("handling"))))
    else:
        truth_match = TRUTH_FIELD_RE.search(raw)
        source_match = SOURCE_FIELD_RE.search(raw)
        closure_match = CLOSURE_FIELD_RE.search(raw)
        truth_value = normalize_truth(truth_match.group("quoted") or truth_match.group("bare") if truth_match else None)
        source_used = normalize_source(source_match.group("source") if source_match else None)
        closure_handling = normalize_closure(closure_match.group("closure") if closure_match else None)

    required = {"truth_value", "source_used", "closure_handling", "rationale"}
    json_valid = bool(
        strict_json
        and parsed is not None
        and required.issubset(parsed)
        and truth_value in TRUTH_VALUES
        and source_used in SOURCE_VALUES
        and closure_handling in CLOSURE_HANDLINGS
    )
    return {
        "truth_value": truth_value,
        "source_used": source_used,
        "closure_handling": closure_handling,
        "json_valid": json_valid,
    }


def source_matches(model_source: str | None, gold_source: str, gold_closure_handling: str) -> bool:
    if model_source == gold_source:
        return True
    return gold_closure_handling == "open_absence" and model_source in {"agent_b", "neither"}


def chat_url(base_url: str, explicit_chat_url: str | None) -> str:
    if explicit_chat_url:
        return explicit_chat_url
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/openai") else base + "/v1/chat/completions"


def call_chat(
    *,
    base_url: str,
    explicit_chat_url: str | None,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    insecure_skip_verify: bool,
    thinking: str,
    reasoning_effort: str | None,
    response_format: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    if thinking != "default":
        payload["thinking"] = {"type": thinking}
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    request = urllib.request.Request(
        chat_url(base_url, explicit_chat_url),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    context = ssl._create_unverified_context() if insecure_skip_verify else None
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def retry_sleep_seconds(exc: urllib.error.HTTPError | None, body: str, attempt: int) -> float:
    if exc is not None:
        retry_after = exc.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after) + 1.0, 120.0)
            except ValueError:
                pass
    match = RETRY_IN_RE.search(body)
    if match:
        return min(float(match.group("seconds")) + 1.0, 120.0)
    return min(2**attempt, 30)


def request_with_retry(
    *,
    base_url: str,
    explicit_chat_url: str | None,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    max_tokens: int,
    retries: int,
    insecure_skip_verify: bool,
    thinking: str,
    reasoning_effort: str | None,
    response_format: bool,
) -> tuple[dict | None, str | None]:
    last_error = None
    for attempt in range(retries + 1):
        sleep_s = min(2**attempt, 30)
        try:
            return call_chat(
                base_url=base_url,
                explicit_chat_url=explicit_chat_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                insecure_skip_verify=insecure_skip_verify,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
            ), None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
            sleep_s = retry_sleep_seconds(exc, body, attempt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(sleep_s)
    return None, last_error


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error") is None:
            done.add(row["id"])
    return done


def response_text(response: dict | None, error: str | None) -> str:
    if error or response is None:
        return ""
    content = response["choices"][0]["message"].get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def wait_for_rate_slot(rate_limit_file: Path | None, min_request_interval: float) -> None:
    if rate_limit_file is None or min_request_interval <= 0:
        return
    rate_limit_file.parent.mkdir(parents=True, exist_ok=True)
    with rate_limit_file.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read().strip()
        try:
            last_request = float(raw) if raw else 0.0
        except ValueError:
            last_request = 0.0
        wait_s = max(0.0, last_request + min_request_interval - time.time())
        if wait_s:
            time.sleep(wait_s)
        fh.seek(0)
        fh.truncate()
        fh.write(str(time.time()))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--chat-url", default=os.environ.get("CHAT_COMPLETIONS_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL", os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--rate-limit-file", type=Path, default=None)
    parser.add_argument("--min-request-interval", type=float, default=0.0)
    parser.add_argument("--thinking", choices=("default", "enabled", "disabled"), default=os.environ.get("THINKING", "disabled"))
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=os.environ.get("REASONING_EFFORT"))
    parser.add_argument("--no-response-format", action="store_true", default=os.environ.get("NO_RESPONSE_FORMAT") == "1")
    parser.add_argument("--insecure-skip-verify", action="store_true", default=os.environ.get("INSECURE_SKIP_VERIFY") == "1")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("API_KEY or DEEPSEEK_API_KEY is not set.", file=sys.stderr)
        return 2

    items = read_jsonl(args.dataset)
    if args.limit is not None:
        items = items[: args.limit]
    done = load_done(args.scored)
    remaining = [item for item in items if item["id"] not in done]
    print(f"Dataset items: {len(items)}; already scored: {len(done)}; remaining: {len(remaining)}", flush=True)

    for index, item in enumerate(remaining, start=1):
        wait_for_rate_slot(args.rate_limit_file, args.min_request_interval)
        started = time.time()
        response, error = request_with_retry(
            base_url=args.base_url,
            explicit_chat_url=args.chat_url,
            api_key=api_key,
            model=args.model,
            prompt=item["prompt"],
            temperature=args.temperature,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            retries=args.retries,
            insecure_skip_verify=args.insecure_skip_verify,
            thinking=args.thinking,
            reasoning_effort=args.reasoning_effort,
            response_format=not args.no_response_format,
        )
        output = response_text(response, error)
        parsed = parse_output(output)
        latency_s = round(time.time() - started, 3)

        truth_correct = parsed["truth_value"] == item["gold_truth_value"]
        source_correct = source_matches(parsed["source_used"], item["gold_source_used"], item["gold_closure_handling"])
        closure_correct = parsed["closure_handling"] == item["gold_closure_handling"]

        raw_row = {
            "id": item["id"],
            "source_id": item["source_id"],
            "model": args.model,
            "temperature": args.temperature,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "latency_s": latency_s,
            "response": response,
            "error": error,
        }
        scored_row = {
            "id": item["id"],
            "source_id": item["source_id"],
            "base_id": item["base_id"],
            "domain": item["domain"],
            "family": item["family"],
            "subset": item["subset"],
            "semantics": item["semantics"],
            "gold_reason_type": item["gold_reason_type"],
            "gold_truth_value": item["gold_truth_value"],
            "model_truth_value": parsed["truth_value"],
            "truth_correct": truth_correct,
            "gold_source_used": item["gold_source_used"],
            "model_source_used": parsed["source_used"],
            "source_correct": source_correct,
            "gold_closure_handling": item["gold_closure_handling"],
            "model_closure_handling": parsed["closure_handling"],
            "closure_correct": closure_correct,
            "correct": truth_correct and source_correct and closure_correct,
            "json_valid": parsed["json_valid"],
            "output_text": output,
            "error": error,
            "model": args.model,
            "temperature": args.temperature,
            "latency_s": latency_s,
        }
        append_jsonl(args.results, raw_row)
        append_jsonl(args.scored, scored_row)

        if index % 25 == 0 or index == len(remaining):
            print(f"Completed {index}/{len(remaining)}", flush=True)
        time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
