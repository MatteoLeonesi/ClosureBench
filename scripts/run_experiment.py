#!/usr/bin/env python3
"""Run closure-contract benchmark prompts against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
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
DEFAULT_DATASET = ROOT / "data" / "closurebench_base.jsonl"
DEFAULT_RESULTS = ROOT / "results" / "closurebench_deepseek_v4_flash_outputs.jsonl"
DEFAULT_SCORED = ROOT / "results" / "closurebench_deepseek_v4_flash_scored.jsonl"


SYSTEM_PROMPT = (
    "You are a careful symbolic reasoner. Use only the provided knowledge base and semantic contract. "
    "Do not use outside knowledge. Return exactly one compact JSON object with keys answer and rationale. "
    "The answer value must be one of true, false, unknown."
)


ANSWER_RE = re.compile(r"\b(true|false|unknown)\b", re.IGNORECASE)
ANSWER_FIELD_RE = re.compile(
    r'"answer"\s*:\s*(?:"(?P<quoted>true|false|unknown)"|(?P<bare>true|false|unknown))',
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_answer(text: str) -> tuple[str | None, bool]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.removeprefix("json").strip()
    try:
        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "")).strip().lower()
        if answer in {"true", "false", "unknown"}:
            return answer, True
    except json.JSONDecodeError:
        pass
    field_match = ANSWER_FIELD_RE.search(raw)
    if field_match:
        answer = (field_match.group("quoted") or field_match.group("bare")).lower()
        return answer, False
    match = ANSWER_RE.search(raw)
    if match:
        return match.group(1).lower(), False
    return None, False


def call_chat_completion(
    base_url: str,
    chat_url: str | None,
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
    if chat_url:
        url = chat_url
    else:
        base = base_url.rstrip("/")
        url = base + "/chat/completions" if base.endswith("/openai") else base + "/v1/chat/completions"
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
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    context = ssl._create_unverified_context() if insecure_skip_verify else None
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def request_with_retry(
    base_url: str,
    chat_url: str | None,
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
        try:
            return call_chat_completion(
                base_url,
                chat_url,
                api_key,
                model,
                prompt,
                temperature,
                timeout,
                max_tokens,
                insecure_skip_verify,
                thinking,
                reasoning_effort,
                response_format,
            ), None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        sleep_s = min(2 ** attempt, 30)
        time.sleep(sleep_s)
    return None, last_error


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("error") is None:
                    done.add(row["id"])
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--chat-url", default=os.environ.get("CHAT_COMPLETIONS_URL"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--thinking",
        choices=("default", "enabled", "disabled"),
        default=os.environ.get("DEEPSEEK_THINKING", "disabled"),
        help="DeepSeek V4 thinking mode. Disabled by default for final-answer JSON evaluation.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=os.environ.get("REASONING_EFFORT"),
        help="Optional OpenRouter/OpenAI-compatible reasoning effort parameter for models that support it.",
    )
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        default=os.environ.get("NO_RESPONSE_FORMAT") == "1",
        help="Do not send response_format=json_object for providers that do not support JSON mode.",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        default=os.environ.get("DEEPSEEK_INSECURE_SKIP_VERIFY") == "1",
        help="Skip TLS certificate verification when the local Python CA store is broken.",
    )
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

    for idx, item in enumerate(remaining, start=1):
        started = time.time()
        response, error = request_with_retry(
            args.base_url,
            args.chat_url,
            api_key,
            args.model,
            item["prompt"],
            args.temperature,
            args.timeout,
            args.max_tokens,
            args.retries,
            args.insecure_skip_verify,
            args.thinking,
            args.reasoning_effort,
            not args.no_response_format,
        )
        completed_at = datetime.now(timezone.utc).isoformat()

        if error:
            output_text = ""
            parsed_answer = None
            json_valid = False
        else:
            content = response["choices"][0]["message"].get("content")
            if isinstance(content, str):
                output_text = content
            elif content is None:
                output_text = ""
            else:
                output_text = json.dumps(content, ensure_ascii=False)
            parsed_answer, json_valid = parse_answer(output_text)

        raw_row = {
            "id": item["id"],
            "model": args.model,
            "temperature": args.temperature,
            "completed_at": completed_at,
            "latency_s": round(time.time() - started, 3),
            "response": response,
            "error": error,
        }
        scored_row = {
            "id": item["id"],
            "base_id": item["base_id"],
            "domain": item["domain"],
            "family": item["family"],
            "semantics": item["semantics"],
            "gold_answer": item["gold_answer"],
            "model_answer": parsed_answer,
            "correct": parsed_answer == item["gold_answer"],
            "json_valid": json_valid,
            "output_text": output_text,
            "error": error,
            "model": args.model,
            "temperature": args.temperature,
            "latency_s": raw_row["latency_s"],
        }
        append_jsonl(args.results, raw_row)
        append_jsonl(args.scored, scored_row)

        if idx % 25 == 0 or idx == len(remaining):
            print(f"Completed {idx}/{len(remaining)}", flush=True)
        time.sleep(args.sleep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
