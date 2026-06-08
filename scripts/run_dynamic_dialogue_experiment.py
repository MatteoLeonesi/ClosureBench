#!/usr/bin/env python3
"""Run ClosureBench dynamic-state dialogues against an OpenAI-compatible API."""

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
DEFAULT_DATASET = ROOT / "data" / "closurebench_dynamic_dialogue.jsonl"
DEFAULT_RESULTS = ROOT / "results" / "closurebench_dynamic_dialogue_outputs.jsonl"
DEFAULT_SCORED = ROOT / "results" / "closurebench_dynamic_dialogue_scored.jsonl"

SYSTEM_PROMPT = (
    "You are a careful symbolic reasoner in a dynamic-state dialogue. "
    "Maintain the knowledge base across turns, apply only the current completeness-state update, "
    "and recompute the target answer from the updated state. "
    "Return exactly one compact JSON object."
)

ANSWER_RE = re.compile(r"\b(true|false|unknown)\b", re.IGNORECASE)
ANSWER_FIELD_RE = re.compile(
    r'"answer"\s*:\s*(?:"(?P<quoted>true|false|unknown)"|(?P<bare>true|false|unknown))',
    re.IGNORECASE,
)
CHANGED_FIELD_RE = re.compile(
    r'"changed_from_previous(?:ly)?"\s*:\s*(?:"(?P<quoted>true|false|null)"|(?P<bare>true|false|null))',
    re.IGNORECASE,
)
APPLIED_UPDATE_FIELD_RE = re.compile(
    r'"applied_update"\s*:\s*"?(?P<value>[A-Za-z0-9 _-]+)"?',
    re.IGNORECASE,
)
VALID_UPDATES = {"open", "global_complete", "local_complete"}
UPDATE_ALIASES = {
    "open": "open",
    "set_complete_predicates": "open",
    "set_complete_predicates_none": "open",
    "none_complete": "open",
    "global_complete": "global_complete",
    "add_complete_predicates": "global_complete",
    "all_complete": "global_complete",
    "complete_all": "global_complete",
    "local_complete": "local_complete",
    "replace_complete_predicates": "local_complete",
    "local_closed": "local_complete",
    "local": "local_complete",
}


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


def parse_answer(text: str) -> tuple[str | None, bool]:
    raw = normalize_json_text(text)
    try:
        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "")).strip().lower()
        if answer in {"true", "false", "unknown"}:
            return answer, True
    except json.JSONDecodeError:
        pass
    field = ANSWER_FIELD_RE.search(raw)
    if field:
        return (field.group("quoted") or field.group("bare")).lower(), False
    fallback = ANSWER_RE.search(raw)
    if fallback:
        return fallback.group(1).lower(), False
    return None, False


def parse_changed(text: str) -> tuple[bool | None, bool, bool]:
    raw = normalize_json_text(text)
    strict_key = False
    try:
        parsed = json.loads(raw)
        if "changed_from_previous" in parsed:
            value = parsed.get("changed_from_previous")
            strict_key = True
        else:
            value = parsed.get("changed_from_previously")
        if value is None:
            return None, True, strict_key
        if isinstance(value, bool):
            return value, True, strict_key
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "null"}:
            normalized = value.strip().lower()
            return None if normalized == "null" else normalized == "true", True, strict_key
    except json.JSONDecodeError:
        pass
    field = CHANGED_FIELD_RE.search(raw)
    if field:
        value = (field.group("quoted") or field.group("bare")).lower()
        strict_key = '"changed_from_previous"' in field.group(0)
        return None if value == "null" else value == "true", False, strict_key
    return None, False, False


def normalize_update(value: object) -> str | None:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return UPDATE_ALIASES.get(normalized)


def parse_applied_update(text: str) -> tuple[str | None, bool]:
    raw = normalize_json_text(text)
    try:
        parsed = json.loads(raw)
        if "applied_update" in parsed:
            value = normalize_update(parsed.get("applied_update"))
            if value in VALID_UPDATES:
                return value, True
    except json.JSONDecodeError:
        pass
    field = APPLIED_UPDATE_FIELD_RE.search(raw)
    if field:
        value = normalize_update(field.group("value"))
        if value in VALID_UPDATES:
            return value, False
    return None, False


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
    messages: list[dict],
    temperature: float,
    timeout: int,
    max_tokens: int,
    insecure_skip_verify: bool,
    thinking: str,
    response_format: bool,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}
    if thinking != "default":
        payload["thinking"] = {"type": thinking}
    request = urllib.request.Request(
        chat_url(base_url, explicit_chat_url),
        data=json.dumps(payload).encode("utf-8"),
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
    *,
    base_url: str,
    explicit_chat_url: str | None,
    api_key: str,
    model: str,
    messages: list[dict],
    temperature: float,
    timeout: int,
    max_tokens: int,
    retries: int,
    insecure_skip_verify: bool,
    thinking: str,
    response_format: bool,
) -> tuple[dict | None, str | None]:
    last_error = None
    for attempt in range(retries + 1):
        try:
            return call_chat(
                base_url=base_url,
                explicit_chat_url=explicit_chat_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
                insecure_skip_verify=insecure_skip_verify,
                thinking=thinking,
                response_format=response_format,
            ), None
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body[:500]}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2**attempt, 30))
    return None, last_error


def load_done_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("error") is None:
                    done[row["id"]] = row
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


def turn_id(dialogue_id: str, turn_index: int) -> str:
    return f"{dialogue_id}__turn_{turn_index}"


def filter_dialogues(dialogues: list[dict], num_shards: int, shard_index: int | None) -> list[dict]:
    if shard_index is None:
        return dialogues
    return [dialogue for index, dialogue in enumerate(dialogues) if index % num_shards == shard_index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--limit-dialogues", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--chat-url", default=os.environ.get("CHAT_COMPLETIONS_URL"))
    parser.add_argument("--model", default=os.environ.get("MODEL", os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0")))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument(
        "--thinking",
        choices=("default", "enabled", "disabled"),
        default=os.environ.get("DEEPSEEK_THINKING", "disabled"),
    )
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        default=os.environ.get("NO_RESPONSE_FORMAT") == "1",
    )
    parser.add_argument(
        "--insecure-skip-verify",
        action="store_true",
        default=os.environ.get("DEEPSEEK_INSECURE_SKIP_VERIFY") == "1",
    )
    args = parser.parse_args()

    if args.shard_index is not None and not (0 <= args.shard_index < args.num_shards):
        print("--shard-index must be in [0, --num-shards)", file=sys.stderr)
        return 2

    api_key = os.environ.get("API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("API_KEY or DEEPSEEK_API_KEY is not set.", file=sys.stderr)
        return 2

    dialogues = read_jsonl(args.dataset)
    dialogues = filter_dialogues(dialogues, args.num_shards, args.shard_index)
    if args.limit_dialogues is not None:
        dialogues = dialogues[: args.limit_dialogues]

    done_rows = load_done_rows(args.scored)
    done = set(done_rows)
    total_turns = sum(len(dialogue["turns"]) for dialogue in dialogues)
    remaining = sum(
        1
        for dialogue in dialogues
        for turn in dialogue["turns"]
        if turn_id(dialogue["id"], turn["turn_index"]) not in done
    )
    print(f"Dialogues: {len(dialogues)}; turns: {total_turns}; already scored: {len(done)}; remaining: {remaining}", flush=True)

    completed = 0
    for dialogue in dialogues:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        skip_dialogue = False
        for turn in dialogue["turns"]:
            row_id = turn_id(dialogue["id"], turn["turn_index"])
            if row_id in done:
                messages.append({"role": "user", "content": turn["prompt"]})
                cached = done_rows[row_id].get("output_text") or json.dumps(
                    {
                        "answer": done_rows[row_id].get("model_answer"),
                        "changed_from_previous": done_rows[row_id].get("changed_from_previous"),
                        "applied_update": turn["update_operation"],
                        "rationale": "Cached turn from resumed run.",
                    },
                    ensure_ascii=False,
                )
                messages.append({"role": "assistant", "content": cached})
                continue
            if skip_dialogue:
                continue

            messages.append({"role": "user", "content": turn["prompt"]})
            started = time.time()
            response, error = request_with_retry(
                base_url=args.base_url,
                explicit_chat_url=args.chat_url,
                api_key=api_key,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                retries=args.retries,
                insecure_skip_verify=args.insecure_skip_verify,
                thinking=args.thinking,
                response_format=not args.no_response_format,
            )
            output = response_text(response, error)
            model_answer, answer_json_valid = parse_answer(output)
            changed, changed_json_valid, strict_changed_key = parse_changed(output)
            applied_update, applied_update_json_valid = parse_applied_update(output)
            expected_changed = turn["expected_changed_from_previous"]
            expected_applied_update = turn.get("expected_applied_update")
            answer_correct = model_answer == turn["gold_answer"]
            changed_correct = changed == expected_changed
            applied_update_correct = applied_update == expected_applied_update
            latency_s = round(time.time() - started, 3)

            raw_row = {
                "id": row_id,
                "dialogue_id": dialogue["id"],
                "turn_index": turn["turn_index"],
                "model": args.model,
                "temperature": args.temperature,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "latency_s": latency_s,
                "response": response,
                "error": error,
            }
            scored_row = {
                "id": row_id,
                "dialogue_id": dialogue["id"],
                "base_id": dialogue["base_id"],
                "domain": dialogue["domain"],
                "family": dialogue["family"],
                "dialogue_type": dialogue["dialogue_type"],
                "gold_pattern": dialogue["gold_pattern"],
                "turn_index": turn["turn_index"],
                "update_operation": turn["update_operation"],
                "complete_predicates_after_update": turn["complete_predicates_after_update"],
                "gold_answer": turn["gold_answer"],
                "expected_changed_from_previous": expected_changed,
                "expected_applied_update": expected_applied_update,
                "model_answer": model_answer,
                "changed_from_previous": changed,
                "applied_update": applied_update,
                "answer_correct": answer_correct,
                "changed_correct": changed_correct,
                "applied_update_correct": applied_update_correct,
                "strict_changed_key": strict_changed_key,
                "correct": answer_correct and changed_correct and applied_update_correct,
                "json_valid": answer_json_valid and changed_json_valid and applied_update_json_valid and strict_changed_key,
                "output_text": output,
                "error": error,
                "model": args.model,
                "temperature": args.temperature,
                "latency_s": latency_s,
            }
            append_jsonl(args.results, raw_row)
            append_jsonl(args.scored, scored_row)

            messages.append({"role": "assistant", "content": output or json.dumps(scored_row, ensure_ascii=False)})
            completed += 1
            if completed % 25 == 0 or completed == remaining:
                print(f"Completed {completed}/{remaining} remaining turns", flush=True)
            time.sleep(args.sleep)
            if error:
                skip_dialogue = True

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
