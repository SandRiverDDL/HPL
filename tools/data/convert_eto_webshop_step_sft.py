#!/usr/bin/env python3
"""将 ETO WebShop trajectory 转成 GiGPO/SPEAR 风格 step-level SFT 数据。"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer


DEFAULT_INPUT = (
    "data/hf_cache/datasets--agent-eto--eto-sft-trajectory/snapshots/"
    "a8f745ef799a5201d2288b9c2a47e40a0ff70a4c/data/webshop_sft.json"
)
DEFAULT_TOKENIZER = (
    "/home/fengshuwen/data/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/"
    "snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DEFAULT_OUTPUT_DIR = "data/processed/gigpo/webshop_eto"

TAG_FORMAT_INSTRUCTION = """Your response must use this format:
<think>your reasoning</think>
<action>search[keywords] or click[value]</action>"""

ACTION_PATTERN = re.compile(r"Thought:\s*(.*?)\s*Action:\s*(.*)\s*$", re.DOTALL)


@dataclass
class StepRecord:
    traj_id: str
    row_index: int
    step_idx: int
    reward: float
    source: str
    messages: list[dict[str, str]]
    raw_action: str
    token_length: int


def load_eto_webshop(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入文件顶层必须是 list: {path}")
    return data


def deduplicate_trajectories(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    seen: OrderedDict[str, dict[str, Any]] = OrderedDict()
    duplicates: list[str] = []
    for row in rows:
        traj_id = str(row.get("id"))
        if traj_id in seen:
            duplicates.append(traj_id)
            continue
        seen[traj_id] = row
    return list(seen.values()), duplicates


def filter_trajectories(rows: list[dict[str, Any]], min_reward: float) -> list[dict[str, Any]]:
    return [row for row in rows if float(row.get("reward", 0.0)) >= min_reward]


def parse_thought_action(text: str) -> tuple[str, str] | None:
    match = ACTION_PATTERN.search(text.strip())
    if match is None:
        return None
    thought = match.group(1).strip()
    action = match.group(2).strip()
    if not action.startswith(("search[", "click[")):
        return None
    return thought, action


def format_assistant_output(thought: str, action: str) -> str:
    return f"<think>{thought}</think>\n<action>{action}</action>"


def rewrite_instruction(instruction: str) -> str:
    marker = "Your response should use the following format:"
    if marker in instruction:
        prefix = instruction.split(marker, 1)[0].rstrip()
        return f"{prefix}\n\n{TAG_FORMAT_INSTRUCTION}"
    return f"{instruction.rstrip()}\n\n{TAG_FORMAT_INSTRUCTION}"


def clean_observation(observation: str, clean_policy: str) -> str:
    if clean_policy != "none":
        raise ValueError(f"暂不支持 clean_policy={clean_policy!r}，当前只支持 none")
    return observation


def build_step_messages(
    conversations: list[dict[str, str]],
    step_idx: int,
    *,
    clean_policy: str,
) -> tuple[list[dict[str, str]], str] | None:
    instruction = rewrite_instruction(conversations[0]["value"])
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": "OK"},
    ]

    for history_idx in range(step_idx):
        obs = clean_observation(conversations[2 + 2 * history_idx]["value"], clean_policy)
        parsed = parse_thought_action(conversations[3 + 2 * history_idx]["value"])
        if parsed is None:
            return None
        thought, action = parsed
        messages.append({"role": "user", "content": obs})
        messages.append({"role": "assistant", "content": format_assistant_output(thought, action)})

    cur_obs = clean_observation(conversations[2 + 2 * step_idx]["value"], clean_policy)
    parsed = parse_thought_action(conversations[3 + 2 * step_idx]["value"])
    if parsed is None:
        return None

    thought, action = parsed
    messages.append({"role": "user", "content": cur_obs})
    messages.append({"role": "assistant", "content": format_assistant_output(thought, action)})
    return messages, action


def count_tokens(messages: list[dict[str, str]], tokenizer: Any) -> int:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct / 100)
    return ordered[idx]


def convert_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    clean_policy: str,
    max_trajectories: int | None,
) -> tuple[list[StepRecord], dict[str, Any]]:
    records: list[StepRecord] = []
    parse_failed = 0
    length_filtered = 0
    raw_steps = 0
    max_seen: dict[str, Any] | None = None

    selected_rows = rows[:max_trajectories] if max_trajectories is not None else rows
    for row_index, row in enumerate(selected_rows):
        conversations = row.get("conversations", [])
        if len(conversations) < 4:
            parse_failed += 1
            continue

        step_count = (len(conversations) - 2) // 2
        for step_idx in range(step_count):
            raw_steps += 1
            built = build_step_messages(conversations, step_idx, clean_policy=clean_policy)
            if built is None:
                parse_failed += 1
                continue

            messages, raw_action = built
            token_length = count_tokens(messages, tokenizer)
            if max_seen is None or token_length > max_seen["token_length"]:
                max_seen = {
                    "token_length": token_length,
                    "traj_id": str(row.get("id")),
                    "step_idx": step_idx,
                    "reward": float(row.get("reward", 0.0)),
                }

            if token_length > max_length:
                length_filtered += 1
                continue

            records.append(
                StepRecord(
                    traj_id=str(row.get("id")),
                    row_index=row_index,
                    step_idx=step_idx,
                    reward=float(row.get("reward", 0.0)),
                    source=str(row.get("source", "")),
                    messages=messages,
                    raw_action=raw_action,
                    token_length=token_length,
                )
            )

    token_lengths = [record.token_length for record in records]
    stats = {
        "selected_trajectories": len(selected_rows),
        "raw_steps": raw_steps,
        "kept_steps": len(records),
        "parse_failed_steps": parse_failed,
        "length_filtered_steps": length_filtered,
        "max_length": max_length,
        "clean_policy": clean_policy,
        "token_length": {
            "max": max(token_lengths) if token_lengths else None,
            "mean": round(mean(token_lengths), 2) if token_lengths else None,
            "p50": percentile(token_lengths, 50),
            "p90": percentile(token_lengths, 90),
            "p95": percentile(token_lengths, 95),
            "p99": percentile(token_lengths, 99),
            "max_seen_before_filter": max_seen,
        },
    }
    return records, stats


def write_jsonl(records: list[StepRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            item = asdict(record)
            item["id"] = f"{record.traj_id}_step_{record.step_idx}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", type=str, default="webshop_step_sft_2048_raw.jsonl")
    parser.add_argument("--report-name", type=str, default="conversion_report.json")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-reward", type=float, default=0.7)
    parser.add_argument("--clean-policy", choices=["none"], default="none")
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--no-dedup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_eto_webshop(args.input)
    duplicate_ids: list[str] = []
    if args.no_dedup:
        deduped_rows = rows
    else:
        deduped_rows, duplicate_ids = deduplicate_trajectories(rows)

    filtered_rows = filter_trajectories(deduped_rows, args.min_reward)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    records, conversion_stats = convert_rows(
        filtered_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        clean_policy=args.clean_policy,
        max_trajectories=args.max_trajectories,
    )

    output_path = args.output_dir / args.output_name
    report_path = args.output_dir / args.report_name
    write_jsonl(records, output_path)

    report = {
        "input_path": str(args.input),
        "tokenizer": args.tokenizer,
        "output_path": str(output_path),
        "input_trajectories": len(rows),
        "deduplicated": not args.no_dedup,
        "deduped_trajectories": len(deduped_rows),
        "duplicate_ids": sorted(set(duplicate_ids)),
        "min_reward": args.min_reward,
        "reward_filtered_trajectories": len(filtered_rows),
        **conversion_stats,
    }
    write_report(report, report_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
