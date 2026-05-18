#!/usr/bin/env python3
"""
GPU 占卡脚本：占用指定 GPU 的显存与可选算力，防止临时空闲卡被抢占。

用法：
    .venv/bin/python scripts/train_model.py
    .venv/bin/python scripts/train_model.py --gpus 0,1
    .venv/bin/python scripts/train_model.py --mem 0.8
    .venv/bin/python scripts/train_model.py --compute none
    .venv/bin/python scripts/train_model.py --compute heavy

按 Ctrl+C 即可释放。
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="要占用的 GPU id，逗号分隔。默认占全部可见 GPU。",
    )
    parser.add_argument(
        "--mem",
        type=float,
        default=0.6,
        help="每张卡目标占用显存比例，范围 0 到 1，默认 0.6。",
    )
    parser.add_argument(
        "--mem-jitter",
        type=float,
        default=0.1,
        help="显存占用波动幅度，例如 0.03 表示在目标比例上下约 3%% 波动，默认 0.1。",
    )
    parser.add_argument(
        "--mem-refresh-interval",
        type=float,
        default=45.0,
        help="重新调整显存占用的间隔秒数，默认 45s。",
    )
    parser.add_argument(
        "--compute",
        choices=["none", "light", "heavy"],
        default="heavy",
        help="算力占用模式：none 只占显存；light 低频随机矩阵乘；heavy 持续较重矩阵乘。",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="light 模式下每轮计算的基础间隔秒数，默认 2s。",
    )
    parser.add_argument(
        "--util-jitter",
        type=float,
        default=0.5,
        help="算力间隔和矩阵大小波动幅度，范围 0 到 1，默认 0.5。",
    )
    return parser.parse_args()


def get_gpu_ids(arg_gpus: str | None) -> list[int]:
    if arg_gpus is not None:
        return [int(x) for x in arg_gpus.split(",") if x.strip()]
    return list(range(torch.cuda.device_count()))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def sample_mem_ratio(target: float, jitter: float) -> float:
    if jitter <= 0:
        return clamp(target, 0.0, 0.98)
    return clamp(random.uniform(target - jitter, target + jitter), 0.0, 0.98)


def allocate_memory(device: int, ratio: float) -> list[torch.Tensor]:
    """在给定 GPU 上占用指定比例的显存，并返回持有显存的 tensor。"""
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    target = int(total * ratio)
    target = min(target, int(free * 0.98))

    elem_size = 4
    numel = target // elem_size
    chunks: list[torch.Tensor] = []
    remaining = numel
    chunk_size = 1 << 28

    while remaining > 0:
        n = min(chunk_size, remaining)
        try:
            tensor = torch.empty(n, dtype=torch.float32, device=device)
            chunks.append(tensor)
            remaining -= n
        except RuntimeError:
            chunk_size //= 2
            if chunk_size < (1 << 20):
                break

    allocated = sum(chunk.numel() * elem_size for chunk in chunks)
    print(
        f"[GPU {device}] 目标比例 {ratio:.3f}，已占用显存: "
        f"{allocated / 1024**3:.2f} GiB / 总 {total / 1024**3:.2f} GiB"
    )
    return chunks


def refresh_memory(gpu_ids: list[int], target_mem: float, mem_jitter: float) -> dict[int, list[torch.Tensor]]:
    """重新采样显存占用比例，让 nvidia-smi 里的显存占用有轻微波动。"""
    held_tensors: dict[int, list[torch.Tensor]] = {}
    torch.cuda.empty_cache()
    for gpu_id in gpu_ids:
        ratio = sample_mem_ratio(target_mem, mem_jitter)
        held_tensors[gpu_id] = allocate_memory(gpu_id, ratio)
    return held_tensors


def setup_signal_handler() -> None:
    def handler(sig: int, frame: object) -> None:
        print("\n收到退出信号，释放显存并退出。")
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def sample_compute_size(mode: str, util_jitter: float) -> int:
    if mode == "heavy":
        base = 4096
        low = int(base * (1.0 - 0.35 * util_jitter))
        high = int(base * (1.0 + 0.35 * util_jitter))
    else:
        base = 1024
        low = int(base * (1.0 - 0.5 * util_jitter))
        high = int(base * (1.0 + 0.8 * util_jitter))
    low = max(256, low)
    high = max(low, high)
    return random.randint(low, high)


def run_compute_once(gpu_id: int, mode: str, util_jitter: float) -> None:
    size = sample_compute_size(mode, util_jitter)
    with torch.cuda.device(gpu_id):
        a = torch.randn(size, size, device=gpu_id)
        b = torch.randn(size, size, device=gpu_id)
        _ = a @ b
        torch.cuda.synchronize(gpu_id)


def sleep_with_jitter(base_interval: float, util_jitter: float) -> None:
    if base_interval <= 0:
        return
    low = max(0.05, base_interval * (1.0 - util_jitter))
    high = max(low, base_interval * (1.0 + util_jitter))
    time.sleep(random.uniform(low, high))


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("未检测到可用的 CUDA 设备。")
        return

    gpu_ids = get_gpu_ids(args.gpus)
    print(
        f"将占用 GPU: {gpu_ids}，显存比例: {args.mem}±{args.mem_jitter}，"
        f"算力模式: {args.compute}，PID: {os.getpid()}"
    )
    print("按 Ctrl+C 释放。")

    setup_signal_handler()
    held_tensors = refresh_memory(gpu_ids, args.mem, args.mem_jitter)
    last_mem_refresh = time.monotonic()

    print("占卡中... 显存和 GPU util 会按配置轻微波动。")
    try:
        while True:
            if time.monotonic() - last_mem_refresh >= args.mem_refresh_interval:
                held_tensors.clear()
                torch.cuda.empty_cache()
                held_tensors = refresh_memory(gpu_ids, args.mem, args.mem_jitter)
                last_mem_refresh = time.monotonic()

            # 保持引用，避免 tensor 被释放。
            _ = held_tensors

            if args.compute == "none":
                time.sleep(min(args.mem_refresh_interval, 60.0))
                continue

            for gpu_id in gpu_ids:
                run_compute_once(gpu_id, args.compute, args.util_jitter)

            if args.compute == "light":
                sleep_with_jitter(args.interval, args.util_jitter)
    except KeyboardInterrupt:
        print("\n已释放。")


if __name__ == "__main__":
    main()
