import argparse
import gzip
import json
from pathlib import Path


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def summarize(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    gather = [
        (event["ts"], event["ts"] + event["dur"])
        for event in kernels
        if "nccl" in event["name"].lower() and "allgather" in event["name"].lower()
    ]
    compute = [
        (event["ts"], event["ts"] + event["dur"])
        for event in kernels
        if "flash" in event["name"].lower() and "nccl" not in event["name"].lower()
    ]
    assert gather and compute, f"Missing AllGather or FlashAttention GPU kernels: {path}"
    gather_union = merge_intervals(gather)
    compute_union = merge_intervals(compute)
    intersection = merge_intervals([
        (max(a, c), min(b, d))
        for a, b in gather_union
        for c, d in compute_union
        if max(a, c) < min(b, d)
    ])
    gather_us, compute_us, overlap_us = map(duration, (gather_union, compute_union, intersection))
    return {
        "trace": path.name,
        "allgather_kernels": len(gather),
        "allgather_us": round(gather_us, 2),
        "attention_us": round(compute_us, 2),
        "overlap_us": round(overlap_us, 2),
        "overlap_percent": round(100 * overlap_us / gather_us, 2),
        "exposed_allgather_us": round(gather_us - overlap_us, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 GPU kernel 时间区间的交集，不代表有效传输带宽")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("*.json.gz"))
    assert paths, args.directory
    for path in paths:
        print(json.dumps(summarize(path)))


if __name__ == "__main__":
    main()
