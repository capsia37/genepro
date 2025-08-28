#!/usr/bin/env python3
"""Compute total time spent in specific source lines from a .speedscope.json sampled profile.

Usage: python scripts/speedscope_totals.py /path/to/profile.speedscope.json

The script sums inclusive time (a sample attributes its duration to every frame on the stack)
across all sampled profiles in the file.
"""
import json
import argparse
import os
import sys


TARGETS = [
    ("genepro/cl_tree_runner.py", 63),
    ("genepro/cl_tree_runner.py", 76),
    ("genepro/cl_tree_runner.py", 78),
    ("genepro/scikit.py", 141),
]

# Human-readable labels for the tracked locations
LABELS = {
    ("genepro/cl_tree_runner.py", 63): "OpenCL build",
    ("genepro/cl_tree_runner.py", 76): "OpenCL run",
    ("genepro/cl_tree_runner.py", 78): "OpenCL transfer",
    ("genepro/scikit.py", 141): "Fitness calculation",
}


def matches_frame(frame, target_path, target_line):
    """Return True if the frame corresponds to target_path:target_line.

    We compare by checking if the frame has a 'file' field that endswith the target path.
    Falls back to checking the frame 'name' if file is empty.
    """
    fpath = frame.get("file", "") or ""
    if fpath:
        if os.path.normpath(fpath).endswith(os.path.normpath(target_path)) and frame.get("line") == target_line:
            return True
    # Fallback: sometimes file is empty and name contains the module/file info
    name = frame.get("name", "")
    if name and target_path in name and frame.get("line") == target_line:
        return True
    return False


def compute_totals(data, targets=TARGETS):
    shared = data.get("shared", {})
    frames = shared.get("frames", [])

    # collect all sampled profiles
    all_sampled = [p for p in data.get("profiles", []) if p.get("type") == "sampled"]

    results = {}

    for idx, profile in enumerate(all_sampled):
        # profile identifier (name or index)
        pname = profile.get("name") or f"profile_{idx}"

        # prepare mapping from target to total seconds for this profile
        totals = {t: 0.0 for t in targets}

        samples = profile.get("samples") or []
        if not samples:
            results[pname] = totals
            continue

        # determine per-sample duration list
        if "weights" in profile and profile.get("weights"):
            weights = profile.get("weights")
            if len(weights) != len(samples):
                weights = None
        else:
            weights = None

        if weights is None:
            start = profile.get("startValue")
            end = profile.get("endValue")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and len(samples) > 0:
                per_sample = (end - start) / len(samples)
                weights = [per_sample] * len(samples)
            else:
                weights = [1.0] * len(samples)

        # for each sample, attribute its weight to each frame index present in the sample
        for i, sample in enumerate(samples):
            dur = weights[i] if i < len(weights) else 0.0
            for fidx in sample:
                if not (0 <= fidx < len(frames)):
                    continue
                frame = frames[fidx]
                for target in targets:
                    if matches_frame(frame, target[0], target[1]):
                        totals[target] += dur

        results[pname] = totals

    return results


def main():
    p = argparse.ArgumentParser(description="Compute total time for specific lines from a .speedscope.json sampled profile")
    p.add_argument("file", help="Path to .speedscope.json file")
    args = p.parse_args()

    path = args.file
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    results = compute_totals(data)

    # results is a mapping from profile name -> { (path,line): seconds }
    TOL = 1e-12
    printed = []
    # accumulator for averages
    acc = {t: 0.0 for t in TARGETS}

    for pname, totals in results.items():
        # skip profiles where none of the targets were found (all totals effectively zero)
        if not any(seconds > TOL for seconds in totals.values()):
            continue
        printed.append((pname, totals))
        # accumulate totals for averaging
        for t, s in totals.items():
            acc[t] += s

    # print per-profile results
    for pname, totals in printed:
        print(f"Profile: {pname}")
        for target, seconds in totals.items():
            label = LABELS.get(target)
            if label:
                print(f"  {label} ({target[0]}:{target[1]}) -> {seconds:.2f} seconds")
            else:
                print(f"  {target[0]}:{target[1]} -> {seconds:.2f} seconds")
        print()

    # print averages across printed profiles
    n = len(printed)
    if n > 0:
        print("Averages:")
        for target in TARGETS:
            avg = acc[target] / n
            label = LABELS.get(target)
            if label:
                print(f"  {label} ({target[0]}:{target[1]}) -> {avg:.2f} seconds (n={n})")
            else:
                print(f"  {target[0]}:{target[1]} -> {avg:.2f} seconds (n={n})")


if __name__ == "__main__":
    main()
