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
    ("genepro/node.py", 65),
]

# Human-readable labels for the tracked locations
LABELS = {
    ("genepro/cl_tree_runner.py", 63): "OpenCL build",
    ("genepro/cl_tree_runner.py", 76): "OpenCL run",
    ("genepro/cl_tree_runner.py", 78): "OpenCL transfer",
    ("genepro/scikit.py", 141): "Fitness calculation",
    ("genepro/node.py", 65): "CPU evaluation",
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
    p.add_argument("file", nargs='?', default=None, help="Path to .speedscope.json file")
    p.add_argument("--csv", action="store_true", help="Output a single CSV line containing only the averages (no human-readable text)")
    p.add_argument("--csv-header", action="store_true", help="Print CSV header line and exit (useful to initialize a results file)")
    p.add_argument("-P", "--param", action="append", default=[], help="Additional key=val parameters to include in CSV output. Can be repeated.")
    args = p.parse_args()

    # parse params early so header printing can use them
    params_list = args.param or []
    params = {}
    for pstr in params_list:
        if "=" in pstr:
            k, v = pstr.split("=", 1)
            params[k] = v
        else:
            params[pstr] = ""

    # Build CSV label names for targets in the same order as TARGETS
    target_cols = [LABELS.get(t, f"{t[0]}:{t[1]}") for t in TARGETS]

    # Desired CSV order: size,n_jobs,mode,max_gens,test_acc,<targets...>,elapsed_s
    # Note: elapsed_s is not part of the prefix; it will be appended after target columns if present
    desired_prefix = ["size", "n_jobs", "mode", "max_gens", "test_acc"]
    # Build param order: keep prefix keys that are present, then append other params (sorted)
    present_prefix = [k for k in desired_prefix if k in params]
    other_keys = sorted([k for k in params.keys() if k not in desired_prefix and k != "elapsed_s"])
    param_order = present_prefix + other_keys
    # header: params in param_order, then targets, and optionally elapsed_s last
    if args.csv_header:
        header_fields = param_order + target_cols
        if "elapsed_s" in params:
            header_fields = header_fields + ["elapsed_s"]
        print(",".join(header_fields))
        return

    path = args.file
    if not path:
        print("No input file specified", file=sys.stderr)
        sys.exit(2)

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

    # If CSV header requested, emit header and exit
    csv_mode = bool(args.csv)
    params_list = args.param or []
    params = {}
    for pstr in params_list:
        if "=" in pstr:
            k, v = pstr.split("=", 1)
            params[k] = v
        else:
            params[pstr] = ""

    # Build CSV label names for targets in the same order as TARGETS
    target_cols = [LABELS.get(t, f"{t[0]}:{t[1]}") for t in TARGETS]
    # CSV header columns: compute param order again from provided params
    desired_prefix = ["size", "n_jobs", "mode", "max_gens", "test_acc"]
    present_prefix = [k for k in desired_prefix if k in params]
    other_keys = sorted([k for k in params.keys() if k not in desired_prefix and k != "elapsed_s"])
    param_order = present_prefix + other_keys
    if args.csv_header:
        header_fields = param_order + target_cols
        if "elapsed_s" in params:
            header_fields = header_fields + ["elapsed_s"]
        print(",".join(header_fields))
        return

    # print per-profile results (human readable) unless csv_mode
    if not csv_mode:
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
        # allow overriding denominator with provided n_jobs parameter
        denom = n
        try:
            nj_param = params.get("n_jobs")
            if nj_param is not None:
                nj_val = int(nj_param)
                if nj_val > 0:
                    denom = nj_val
        except Exception:
            # fall back to number of printed profiles if parsing fails
            denom = n

        if n > 0:
            print("Averages:")
            for target in TARGETS:
                # divide accumulated totals by denom (n_jobs if provided)
                avg = acc[target] / denom if denom != 0 else 1.0
                label = LABELS.get(target)
                if label:
                    print(f"  {label} ({target[0]}:{target[1]}) -> {avg:.2f} seconds (n={denom})")
                else:
                    print(f"  {target[0]}:{target[1]} -> {avg:.2f} seconds (n={denom})")
        return

    # CSV mode: print only averages as a single CSV line
    n = len(printed)
    if n == 0:
        # nothing to output
        return

    # allow overriding denominator with provided n_jobs parameter
    denom = n
    try:
        nj_param = params.get("n_jobs")
        if nj_param is not None:
            nj_val = int(nj_param)
            if nj_val > 0:
                denom = nj_val
    except Exception:
        denom = n

    avg_values = [acc[t] / denom if denom != 0 else 1.0 for t in TARGETS]
    # Build CSV row in explicit order: prefix params, then target averages, then elapsed_s
    # Build row fields in the computed param_order
    row_fields = [params.get(k, "") for k in param_order]
    # format averages with two decimal places
    row_fields += [f"{v:.2f}" for v in avg_values]
    # append elapsed_s after target columns only if present in params
    if "elapsed_s" in params:
        row_fields.append(params.get("elapsed_s", ""))
    print(",".join(row_fields))


if __name__ == "__main__":
    main()
