"""Compares the PyTorch replication with the unregularised runs of the main grid (rust-lstm).

    python compare.py results/main/runs.csv results/pytorch/runs.csv
"""
import csv
import statistics
import sys
from collections import defaultdict


def load(path, rust):
    groups = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if rust and r["regulariser"] != "none":
                continue
            if int(r["seed"]) > 4:
                continue
            key = (r["arch"], r["task"], int(r["length"]), r["init"])
            groups[key].append(int(r["solved_at"]) if r["solved_at"] else None)
    return groups


def summary(values):
    solved = [v for v in values if v is not None]
    median = statistics.median(solved) if solved else None
    return f"{len(solved)}/{len(values)}", median


def main():
    rust, torch = load(sys.argv[1], True), load(sys.argv[2], False)
    print("arch,task,length,init,rust_solved,rust_median_updates,torch_solved,torch_median_updates")
    for key in sorted(set(rust) | set(torch)):
        r, t = summary(rust.get(key, [])), summary(torch.get(key, []))
        print(",".join(map(str, (*key, *r, *t))))


if __name__ == "__main__":
    main()
