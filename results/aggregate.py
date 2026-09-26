#!/usr/bin/env python3
# Copyright (c) 2026 Mahesh Godavarti
# Licensed under CC BY-NC-SA 4.0. See LICENSE file for details.

"""Result tables: hits@5 and perplexity on unseen names, averaged over the five
positions of the true fact among the priming facts, mean ± sample standard
deviation over training seeds. Reads results_pos0.json .. results_pos4.json.

    python results/aggregate.py
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS = [f"pos{k}" for k in range(5)]
ROWS = [(ft, mode, f"{ft}, {label}")
        for ft in ("chain2", "chain3")
        for mode, label in (("none", "unprimed"), ("same_entities", "same entities"),
                            ("same_rel", "same relation"))]


def main():
    data = {}
    for pos in POSITIONS:
        p = os.path.join(HERE, f"results_{pos}.json")
        if os.path.exists(p):
            data[pos] = json.load(open(p))
    if not data:
        print("no results files"); return
    first = next(iter(data.values()))
    models = first["config"]["models"]
    n_seeds = len(first["per_seed"])
    for metric, name in (("hit5", "hits@5"), ("ppl", "perplexity")):
        print(f"\n### {name}, unseen names, averaged over the true fact's position "
              f"(mean ± std over {n_seeds} seeds; {len(data)} positions)\n")
        print("| | " + " | ".join(models) + " |")
        print("|---|" + "---|" * len(models))
        for ft, mode, label in ROWS:
            key = str(('kg_to_text', ft, 'test', mode))
            cells = []
            for m in models:
                per_seed = [statistics.mean(data[pos]["per_seed"][si][m][key][metric]
                                            for pos in data) for si in range(n_seeds)]
                nd = 2 if metric == "ppl" else 3
                cells.append(f"{statistics.mean(per_seed):.{nd}f} ± {statistics.stdev(per_seed):.{nd}f}"
                             if n_seeds > 1 else f"{per_seed[0]:.{nd}f}")
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
