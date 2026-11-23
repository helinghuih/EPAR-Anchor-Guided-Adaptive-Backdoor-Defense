#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Significance Analysis for Table 2 (EPAR vs. Seven Baselines).

Runs setting-level exact two-sided paired Wilcoxon signed-rank tests across the
N = 14 dataset-attack configurations in Table 2 (5 on CIFAR-10, 5 on GTSRB,
4 on Tiny-ImageNet), pairing EPAR's 3-seed mean against each baseline's
shared seed-666 benchmark record for ACC and ASR separately, together with
Holm-Bonferroni multiple-comparison adjustment across the 7 baselines.

Usage:
    python table2_wilcoxon_stats.py
    python table2_wilcoxon_stats.py --results_json results/per_seed_results.json
"""

import argparse
import json
import os
import numpy as np
from scipy.stats import wilcoxon

SETTINGS = [
    ("CIFAR-10", "BadNets"), ("CIFAR-10", "Blended"), ("CIFAR-10", "Refool"),
    ("CIFAR-10", "WaNet"),   ("CIFAR-10", "SIG"),
    ("GTSRB", "BadNets"),    ("GTSRB", "Blended"),    ("GTSRB", "Refool"),
    ("GTSRB", "WaNet"),      ("GTSRB", "SIG"),
    ("Tiny-ImageNet", "BadNets"), ("Tiny-ImageNet", "Blended"),
    ("Tiny-ImageNet", "Refool"),  ("Tiny-ImageNet", "WaNet"),
]

BASELINES = ["ABL", "NAD", "DBD", "PDB", "REFINE", "EBYD", "AIBD"]

TABLE2_ACC = {
    "ABL":    [91.48, 90.77, 59.02, 92.26, 74.90,  97.67, 92.19, 94.94, 80.22, 89.53,  48.17, 45.24,  9.14, 23.36],
    "NAD":    [90.87, 91.86, 87.50, 91.62, 89.98,  87.62, 79.04, 91.66, 91.55, 86.54,  50.67, 50.27, 50.24, 49.82],
    "DBD":    [91.97, 91.49, 88.28, 91.28, 83.52,  83.86, 85.20, 84.86, 84.38, 80.58,  37.54, 48.73, 46.38, 44.54],
    "PDB":    [90.34, 91.37, 88.09, 90.89, 91.59,  94.67, 95.09, 95.06, 94.60, 94.49,  41.71, 40.24, 40.85, 41.23],
    "REFINE": [89.83, 89.48, 90.54, 89.42, 90.13,  93.78, 97.07, 96.87, 95.91, 96.22,  44.98, 45.13, 44.80, 45.62],
    "EBYD":   [85.72, 87.13, 86.95, 85.19, 84.78,  97.74, 97.82, 97.87, 97.70, 97.59,  46.12, 45.78, 45.34, 46.05],
    "AIBD":   [91.78, 91.38, 92.39, 92.04, 93.67,  91.35, 92.14, 91.39, 91.68, 92.46,  51.95, 50.57, 51.77, 50.17],
    "EPAR":   [93.80, 93.62, 93.70, 93.59, 94.00,  98.23, 98.04, 97.79, 97.70, 97.78,  50.85, 50.74, 51.02, 51.09],
}

TABLE2_ASR = {
    "ABL":    [0.11,  2.50, 99.41,  1.14,  0.00,   0.02,  4.69, 65.94,  1.20, 95.75,   0.15,  0.01,  0.33,  0.05],
    "NAD":    [11.15, 7.31,  1.78,  1.75,  0.71,   0.34,  5.30, 17.45,  0.99, 12.74,   0.20, 96.28, 58.13, 20.59],
    "DBD":    [2.25,  4.48,  4.05,  2.15,  0.44,   0.01, 54.50, 45.79,  0.01,  3.70,  96.29, 99.98, 99.98, 99.98],
    "PDB":    [0.85,  4.52,  1.76,  0.35,  0.12,   0.00,  1.72,  1.60,  0.01,  0.31,   0.00,  0.00,  0.00,  0.00],
    "REFINE": [1.69,  1.89,  1.45,  1.39,  0.11,   0.26,  0.23,  2.87,  0.00,  1.66,   0.23,  0.00,  0.56,  0.17],
    "EBYD":   [1.77, 14.02,  2.54,  2.80,  1.73,   0.02, 43.42,  9.49,  0.34, 50.51,   1.85, 18.42, 12.65,  3.40],
    "AIBD":   [0.11, 25.12, 98.96, 99.12,  0.01,   0.89,  1.86, 10.71,  8.15,  0.01,   4.11,  4.19,  3.91,  4.63],
    "EPAR":   [0.30,  1.31,  0.35,  0.52,  0.04,   0.18,  0.60,  0.55,  0.09,  1.07,   0.13,  0.04,  0.96,  0.31],
}


def holm_bonferroni(p_values):
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_values[idx]) * (m - rank))
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted


def run_wilcoxon_table(acc_dict, asr_dict):
    print("=" * 98)
    print("  Setting-Level Exact Two-Sided Paired Wilcoxon Signed-Rank Tests (N = 14 Configurations)")
    print("=" * 98)

    for metric_name, data in [("ACC (higher is better)", acc_dict), ("ASR (lower is better)", asr_dict)]:
        print(f"\nMetric: {metric_name}")
        print(
            f"  {'Baseline':<10} | {'W-stat':>6} | {'Zeros':>5} | "
            f"{'p (exact)':>10} | {'p (Holm)':>10} | {'p (approx)':>10} | {'Sig (Holm<0.01)':>15}"
        )
        print("  " + "-" * 84)
        epar_vals = np.asarray(data["EPAR"], dtype=np.float64)
        rows = []
        for b in BASELINES:
            base_vals = np.asarray(data[b], dtype=np.float64)
            n_zeros = int(np.sum((epar_vals - base_vals) == 0))
            res_exact = wilcoxon(epar_vals, base_vals, alternative="two-sided", method="exact")
            res_approx = wilcoxon(epar_vals, base_vals, alternative="two-sided", method="approx")
            rows.append((b, res_exact.statistic, n_zeros, res_exact.pvalue, res_approx.pvalue))

        holm_ps = holm_bonferroni([r[3] for r in rows])
        for (b, w_stat, n_zeros, p_ex, p_ap), p_holm in zip(rows, holm_ps):
            sig = "YES" if p_holm < 0.01 else "NO"
            print(
                f"  {b:<10} | {w_stat:6.1f} | {n_zeros:5d} | "
                f"{p_ex:10.5f} | {p_holm:10.5f} | {p_ap:10.5f} | {sig:>15}"
            )


def main():
    parser = argparse.ArgumentParser(description="Wilcoxon signed-rank statistical analysis for EPAR Table 2")
    parser.add_argument("--results_json", type=str, default=None,
                        help="Optional path to per-seed JSON results to verify and test")
    args = parser.parse_args()

    acc_dict, asr_dict = TABLE2_ACC, TABLE2_ASR
    if args.results_json and os.path.exists(args.results_json):
        with open(args.results_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "epar_per_seed" in payload:
            epar_acc, epar_asr = [], []
            for ds, atk in SETTINGS:
                rec = payload["epar_per_seed"][ds][atk]
                epar_acc.append(round(float(np.mean(rec["ACC"])), 2))
                epar_asr.append(round(float(np.mean(rec["ASR"])), 2))
            acc_dict = {**TABLE2_ACC, "EPAR": epar_acc}
            asr_dict = {**TABLE2_ASR, "EPAR": epar_asr}
            print(f"[Loaded per-seed EPAR observations from {args.results_json}]")

    run_wilcoxon_table(acc_dict, asr_dict)


if __name__ == "__main__":
    main()