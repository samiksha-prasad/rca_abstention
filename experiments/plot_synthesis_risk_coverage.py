"""
Computes and plots risk-coverage curves for RAW, ARITHMETIC, and SYNTHESIS
confidence, all on the same chart, using a synthesis_results_<dataset>.csv
file produced by run_synthesis_experiment.py.

This directly answers the project's original research question for the
SYNTHESIS approach specifically: does its confidence actually track
correctness -- i.e., does being more selective about when it answers
genuinely reduce errors? A clean, monotonically-decreasing risk curve
means yes; a flat or inverted one means no, regardless of how good its
Brier score looks in aggregate.

Usage:
    python experiments/plot_synthesis_risk_coverage.py results/synthesis_results_re1.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval import metrics


def load_records(csv_path, guess_key, conf_key):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "decision": "diagnosed",
            "root_cause": r[guess_key],
            "confidence": float(r[conf_key]),
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ]


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/plot_synthesis_risk_coverage.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]
    dataset_label = os.path.basename(csv_path).replace("synthesis_results_", "").replace(".csv", "").upper()

    arms = {
        "RAW": ("raw_guess", "raw_confidence"),
        "ARITHMETIC": ("arithmetic_guess", "arithmetic_confidence"),
        "SYNTHESIS": ("synthesis_guess", "synthesis_confidence"),
    }
    colors = {"RAW": "#4C72B0", "ARITHMETIC": "#C44E52", "SYNTHESIS": "#55A868"}

    fig, ax = plt.subplots(figsize=(8, 6))

    print(f"\n=== Risk-coverage curves: {dataset_label} ===\n")
    for label, (guess_key, conf_key) in arms.items():
        records = load_records(csv_path, guess_key, conf_key)
        curve = metrics.risk_coverage_curve(records)

        coverages = [c["coverage"] for c in curve]
        risks = [c["risk"] for c in curve]

        print(f"--- {label} ---")
        for point in curve:
            risk_str = f"{point['risk']:.3f}" if point["risk"] == point["risk"] else "nan"  # nan check
            print(f"  threshold={point['threshold']:.2f}  coverage={point['coverage']:.2f}  risk={risk_str}")
        print()

        ax.plot(coverages, risks, marker="o", label=label, color=colors[label], linewidth=2, markersize=4)

    ax.set_xlabel("Coverage (fraction of incidents answered)", fontsize=12)
    ax.set_ylabel("Risk (error rate among answered incidents)", fontsize=12)
    ax.set_title(f"Risk-Coverage Curve Comparison -- {dataset_label}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_path = f"results/figures/risk_coverage_comparison_{dataset_label.lower()}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart to {out_path}")


if __name__ == "__main__":
    main()