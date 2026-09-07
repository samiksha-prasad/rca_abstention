"""
The two headline figures for the project.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_risk_coverage(curve: list, save_path: str = None):
    coverages = [c["coverage"] for c in curve]
    risks = [c["risk"] for c in curve]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(coverages, risks, marker="o")
    ax.set_xlabel("Coverage (fraction of incidents answered)")
    ax.set_ylabel("Risk (error rate among answered incidents)")
    ax.set_title("Risk-Coverage Curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_reliability_diagram(records: list, n_bins: int = 10, save_path: str = None):
    """Standard calibration plot: mean predicted confidence vs. observed
    accuracy, per confidence bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confidences, bin_accuracies, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [r for r in records if lo <= r["confidence"] < hi]
        if not in_bin:
            continue
        mean_conf = np.mean([r["confidence"] for r in in_bin])
        acc = np.mean([
            1.0 if (r["decision"] == "diagnosed" and r["root_cause"] == r["ground_truth"]) else 0.0
            for r in in_bin
        ])
        bin_confidences.append(mean_conf)
        bin_accuracies.append(acc)
        bin_counts.append(len(in_bin))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(bin_confidences, bin_accuracies, s=[c * 5 for c in bin_counts], alpha=0.7)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
