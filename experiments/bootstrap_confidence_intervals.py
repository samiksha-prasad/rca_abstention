"""
Paired bootstrap confidence intervals for accuracy, and for the
differences between approaches (SYNTHESIS-ARITHMETIC, SYNTHESIS-RAW,
RAW-ARITHMETIC) -- paired because the same incidents were evaluated by
all three approaches, which is the statistically correct way to quantify
whether the observed gaps (e.g. 83.2% vs 44.8%) are more than sampling
noise.

Usage:
    python experiments/bootstrap_confidence_intervals.py results/synthesis_results_re2.csv
"""
import csv
import sys
import random


def accuracy(rows, guess_key):
    correct = sum(1 for r in rows if r[guess_key] == r["ground_truth"])
    return correct / len(rows)


def bootstrap_ci(rows, guess_key, n_boot=5000, ci=0.95, seed=42):
    """Resamples cases WITH replacement n_boot times, computing accuracy
    each time, and returns the point estimate + the ci% percentile interval."""
    rng = random.Random(seed)
    n = len(rows)
    point = accuracy(rows, guess_key)
    boot_accs = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        boot_accs.append(accuracy(sample, guess_key))
    boot_accs.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return point, boot_accs[lo_idx], boot_accs[hi_idx]


def bootstrap_paired_diff_ci(rows, guess_key_a, guess_key_b, n_boot=5000, ci=0.95, seed=42):
    """Paired bootstrap for accuracy_a - accuracy_b, resampling INCIDENTS
    (not each approach's results independently) -- correct because all
    three approaches were run on the SAME cases."""
    rng = random.Random(seed)
    n = len(rows)
    point = accuracy(rows, guess_key_a) - accuracy(rows, guess_key_b)
    boot_diffs = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        diff = accuracy(sample, guess_key_a) - accuracy(sample, guess_key_b)
        boot_diffs.append(diff)
    boot_diffs.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    lo, hi = boot_diffs[lo_idx], boot_diffs[hi_idx]
    # A CI that excludes 0 means the difference is unlikely to be pure noise.
    excludes_zero = (lo > 0) or (hi < 0)
    return point, lo, hi, excludes_zero


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/bootstrap_confidence_intervals.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    arms = {"RAW": "raw_guess", "ARITHMETIC": "arithmetic_guess", "SYNTHESIS": "synthesis_guess"}

    print(f"\n=== Accuracy with 95% bootstrap confidence intervals (n={len(rows)}, 5000 resamples) ===\n")
    for label, guess_key in arms.items():
        point, lo, hi = bootstrap_ci(rows, guess_key)
        print(f"  {label:<12}: {point:.3f}  [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== Paired differences (same incidents), 95% CI ===\n")
    comparisons = [
        ("SYNTHESIS - ARITHMETIC", "synthesis_guess", "arithmetic_guess"),
        ("SYNTHESIS - RAW", "synthesis_guess", "raw_guess"),
        ("RAW - ARITHMETIC", "raw_guess", "arithmetic_guess"),
    ]
    for label, key_a, key_b in comparisons:
        point, lo, hi, excludes_zero = bootstrap_paired_diff_ci(rows, key_a, key_b)
        sig_marker = " <-- CI excludes 0 (unlikely to be noise)" if excludes_zero else " (CI includes 0)"
        print(f"  {label:<26}: {point:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig_marker}")

    print("\nNote: these are bootstrap CIs, not a formal hypothesis test (e.g. McNemar's")
    print("test would be the more standard paired test for this kind of binary-outcome")
    print("comparison) -- but a CI that clearly excludes 0 across 5000 resamples is a")
    print("strong practical indicator the observed difference isn't just sampling noise.")


if __name__ == "__main__":
    main()