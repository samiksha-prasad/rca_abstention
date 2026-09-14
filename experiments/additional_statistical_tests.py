"""
Three additional statistical tests, closing real gaps left by the
bootstrap CIs and McNemar's test alone:

1. COCHRAN'S Q TEST -- an omnibus test across all three paired methods at
   once (RAW, ARITHMETIC, SYNTHESIS), which should be run BEFORE drilling
   into pairwise McNemar tests. Running three separate pairwise tests
   without an omnibus test first, and without correcting for multiple
   comparisons, risks inflating the false-positive rate.

2. BONFERRONI CORRECTION -- re-checks whether the three pairwise McNemar
   p-values survive a stricter significance threshold (0.05 / 3 tests)
   appropriate for multiple comparisons.

3. WILCOXON SIGNED-RANK TEST -- tests whether SYNTHESIS's better Brier
   score (calibration) is actually statistically significant, not just a
   numerically smaller value. This closes a real gap: this project
   statistically tested the ACCURACY difference (McNemar, bootstrap) but
   never tested whether the CALIBRATION difference is real, even though
   "SYNTHESIS is better calibrated despite lower accuracy" is one of the
   paper's more nuanced central claims.

Usage:
    python experiments/additional_statistical_tests.py results/synthesis_results_re2.csv
"""
import csv
import sys
from scipy.stats import chi2, wilcoxon


def cochrans_q(rows, guess_keys):
    """
    Cochran's Q test for k >= 2 paired binary outcomes (here k=3: RAW,
    ARITHMETIC, SYNTHESIS). Tests whether there are ANY significant
    differences among the k methods, before running pairwise follow-ups.
    """
    k = len(guess_keys)
    n = len(rows)

    # Binary correctness matrix: outcomes[i][j] = 1 if method j correct on case i
    outcomes = []
    for r in rows:
        outcomes.append([1 if r[key] == r["ground_truth"] else 0 for key in guess_keys])

    col_sums = [sum(outcomes[i][j] for i in range(n)) for j in range(k)]  # per-method total correct
    row_sums = [sum(outcomes[i][j] for j in range(k)) for i in range(n)]  # per-case agreement count

    numerator = (k - 1) * (k * sum(c ** 2 for c in col_sums) - sum(col_sums) ** 2)
    denominator = k * sum(row_sums) - sum(r ** 2 for r in row_sums)

    if denominator == 0:
        return None, None  # degenerate case, can't compute

    q_stat = numerator / denominator
    df = k - 1
    p_value = 1 - chi2.cdf(q_stat, df)
    return q_stat, p_value


def per_case_squared_error(rows, guess_key, conf_key):
    """Per-case Brier component: (confidence - correctness)^2."""
    errors = []
    for r in rows:
        correct = 1.0 if r[guess_key] == r["ground_truth"] else 0.0
        conf = float(r[conf_key])
        errors.append((conf - correct) ** 2)
    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/additional_statistical_tests.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # --- 1. Cochran's Q omnibus test ---
    print(f"\n=== 1. Cochran's Q test: are RAW, ARITHMETIC, SYNTHESIS different AT ALL? (n={len(rows)}) ===\n")
    q_stat, p_value = cochrans_q(rows, ["raw_guess", "arithmetic_guess", "synthesis_guess"])
    if q_stat is not None:
        print(f"  Q statistic: {q_stat:.3f}  (chi-square, df=2)")
        print(f"  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print("  --> SIGNIFICANT: at least one method differs from the others.")
            print("      (Proceed to pairwise McNemar tests to find out which.)")
        else:
            print("  --> NOT significant: no strong evidence any method differs.")
    else:
        print("  Could not compute (degenerate case).")

    # --- 2. Bonferroni correction on existing McNemar p-values ---
    print(f"\n=== 2. Bonferroni-corrected significance check (3 comparisons) ===\n")
    n_comparisons = 3
    bonferroni_alpha = 0.05 / n_comparisons
    print(f"  Adjusted significance threshold: 0.05 / {n_comparisons} = {bonferroni_alpha:.4f}")
    print(f"  (Re-check your McNemar p-values from mcnemar_test.py against this stricter")
    print(f"   threshold, not the usual 0.05, since 3 pairwise tests were run on the same data.)")

    # --- 3. Wilcoxon signed-rank test on calibration (Brier components) ---
    print(f"\n=== 3. Wilcoxon signed-rank test: is the CALIBRATION difference significant? ===\n")
    arms = {"RAW": ("raw_guess", "raw_confidence"),
            "ARITHMETIC": ("arithmetic_guess", "arithmetic_confidence"),
            "SYNTHESIS": ("synthesis_guess", "synthesis_confidence")}

    errors = {label: per_case_squared_error(rows, gk, ck) for label, (gk, ck) in arms.items()}

    comparisons = [("SYNTHESIS", "RAW"), ("SYNTHESIS", "ARITHMETIC"), ("RAW", "ARITHMETIC")]
    for a, b in comparisons:
        try:
            stat, p = wilcoxon(errors[a], errors[b])
            mean_a = sum(errors[a]) / len(errors[a])
            mean_b = sum(errors[b]) / len(errors[b])
            print(f"  {a} (mean sq. error={mean_a:.3f}) vs {b} (mean sq. error={mean_b:.3f})")
            print(f"    Wilcoxon statistic={stat:.1f}, p-value={p:.4f}", end="  ")
            print("--> SIGNIFICANT" if p < 0.05 else "--> not significant")
        except ValueError as e:
            print(f"  {a} vs {b}: could not compute ({e})")

    print("\nInterpretation guide:")
    print("  If Cochran's Q is significant AND the SYNTHESIS/RAW Wilcoxon test on Brier")
    print("  components is ALSO significant, that's real statistical support for the")
    print("  paper's nuanced claim: SYNTHESIS is significantly worse on accuracy (per")
    print("  McNemar/bootstrap) but ALSO significantly better on calibration (per this")
    print("  test) -- i.e., these really are separable, both statistically real properties,")
    print("  not just two numbers that happened to move in different directions.")


if __name__ == "__main__":
    main()