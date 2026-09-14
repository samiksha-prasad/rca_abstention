"""
McNemar's test: the standard statistical test for comparing two methods'
accuracy on PAIRED binary outcomes -- exactly this project's situation,
since RAW, ARITHMETIC, and SYNTHESIS were all evaluated on the SAME
incidents.

Unlike a general bootstrap CI (used elsewhere in this project), McNemar's
test looks only at the DISCORDANT pairs -- cases where the two methods
disagree (one right, one wrong) -- since agreements (both right or both
wrong) carry no information about which method is actually better. This
is the more standard, purpose-built test for this exact comparison.

Uses the exact binomial form (appropriate for small-to-moderate numbers of
discordant pairs) rather than the chi-square approximation, which can be
unreliable when discordant counts are small.

Usage:
    python experiments/mcnemar_test.py results/synthesis_results_re2.csv
"""
import csv
import sys
from math import comb


def mcnemar_exact_p(b, c):
    """
    Exact two-sided McNemar test p-value via the binomial distribution.
    b = count where method A right, method B wrong
    c = count where method A wrong, method B right
    Under the null hypothesis (no real difference), b and c should each be
    about half of (b+c) by chance.
    """
    n = b + c
    if n == 0:
        return 1.0, "no discordant pairs -- methods never disagreed"
    k = min(b, c)
    p_one_side = sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    p_two_sided = min(1.0, p_one_side * 2)
    return p_two_sided, None


def compare(rows, key_a, key_b, label_a, label_b):
    both_correct = 0
    a_only_correct = 0
    b_only_correct = 0
    both_wrong = 0

    for r in rows:
        a_correct = r[key_a] == r["ground_truth"]
        b_correct = r[key_b] == r["ground_truth"]
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only_correct += 1
        elif not a_correct and b_correct:
            b_only_correct += 1
        else:
            both_wrong += 1

    p_value, note = mcnemar_exact_p(a_only_correct, b_only_correct)

    print(f"\n--- {label_a} vs {label_b} ---")
    print(f"  Both correct:              {both_correct}")
    print(f"  {label_a} right, {label_b} wrong:  {a_only_correct}")
    print(f"  {label_a} wrong, {label_b} right:  {b_only_correct}")
    print(f"  Both wrong:                {both_wrong}")
    if note:
        print(f"  p-value: n/a ({note})")
    else:
        print(f"  Discordant pairs: {a_only_correct + b_only_correct}")
        print(f"  McNemar exact two-sided p-value: {p_value:.4f}")
        if p_value < 0.05:
            winner = label_a if a_only_correct > b_only_correct else label_b
            print(f"  --> SIGNIFICANT at p<0.05: {winner} wins the disagreements more often")
        else:
            print(f"  --> NOT significant at p<0.05 -- disagreements could plausibly be chance")


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/mcnemar_test.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    print(f"\n=== McNemar's test (n={len(rows)} paired incidents) ===")

    compare(rows, "synthesis_guess", "arithmetic_guess", "SYNTHESIS", "ARITHMETIC")
    compare(rows, "raw_guess", "synthesis_guess", "RAW", "SYNTHESIS")
    compare(rows, "raw_guess", "arithmetic_guess", "RAW", "ARITHMETIC")

    print("\nNote: this uses the EXACT binomial form of McNemar's test, appropriate")
    print("regardless of how small the discordant counts are (the common chi-square")
    print("approximation can be unreliable with small samples).")


if __name__ == "__main__":
    main()