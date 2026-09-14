"""
McNemar's test comparing two SEPARATE result CSVs (e.g. an evidence-check
ablation condition vs. the full-checks baseline), joined by incident_id --
since run_evidence_ablation.py saves each condition to its own file rather
than adding columns to one shared file.

Usage:
    python experiments/mcnemar_test_two_files.py \\
        results/synthesis_results_re2.csv results/ablation_no_threshold_re2.csv \\
        "Full checks" "No threshold"
"""
import csv
import sys
from math import comb


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0, "no discordant pairs -- methods never disagreed"
    k = min(b, c)
    p_one_side = sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    p_two_sided = min(1.0, p_one_side * 2)
    return p_two_sided, None


def main():
    if len(sys.argv) < 5:
        print("Usage: python experiments/mcnemar_test_two_files.py "
              "<csv_a> <csv_b> <label_a> <label_b>")
        return
    csv_a_path, csv_b_path, label_a, label_b = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    with open(csv_a_path) as f:
        rows_a = {r["incident_id"]: r for r in csv.DictReader(f)}
    with open(csv_b_path) as f:
        rows_b = {r["incident_id"]: r for r in csv.DictReader(f)}

    shared_ids = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if len(shared_ids) < len(rows_a) or len(shared_ids) < len(rows_b):
        print(f"NOTE: only {len(shared_ids)} incidents are shared between the two files "
              f"({len(rows_a)} in A, {len(rows_b)} in B) -- comparing only the overlap.")

    both_correct = a_only_correct = b_only_correct = both_wrong = 0
    for incident_id in shared_ids:
        ra, rb = rows_a[incident_id], rows_b[incident_id]
        gt = ra["ground_truth"]
        a_correct = ra["synthesis_guess"] == gt
        b_correct = rb["synthesis_guess"] == gt
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct and not b_correct:
            a_only_correct += 1
        elif not a_correct and b_correct:
            b_only_correct += 1
        else:
            both_wrong += 1

    p_value, note = mcnemar_exact_p(a_only_correct, b_only_correct)

    print(f"\n=== McNemar's test: {label_a} vs {label_b} (n={len(shared_ids)}) ===\n")
    print(f"  Both correct:                     {both_correct}")
    print(f"  {label_a} right, {label_b} wrong:  {a_only_correct}")
    print(f"  {label_a} wrong, {label_b} right:  {b_only_correct}")
    print(f"  Both wrong:                       {both_wrong}")
    if note:
        print(f"  p-value: n/a ({note})")
    else:
        print(f"  Discordant pairs: {a_only_correct + b_only_correct}")
        print(f"  McNemar exact two-sided p-value: {p_value:.4f}")
        if p_value < 0.05:
            winner = label_a if a_only_correct > b_only_correct else label_b
            print(f"  --> SIGNIFICANT at p<0.05: {winner} wins the disagreements more often")
        else:
            print(f"  --> NOT significant at p<0.05 -- difference could plausibly be chance")


if __name__ == "__main__":
    main()