"""
Tests whether EVIDENCE CONSENSUS -- how many checks supported vs.
contradicted the raw guess, a signal completely independent of any stated
confidence number -- predicts correctness better than the model's own
RAW or SYNTHESIS confidence.

Requires a synthesis_results CSV produced by the UPDATED
run_synthesis_experiment.py (must have evidence_consensus_net,
raw_guess_supports, raw_guess_contradicts columns -- older CSVs won't
have these and this script will tell you so).

Usage:
    python experiments/analyze_evidence_consensus.py results/synthesis_results_re2.csv
"""
import csv
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/analyze_evidence_consensus.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if "evidence_consensus_net" not in rows[0]:
        print("This CSV doesn't have 'evidence_consensus_net' -- it was produced by an "
              "older version of run_synthesis_experiment.py. Re-run the experiment with "
              "the updated script to get this column.")
        return

    # Bucket by evidence_consensus_net value, check accuracy of RAW guess
    # in each bucket (does more net support for the raw guess actually
    # correlate with it being right?)
    by_consensus = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        net = int(row["evidence_consensus_net"])
        # Group into buckets: strongly negative, negative, zero, positive, strongly positive
        if net <= -2:
            bucket = "<= -2 (checks strongly disagree)"
        elif net == -1:
            bucket = "-1"
        elif net == 0:
            bucket = "0 (no net signal)"
        elif net == 1:
            bucket = "+1"
        else:
            bucket = ">= +2 (checks strongly agree)"

        by_consensus[bucket]["total"] += 1
        if row["raw_guess"] == row["ground_truth"]:
            by_consensus[bucket]["correct"] += 1

    print(f"\n=== Does evidence consensus predict RAW guess correctness? ({len(rows)} cases) ===\n")
    print(f"{'Consensus bucket':<35}{'N':<6}{'Accuracy':<10}")
    order = ["<= -2 (checks strongly disagree)", "-1", "0 (no net signal)", "+1", ">= +2 (checks strongly agree)"]
    for bucket in order:
        stats = by_consensus[bucket]
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        print(f"{bucket:<35}{stats['total']:<6}{acc:<10.3f}")

    # Compare: does evidence_consensus_net correlate with correctness at
    # least as well as raw_confidence does? Simple check: split cases into
    # "high raw_confidence" vs "high consensus" and compare which set is
    # more accurate.
    print("\n=== Direct comparison: ranking by stated confidence vs. by evidence consensus ===")
    sorted_by_confidence = sorted(rows, key=lambda r: -float(r["raw_confidence"]))
    sorted_by_consensus = sorted(rows, key=lambda r: -int(r["evidence_consensus_net"]))

    top_n = max(1, len(rows) // 4)  # top quartile by each ranking
    top_by_confidence = sorted_by_confidence[:top_n]
    top_by_consensus = sorted_by_consensus[:top_n]

    acc_top_confidence = sum(1 for r in top_by_confidence if r["raw_guess"] == r["ground_truth"]) / top_n
    acc_top_consensus = sum(1 for r in top_by_consensus if r["raw_guess"] == r["ground_truth"]) / top_n

    print(f"  Top quartile by RAW confidence:      accuracy = {acc_top_confidence:.3f}")
    print(f"  Top quartile by evidence consensus:  accuracy = {acc_top_consensus:.3f}")

    print("\nInterpretation guide:")
    print("  If accuracy climbs steadily as the consensus bucket goes from strongly")
    print("  negative to strongly positive, evidence consensus IS a real, independent")
    print("  signal of correctness -- and the two 'top quartile' numbers tell you")
    print("  whether it's MORE or LESS informative than the model's own stated")
    print("  confidence for picking out the most trustworthy cases.")


if __name__ == "__main__":
    main()