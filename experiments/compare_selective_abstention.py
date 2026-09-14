"""
Ties the whole project back to its original question directly: at the
SAME coverage level (e.g. "answer 50% of incidents"), does SYNTHESIS +
abstention actually produce fewer mistakes than RAW + abstention?

Rather than showing two separate curves and asking the reader to compare
them visually, this finds, for each target coverage level, the confidence
threshold that achieves it for EACH approach, and reports accuracy at
that matched point directly.

Usage:
    python experiments/compare_selective_abstention.py results/synthesis_results_re2.csv
"""
import csv
import sys

sys.path.insert(0, __file__.rsplit("/experiments", 1)[0])
from eval import metrics


def load_records(rows, guess_key, conf_key):
    return [
        {"decision": "diagnosed", "root_cause": r[guess_key], "confidence": float(r[conf_key]),
         "ground_truth": r["ground_truth"]}
        for r in rows
    ]


def accuracy_at_coverage(records, target_coverage, tolerance=0.05):
    """Finds the finest threshold that gets coverage close to the target,
    and returns (achieved_coverage, accuracy_at_that_point)."""
    curve = metrics.risk_coverage_curve(records, thresholds=[i / 100 for i in range(0, 101)])
    best = min(curve, key=lambda c: abs(c["coverage"] - target_coverage) if c["coverage"] == c["coverage"] else 999)
    accuracy = 1 - best["risk"] if best["risk"] == best["risk"] else float("nan")
    return best["coverage"], accuracy


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/compare_selective_abstention.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    raw_records = load_records(rows, "raw_guess", "raw_confidence")
    synth_records = load_records(rows, "synthesis_guess", "synthesis_confidence")

    print(f"\n=== RAW+abstention vs SYNTHESIS+abstention at matched coverage (n={len(rows)}) ===\n")
    print(f"{'Target coverage':<18}{'RAW accuracy':<18}{'SYNTHESIS accuracy':<20}{'Winner'}")

    for target in [0.25, 0.50, 0.75, 1.00]:
        raw_cov, raw_acc = accuracy_at_coverage(raw_records, target)
        synth_cov, synth_acc = accuracy_at_coverage(synth_records, target)

        if raw_acc != raw_acc or synth_acc != synth_acc:  # nan check
            winner = "n/a (insufficient data at this coverage)"
        elif abs(raw_acc - synth_acc) < 0.01:
            winner = "tie"
        else:
            winner = "RAW" if raw_acc > synth_acc else "SYNTHESIS"

        print(f"{target:.0%} (actual: {raw_cov:.0%}/{synth_cov:.0%})".ljust(18) +
              f"{raw_acc:.3f}".ljust(18) + f"{synth_acc:.3f}".ljust(20) + winner)

    print("\nInterpretation guide:")
    print("  This directly answers the project's original question: if you have to")
    print("  answer roughly the same FRACTION of incidents either way, which approach's")
    print("  abstention mechanism leaves you with fewer mistakes among the ones it")
    print("  DOES answer? 'Actual' coverage may differ slightly from the target since")
    print("  thresholds only take discrete values in the underlying confidence data.")


if __name__ == "__main__":
    main()