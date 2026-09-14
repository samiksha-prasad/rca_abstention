"""
Check-type attribution via text mining: which evidence check types does the
LLM's synthesis reasoning actually lean on, and does mentioning a
particular check type correlate with being MORE or LESS likely correct?

This doesn't require a new experiment run -- it re-mines the
synthesis_reasoning text already saved by run_synthesis_experiment.py.

Usage:
    python experiments/analyze_check_attribution.py results/synthesis_results_re2.csv
"""
import csv
import sys
from collections import defaultdict

# Keywords a reasoning string might use to reference each check type.
# Deliberately generous (multiple phrasings) since the LLM doesn't always
# use the exact internal check names.
CHECK_KEYWORDS = {
    "temporal_ordering": ["temporal", "onset", "earliest", "timing", "before", "precede"],
    "threshold": ["threshold", "deviat", "spike", "cpu", "mem", "std", "baseline"],
    "topology_consistency": ["topology", "dependen", "downstream", "upstream", "propagat"],
    "log_pattern": ["log", "error pattern", "error message"],
}


def mentioned_checks(reasoning_text):
    text = reasoning_text.lower()
    return [check for check, keywords in CHECK_KEYWORDS.items() if any(k in text for k in keywords)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/analyze_check_attribution.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if "synthesis_reasoning" not in rows[0]:
        print("This CSV doesn't have a 'synthesis_reasoning' column.")
        return

    # For each check type: how often mentioned, and accuracy when mentioned vs not.
    mention_count = defaultdict(int)
    correct_when_mentioned = defaultdict(int)
    correct_when_not_mentioned = defaultdict(int)
    total_when_mentioned = defaultdict(int)
    total_when_not_mentioned = defaultdict(int)

    for row in rows:
        checks_mentioned = mentioned_checks(row.get("synthesis_reasoning", ""))
        is_correct = row["synthesis_guess"] == row["ground_truth"]

        for check in CHECK_KEYWORDS:
            if check in checks_mentioned:
                mention_count[check] += 1
                total_when_mentioned[check] += 1
                if is_correct:
                    correct_when_mentioned[check] += 1
            else:
                total_when_not_mentioned[check] += 1
                if is_correct:
                    correct_when_not_mentioned[check] += 1

    n = len(rows)
    print(f"\n=== Check-type attribution ({n} total cases) ===\n")
    print(f"{'Check type':<24}{'Mentioned':<12}{'Acc if mentioned':<20}{'Acc if NOT mentioned':<20}")
    for check in CHECK_KEYWORDS:
        mentioned_n = total_when_mentioned[check]
        not_mentioned_n = total_when_not_mentioned[check]
        acc_mentioned = correct_when_mentioned[check] / mentioned_n if mentioned_n else float("nan")
        acc_not_mentioned = correct_when_not_mentioned[check] / not_mentioned_n if not_mentioned_n else float("nan")
        print(f"{check:<24}{mentioned_n:<12}{acc_mentioned:<20.3f}{acc_not_mentioned:<20.3f}")

    print("\nInterpretation guide:")
    print("  If 'Acc if mentioned' is clearly HIGHER than 'Acc if NOT mentioned' for a")
    print("  check type, that check's evidence tends to be genuinely useful when the")
    print("  model leans on it. If LOWER, that check may be actively misleading the")
    print("  model when it gets invoked in reasoning -- a candidate for improvement")
    print("  or removal from the evidence set.")


if __name__ == "__main__":
    main()