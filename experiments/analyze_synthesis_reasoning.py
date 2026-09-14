"""
Qualitative analysis of the `synthesis_reasoning` text captured by
run_synthesis_experiment.py -- answers not just "does SYNTHESIS work
better" but "HOW does it work better": does the model explicitly say it's
distrusting noisy evidence? Deferring to its own prior? Something else?

Categorizes each case's reasoning text by simple keyword matching into:
  - "distrusts evidence"   (mentions unreliable/discount/ignore/despite/however)
  - "defers to evidence"   (mentions supports/confirms/consistent/aligns)
  - "other / mixed"        (neither pattern clearly present)

Then cross-tabulates this against flip type:
  - GOOD flip  (raw wrong -> synthesis correct)
  - BAD flip   (raw correct -> synthesis wrong)
  - no flip    (raw and synthesis guess the same)

And prints a few real example reasoning strings for each flip type, since
reading actual examples matters as much as the counts.

Usage:
    python experiments/analyze_synthesis_reasoning.py results/synthesis_results_re1.csv
"""
import csv
import sys

DISTRUST_KEYWORDS = [
    "unreliable", "discount", "ignore", "despite", "however", "override",
    "disagree", "not convinced", "outweigh", "own reasoning", "original",
    "noisy", "false positive", "doubt",
]
DEFER_KEYWORDS = [
    "supports", "confirms", "consistent", "aligns", "corroborates",
    "agrees with", "strong evidence", "clear evidence",
]


def categorize(reasoning_text):
    text = reasoning_text.lower()
    has_distrust = any(k in text for k in DISTRUST_KEYWORDS)
    has_defer = any(k in text for k in DEFER_KEYWORDS)
    if has_distrust and not has_defer:
        return "distrusts evidence"
    elif has_defer and not has_distrust:
        return "defers to evidence"
    else:
        return "other / mixed"


def flip_type(row):
    raw_correct = row["raw_guess"] == row["ground_truth"]
    synth_correct = row["synthesis_guess"] == row["ground_truth"]
    if row["raw_guess"] == row["synthesis_guess"]:
        return "no flip"
    if not raw_correct and synth_correct:
        return "GOOD flip"
    if raw_correct and not synth_correct:
        return "BAD flip"
    return "neutral flip"


def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/analyze_synthesis_reasoning.py <path_to_synthesis_results.csv>")
        return
    csv_path = sys.argv[1]

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if "synthesis_reasoning" not in rows[0]:
        print("This CSV doesn't have a 'synthesis_reasoning' column -- "
              "make sure you're pointing at a file from run_synthesis_experiment.py.")
        return

    # Cross-tab: flip type x reasoning category
    from collections import defaultdict
    table = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(list)

    for row in rows:
        ft = flip_type(row)
        cat = categorize(row.get("synthesis_reasoning", ""))
        table[ft][cat] += 1
        if len(examples[ft]) < 3:
            examples[ft].append(row.get("synthesis_reasoning", ""))

    print(f"\n=== Cross-tab: flip type x reasoning category ({len(rows)} total cases) ===\n")
    categories = ["distrusts evidence", "defers to evidence", "other / mixed"]
    print(f"{'Flip type':<15}" + "".join(f"{c:<22}" for c in categories) + "Total")
    for ft in ["GOOD flip", "BAD flip", "neutral flip", "no flip"]:
        counts = [table[ft][c] for c in categories]
        total = sum(counts)
        if total == 0:
            continue
        print(f"{ft:<15}" + "".join(f"{c:<22}" for c in counts) + str(total))

    print("\n=== Example reasoning text, by flip type ===")
    for ft in ["GOOD flip", "BAD flip"]:
        print(f"\n--- {ft} examples ---")
        for ex in examples[ft]:
            print(f"  \"{ex}\"")

    print("\nInterpretation guide:")
    print("  If GOOD flips skew toward 'distrusts evidence' and BAD flips skew toward")
    print("  'defers to evidence' (or vice versa), that tells you something concrete")
    print("  about WHEN the model's evidence-weighing judgment is trustworthy.")


if __name__ == "__main__":
    main()