"""
Reads results/ablation_results.csv (produced by run_ablation.py) and asks:
does whether verification HELPS or HURTS depend on how confident the raw
LLM guess already was?

Buckets cases by raw_confidence into low/medium/high, and within each
bucket reports:
  - how often verification changed the guess at all
  - of those changes, how many FIXED a wrong raw guess (good flip)
  - how many BROKE a correct raw guess (bad flip)
  - how many changed a wrong guess into a different wrong guess (neutral)

Usage:
    python experiments/analyze_flips.py results/ablation_results.csv
"""
import csv
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/ablation_results.csv"

    with open(path) as f:
        rows = list(csv.DictReader(f))

    buckets = {
        "low (<0.35)": lambda c: c < 0.35,
        "medium (0.35-0.50)": lambda c: 0.35 <= c < 0.50,
        "high (>=0.50)": lambda c: c >= 0.50,
    }

    print(f"{'Bucket':<20}{'N':<6}{'Changed':<10}{'Good flip':<12}{'Bad flip':<11}{'Neutral':<10}{'Raw acc':<10}{'Final acc':<10}")
    print("-" * 90)

    for label, in_bucket in buckets.items():
        subset = [r for r in rows if in_bucket(float(r["raw_confidence"]))]
        if not subset:
            print(f"{label:<20}{'0':<6}(no cases)")
            continue

        n = len(subset)
        changed = sum(1 for r in subset if r["raw_guess"] != r["final_guess"])

        good_flip = sum(
            1 for r in subset
            if r["raw_guess"] != r["final_guess"]
            and r["raw_guess"] != r["ground_truth"]
            and r["final_guess"] == r["ground_truth"]
        )
        bad_flip = sum(
            1 for r in subset
            if r["raw_guess"] != r["final_guess"]
            and r["raw_guess"] == r["ground_truth"]
            and r["final_guess"] != r["ground_truth"]
        )
        neutral_flip = changed - good_flip - bad_flip

        raw_acc = sum(1 for r in subset if r["raw_guess"] == r["ground_truth"]) / n
        final_acc = sum(1 for r in subset if r["final_guess"] == r["ground_truth"]) / n

        print(f"{label:<20}{n:<6}{changed:<10}{good_flip:<12}{bad_flip:<11}{neutral_flip:<10}"
              f"{raw_acc:<10.3f}{final_acc:<10.3f}")

    print("\nInterpretation:")
    print("  'Bad flip' > 'Good flip' in a bucket means verification is net-harmful there.")
    print("  If bad flips concentrate in the 'high' bucket, that supports the idea:")
    print("  trust the LLM more when it's already confident, and let verification")
    print("  focus on low/medium-confidence cases instead of overriding confident ones.")


if __name__ == "__main__":
    main()