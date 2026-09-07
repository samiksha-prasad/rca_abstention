"""
Run this first. It exercises the ENTIRE loop -- hypothesis generation,
evidence selection, verification, confidence updates, and the final
diagnose/abstain decision -- on one synthetic incident, so you can watch it
work before touching real data or spending API budget.

Usage:
    python experiments/run_toy_example.py            # mock LLM, free
    python experiments/run_toy_example.py --live      # real Anthropic API call
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import make_synthetic_incident
from src.pipeline import diagnose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use the real LLM instead of the mock")
    args = parser.parse_args()

    incident = make_synthetic_incident()
    print(f"Incident: {incident['alert_text']}")
    print(f"Ground truth root cause: {incident['ground_truth']}\n")

    result = diagnose(incident, live=args.live)

    print("=== Evidence trail ===")
    for step in result.evidence_trail:
        print(f"  [{step['check']}] {step['hypothesis_id']}: {step['verdict']} -- {step['reason']}")

    print("\n=== Final hypotheses ===")
    for h in sorted(result.hypotheses, key=lambda x: -x["confidence"]):
        print(f"  {h['id']} ({h['service']}): confidence={h['confidence']}")

    print("\n=== Decision ===")
    print(json.dumps(result.to_dict(), indent=2))

    correct = result.root_cause == incident["ground_truth"]
    print(f"\nCorrect diagnosis? {correct} (decision={result.decision})")


if __name__ == "__main__":
    main()
