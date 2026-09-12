"""
Ablation: compares the LLM's RAW, unverified confidence (right after
hypothesis generation, before any evidence check runs) against the FINAL
confidence after the full evidence-verification loop -- on the SAME cases,
so the comparison is controlled.

This exists to answer a specific question raised by a real result: running
diagnose() end-to-end with --live showed risk INCREASING as confidence
increased across most of the range (the opposite of what a calibrated
system should do). That could mean either (a) the LLM's own confidence is
miscalibrated and the verification loop doesn't fix it, or (b) something
else is going on. Comparing raw vs. final confidence directly is the only
way to tell which.

Usage:
    python experiments/run_ablation.py --dataset re1 --limit 30          # mock, free
    python experiments/run_ablation.py --dataset re1 --limit 30 --live  # real LLM
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src import config
from src.data_loader import load_rcaeval_case, load_re2_case
from src.llm_client import generate_hypotheses
from src.pipeline import _select_next_check, _apply_evidence, _normalize
from src.evidence import CHECK_REGISTRY
from eval import metrics


def find_cases(root, main_metrics_filename, limit=None):
    cases = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if main_metrics_filename in filenames:
            cases.append(dirpath)
    cases.sort()
    if limit:
        cases = cases[:limit]
    return cases


def run_case(incident, live, provider):
    """
    Generates hypotheses, records the RAW top guess/confidence BEFORE any
    evidence check, then runs the exact same verification loop diagnose()
    uses, and records the FINAL top guess/confidence after it.
    """
    hypotheses = generate_hypotheses(incident, live=live, provider=provider)
    _normalize(hypotheses)

    raw_top = max(hypotheses, key=lambda h: h.confidence)
    raw_guess, raw_confidence = raw_top.service, raw_top.confidence

    used_checks = {h.id: set() for h in hypotheses}
    num_queries = 0
    max_queries = max(config.MAX_EVIDENCE_QUERIES, len(hypotheses) * config.MIN_CHECKS_PER_HYPOTHESIS)

    while num_queries < max_queries:
        selection = _select_next_check(hypotheses, used_checks)
        if selection is None:
            break
        hyp, check_name = selection
        result = CHECK_REGISTRY[check_name](hyp, incident)
        used_checks[hyp.id].add(check_name)
        num_queries += 1
        _apply_evidence(hyp, result)
        _normalize(hypotheses)

    final_top = max(hypotheses, key=lambda h: h.confidence)
    return raw_guess, raw_confidence, final_top.service, final_top.confidence, num_queries


def print_risk_coverage(records, guess_key, conf_key, label):
    """Adapts records (which use raw_/final_ prefixed keys) into the shape
    eval.metrics.risk_coverage_curve expects, and prints the table."""
    adapted = [
        {"decision": "diagnosed", "root_cause": r[guess_key], "confidence": r[conf_key],
         "ground_truth": r["ground_truth"]}
        for r in records
    ]
    acc = metrics.accuracy_on_answered(adapted)
    brier = metrics.brier_score(adapted)
    print(f"\n=== {label}: accuracy={acc:.3f}, Brier={brier:.3f} ===")
    print("  threshold | coverage | risk")
    curve = metrics.risk_coverage_curve(adapted)
    for point in curve:
        print(f"    {point['threshold']:.2f}    |   {point['coverage']:.2f}   |  {point['risk']:.3f}")
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["re1", "re2"], default="re1")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--provider", choices=["groq", "anthropic"], default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dataset == "re2":
        loader_fn = load_re2_case
        main_metrics_filename = "simple_metrics.csv"
        data_root = args.data_root or "data/RE2/RE2-OB"
    else:
        loader_fn = load_rcaeval_case
        main_metrics_filename = "data.csv"
        data_root = args.data_root or "data/RE1/RE1-OB"

    cases = find_cases(data_root, main_metrics_filename, args.limit)
    print(f"Found {len(cases)} cases under {data_root} (dataset={args.dataset})")
    if not cases:
        print("No cases found.")
        return

    records = []
    for i, case_path in enumerate(cases):
        try:
            incident = loader_fn(case_path)
        except Exception as e:
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: failed to load ({e})")
            continue

        try:
            raw_guess, raw_conf, final_guess, final_conf, n_queries = run_case(
                incident, live=args.live, provider=args.provider
            )
        except Exception as e:
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: run_case failed ({e})")
            continue

        records.append({
            "incident_id": incident["incident_id"],
            "ground_truth": incident["ground_truth"],
            "raw_guess": raw_guess,
            "raw_confidence": raw_conf,
            "final_guess": final_guess,
            "final_confidence": final_conf,
            "num_queries": n_queries,
        })

        raw_status = "OK" if raw_guess == incident["ground_truth"] else "WRONG"
        final_status = "OK" if final_guess == incident["ground_truth"] else "WRONG"
        print(f"  [{i + 1}/{len(cases)}] {incident['incident_id']}: "
              f"RAW={raw_guess}({raw_conf:.2f})[{raw_status}] -> "
              f"FINAL={final_guess}({final_conf:.2f})[{final_status}]")

        if args.live and args.delay > 0:
            time.sleep(args.delay)

    if not records:
        print("No cases completed.")
        return

    results_csv = os.path.join(args.out, "ablation_results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records to {results_csv}")

    print_risk_coverage(records, "raw_guess", "raw_confidence", "RAW (unverified LLM confidence)")
    print_risk_coverage(records, "final_guess", "final_confidence", "FINAL (after evidence verification)")

    flips = sum(1 for r in records if r["raw_guess"] != r["final_guess"])
    print(f"\nVerification CHANGED the guess in {flips}/{len(records)} cases "
          f"({flips / len(records) * 100:.1f}%).")


if __name__ == "__main__":
    main()