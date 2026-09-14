"""
Evidence-check ablation: runs SYNTHESIS with one specific check type
REMOVED from the evidence-gathering loop entirely, to causally test
whether that check actually matters -- rather than just correlating
whether the LLM's reasoning happens to mention it (which is what
analyze_check_attribution.py does; this script is the controlled
follow-up that tests it directly).

Compare this script's output against your EXISTING full-checks SYNTHESIS
results (results/synthesis_results_re2.csv) -- there's no need to re-run
the full-checks condition, since you already have it.

Usage:
    python experiments/run_evidence_ablation.py --dataset re2 --live --provider anthropic --exclude-check threshold
    python experiments/run_evidence_ablation.py --dataset re2 --live --provider anthropic --exclude-check topology_consistency
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
from src.llm_client import generate_hypotheses, synthesize_final_diagnosis
from src.pipeline import _apply_evidence, _normalize
from src.evidence import CHECK_REGISTRY


def select_next_check_filtered(hypotheses, used_checks, allowed_checks):
    """
    Same round-robin logic as src.pipeline._select_next_check (fewest
    checks first, ties broken by confidence), but restricted to
    `allowed_checks` only -- a local copy rather than modifying the shared
    pipeline function, since this filtering is specific to this ablation
    script and shouldn't risk changing behavior anywhere else.
    """
    candidates = []
    for hyp in hypotheses:
        tried = used_checks.get(hyp.id, set())
        remaining = [c for c in allowed_checks if c not in tried]
        if remaining:
            candidates.append((len(tried), -hyp.confidence, hyp, remaining[0]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, _, hyp, check_name = candidates[0]
    return hyp, check_name


def find_cases(root, main_metrics_filename, limit=None):
    cases = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if main_metrics_filename in filenames:
            cases.append(dirpath)
    cases.sort()
    if limit:
        cases = cases[:limit]
    return cases


def run_case(incident, provider, allowed_checks):
    hypotheses = generate_hypotheses(incident, live=True, provider=provider)
    _normalize(hypotheses)

    raw_top = max(hypotheses, key=lambda h: h.confidence)
    raw_guess = raw_top.service

    used_checks = {h.id: set() for h in hypotheses}
    num_queries = 0
    max_queries = max(config.MAX_EVIDENCE_QUERIES, len(hypotheses) * config.MIN_CHECKS_PER_HYPOTHESIS)

    evidence_trail = []
    while num_queries < max_queries:
        selection = select_next_check_filtered(hypotheses, used_checks, allowed_checks)
        if selection is None:
            break
        hyp, check_name = selection
        result = CHECK_REGISTRY[check_name](hyp, incident)
        used_checks[hyp.id].add(check_name)
        num_queries += 1
        evidence_trail.append({
            "check": result.check_name, "service": hyp.service,
            "verdict": result.verdict, "reason": result.reason,
        })

    synthesis = synthesize_final_diagnosis(incident, hypotheses, evidence_trail, provider=provider)

    return {
        "raw_guess": raw_guess,
        "synthesis_guess": synthesis["service"],
        "synthesis_confidence": synthesis["confidence"],
        "num_queries": num_queries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["re1", "re2"], default="re1")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--provider", choices=["groq", "anthropic"], default=None)
    parser.add_argument("--exclude-check", required=True, choices=list(CHECK_REGISTRY.keys()),
                         help="which check type to REMOVE from the evidence loop entirely")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    allowed_checks = [c for c in CHECK_REGISTRY if c != args.exclude_check]
    print(f"Running SYNTHESIS with '{args.exclude_check}' REMOVED. "
          f"Remaining checks: {allowed_checks}")

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
            result = run_case(incident, provider=args.provider, allowed_checks=allowed_checks)
        except Exception as e:
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: run_case failed ({e})")
            continue

        result["incident_id"] = incident["incident_id"]
        result["ground_truth"] = incident["ground_truth"]
        result["excluded_check"] = args.exclude_check
        records.append(result)

        gt = incident["ground_truth"]
        status = "OK" if result["synthesis_guess"] == gt else "WRONG"
        print(f"  [{i + 1}/{len(cases)}] {incident['incident_id']}: "
              f"SYNTH(no {args.exclude_check})={result['synthesis_guess']}"
              f"({result['synthesis_confidence']:.2f}) [{status}]")

        if args.delay > 0:
            time.sleep(args.delay)

    if not records:
        print("No cases completed.")
        return

    results_csv = os.path.join(args.out, f"ablation_no_{args.exclude_check}_{args.dataset}.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records to {results_csv}")

    accuracy = sum(1 for r in records if r["synthesis_guess"] == r["ground_truth"]) / len(records)
    print(f"\nSYNTHESIS accuracy WITHOUT '{args.exclude_check}': {accuracy:.3f}")
    print(f"Compare this against your existing full-checks SYNTHESIS accuracy "
          f"in results/synthesis_results_{args.dataset}.csv")


if __name__ == "__main__":
    main()