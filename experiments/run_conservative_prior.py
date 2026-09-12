"""
A/B/C test for the "conservative prior" idea: does requiring 2+ contradicting
checks (instead of just 1) before docking a hypothesis's confidence reduce
BAD flips (verification breaking an already-correct LLM guess) without also
suppressing GOOD flips (verification catching a genuinely wrong guess)?

Three conditions, all derived from the SAME generated hypotheses per case
(so differences are attributable to the confidence-update rule alone, not
to re-sampling the LLM or re-drawing evidence checks):

  - baseline:   contradictions_to_overturn=1 for every hypothesis (today's
                rule -- one contradiction has the same say as one support).
  - uniform:    contradictions_to_overturn=N for every hypothesis. TRIED
                FIRST, and it made things WORSE in the medium/high raw-
                confidence buckets, not better. Kept here for comparison.
  - targeted:   contradictions_to_overturn=N ONLY for the hypothesis that
                was the LLM's own raw top pick before any evidence ran;
                every OTHER (challenger) hypothesis still uses the normal
                rule (1 contradiction). This is what "treat the LLM's guess
                as a strong prior" actually means -- the uniform condition
                instead gave every challenger hypothesis the same immunity,
                which let a wrong challenger survive contradictions it
                should have been knocked out by and win by default. That's
                the leading explanation for why uniform hurt medium/high-
                confidence cases instead of protecting them.

Usage:
    python experiments/run_conservative_prior.py --dataset re1 --limit 30
    python experiments/run_conservative_prior.py --dataset re1 --limit 30 --live
    python experiments/run_conservative_prior.py --dataset re1 --overturn-at 3
"""
import argparse
import copy
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


def _run_loop(hypotheses, incident, contradictions_to_overturn=1,
              protected_id=None, protected_contradictions_to_overturn=None):
    """
    Same loop diagnose() runs, parameterized by the confidence-update rule
    under test. Mutates `hypotheses` in place (caller passes a copy).

    contradictions_to_overturn is the rule used for every hypothesis EXCEPT
    protected_id, which (if given) uses protected_contradictions_to_overturn
    instead. Passing protected_id=None reduces to the uniform rule -- this
    is what makes the same function serve baseline, uniform, and targeted.
    """
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

        if protected_id is not None and hyp.id == protected_id:
            threshold = protected_contradictions_to_overturn
        else:
            threshold = contradictions_to_overturn
        _apply_evidence(hyp, result, contradictions_to_overturn=threshold)
        _normalize(hypotheses)

    top = max(hypotheses, key=lambda h: h.confidence)
    return top.service, top.confidence, num_queries


def run_case(incident, live, provider, overturn_at):
    """
    Generates hypotheses ONCE, then runs three rules on independent deep
    copies of the same starting hypotheses, so all three start from the
    identical raw guess/confidence and only the confidence-update rule
    differs:
      - baseline:   contradictions_to_overturn=1 for everyone.
      - uniform:    contradictions_to_overturn=overturn_at for everyone.
      - targeted:   contradictions_to_overturn=overturn_at ONLY for the
                    hypothesis that was the raw top pick; everyone else
                    stays at 1.
    """
    hypotheses = generate_hypotheses(incident, live=live, provider=provider)
    _normalize(hypotheses)

    raw_top = max(hypotheses, key=lambda h: h.confidence)
    raw_guess, raw_confidence = raw_top.service, raw_top.confidence

    baseline_guess, baseline_conf, n_queries = _run_loop(
        copy.deepcopy(hypotheses), incident, contradictions_to_overturn=1
    )
    uniform_guess, uniform_conf, _ = _run_loop(
        copy.deepcopy(hypotheses), incident, contradictions_to_overturn=overturn_at
    )
    targeted_guess, targeted_conf, _ = _run_loop(
        copy.deepcopy(hypotheses), incident,
        contradictions_to_overturn=1,
        protected_id=raw_top.id,
        protected_contradictions_to_overturn=overturn_at,
    )

    return {
        "raw_guess": raw_guess,
        "raw_confidence": raw_confidence,
        "baseline_guess": baseline_guess,
        "baseline_confidence": baseline_conf,
        "uniform_guess": uniform_guess,
        "uniform_confidence": uniform_conf,
        "targeted_guess": targeted_guess,
        "targeted_confidence": targeted_conf,
        "num_queries": n_queries,
    }


def print_risk_coverage(records, guess_key, conf_key, label):
    adapted = [
        {"decision": "diagnosed", "root_cause": r[guess_key], "confidence": r[conf_key],
         "ground_truth": r["ground_truth"]}
        for r in records
    ]
    acc = metrics.accuracy_on_answered(adapted)
    brier = metrics.brier_score(adapted)
    print(f"\n=== {label}: accuracy={acc:.3f}, Brier={brier:.3f} ===")
    print("  threshold | coverage | risk")
    for point in metrics.risk_coverage_curve(adapted):
        print(f"    {point['threshold']:.2f}    |   {point['coverage']:.2f}   |  {point['risk']:.3f}")


def print_flip_comparison(records, variant_key, variant_label):
    """Same good/bad/neutral-flip framing as analyze_flips.py, but comparing
    a variant rule's final guess against the BASELINE rule's final guess
    (not raw vs. final within one rule) -- this isolates exactly what
    switching to that rule bought or cost, bucketed by how confident the
    raw LLM guess already was."""
    buckets = {
        "low (<0.35)": lambda c: c < 0.35,
        "medium (0.35-0.50)": lambda c: 0.35 <= c < 0.50,
        "high (>=0.50)": lambda c: c >= 0.50,
    }
    variant_guess_key = f"{variant_key}_guess"

    print(f"\n=== {variant_label} vs. baseline (overturn_at=1), bucketed by RAW LLM confidence ===")
    print(f"{'Bucket':<20}{'N':<6}{'Changed':<10}{'Good flip':<12}{'Bad flip':<11}{'Neutral':<10}"
          f"{'Baseline acc':<14}{variant_label + ' acc':<20}")
    print("-" * 120)

    for label, in_bucket in buckets.items():
        subset = [r for r in records if in_bucket(r["raw_confidence"])]
        if not subset:
            print(f"{label:<20}{'0':<6}(no cases)")
            continue

        n = len(subset)
        changed = sum(1 for r in subset if r["baseline_guess"] != r[variant_guess_key])
        good_flip = sum(
            1 for r in subset
            if r["baseline_guess"] != r[variant_guess_key]
            and r["baseline_guess"] != r["ground_truth"]
            and r[variant_guess_key] == r["ground_truth"]
        )
        bad_flip = sum(
            1 for r in subset
            if r["baseline_guess"] != r[variant_guess_key]
            and r["baseline_guess"] == r["ground_truth"]
            and r[variant_guess_key] != r["ground_truth"]
        )
        neutral_flip = changed - good_flip - bad_flip
        baseline_acc = sum(1 for r in subset if r["baseline_guess"] == r["ground_truth"]) / n
        variant_acc = sum(1 for r in subset if r[variant_guess_key] == r["ground_truth"]) / n

        print(f"{label:<20}{n:<6}{changed:<10}{good_flip:<12}{bad_flip:<11}{neutral_flip:<10}"
              f"{baseline_acc:<14.3f}{variant_acc:<20.3f}")

    print("\nInterpretation:")
    print(f"  Good flip = {variant_label} rule fixed a case baseline got wrong.")
    print(f"  Bad flip  = {variant_label} rule broke a case baseline got right.")
    print("  If bad flips concentrate in 'high' and good flips in 'low'/'medium', that")
    print("  supports the hypothesis: protecting the raw top guess shields strong")
    print("  guesses from single noisy checks while still letting weak guesses be")
    print("  overturned by sustained (2+) disagreement.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["re1", "re2"], default="re1")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--provider", choices=["groq", "anthropic"], default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--overturn-at", type=int, default=2,
                         help="contradictions_to_overturn for the conservative condition "
                              "(baseline is always 1, today's default). Default: 2.")
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
            result = run_case(incident, live=args.live, provider=args.provider, overturn_at=args.overturn_at)
        except Exception as e:
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: run_case failed ({e})")
            continue

        result["incident_id"] = incident["incident_id"]
        result["ground_truth"] = incident["ground_truth"]
        records.append(result)

        baseline_status = "OK" if result["baseline_guess"] == incident["ground_truth"] else "WRONG"
        uniform_status = "OK" if result["uniform_guess"] == incident["ground_truth"] else "WRONG"
        targeted_status = "OK" if result["targeted_guess"] == incident["ground_truth"] else "WRONG"
        print(f"  [{i + 1}/{len(cases)}] {incident['incident_id']}: "
              f"BASELINE={result['baseline_guess']}({result['baseline_confidence']:.2f})[{baseline_status}] -> "
              f"UNIFORM={result['uniform_guess']}({result['uniform_confidence']:.2f})[{uniform_status}] -> "
              f"TARGETED={result['targeted_guess']}({result['targeted_confidence']:.2f})[{targeted_status}]")

        if args.live and args.delay > 0:
            time.sleep(args.delay)

    if not records:
        print("No cases completed.")
        return

    results_csv = os.path.join(args.out, "conservative_prior_results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records to {results_csv}")

    print_risk_coverage(records, "baseline_guess", "baseline_confidence",
                         "BASELINE (contradictions_to_overturn=1, today's default)")
    print_risk_coverage(records, "uniform_guess", "uniform_confidence",
                         f"UNIFORM (contradictions_to_overturn={args.overturn_at} for all hypotheses)")
    print_risk_coverage(records, "targeted_guess", "targeted_confidence",
                         f"TARGETED (contradictions_to_overturn={args.overturn_at} only for the raw top pick)")
    print_flip_comparison(records, "uniform", "uniform")
    print_flip_comparison(records, "targeted", "targeted")


if __name__ == "__main__":
    main()
