"""
Runs the diagnosis pipeline over an entire RCAEval sub-dataset (e.g. all 125
Online Boutique RE1 cases), saves per-case results, and computes the real
evaluation metrics -- most importantly the risk-coverage curve, which is
the core result for the whole project.

Usage:
    python experiments/run_experiment.py                       # all cases, mock LLM
    python experiments/run_experiment.py --limit 20             # quick test on 20 cases
    python experiments/run_experiment.py --live                 # real LLM calls (costs money)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # loads GROQ_API_KEY / ANTHROPIC_API_KEY from a local .env file, if present

from src.data_loader import load_rcaeval_case, load_re2_case
from src.pipeline import diagnose
from eval import metrics, plots


def find_cases(root, main_metrics_filename, limit=None):
    """Find every case folder (a leaf directory containing the dataset's
    main metrics file) under root. RE1 uses 'data.csv', RE2 uses
    'simple_metrics.csv' -- discovered by inspecting real downloads."""
    cases = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if main_metrics_filename in filenames:
            cases.append(dirpath)
    cases.sort()
    if limit:
        cases = cases[:limit]
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["re1", "re2"], default="re1",
                         help="which RCAEval sub-dataset to run on (re1: metrics only, "
                              "re2: metrics + real logs)")
    parser.add_argument("--data-root", default=None,
                         help="folder to search for cases (default depends on --dataset: "
                              "data/RE1/RE1-OB for re1, data/RE2/RE2-OB for re2)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only run the first N cases found (useful for a quick smoke test)")
    parser.add_argument("--live", action="store_true",
                         help="use the real LLM instead of the mock hypothesis generator")
    parser.add_argument("--provider", choices=["groq", "anthropic"], default=None,
                         help="override config.LLM_PROVIDER for this run (e.g. switch to anthropic "
                              "if Groq's free-tier daily quota is exhausted)")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="seconds to wait between cases when --live (avoids free-tier rate limits; ignored in mock mode)")
    parser.add_argument("--out", default="results",
                         help="folder to write results.csv and plots into")
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
        print("No cases found -- did you download the dataset? See DOWNLOAD_RCAEVAL.md")
        return

    records = []
    topology = None

    for i, case_path in enumerate(cases):
        try:
            incident = loader_fn(case_path)
        except Exception as e:
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: failed to load ({e})")
            continue

        if topology is None:
            topology = incident["topology"]

        # confidence_threshold=0.0 so the loop runs to completion (checking
        # evidence until the budget or available checks are exhausted) and
        # reports the RAW top confidence, instead of abstaining early. We
        # need every case's true confidence value to build the risk-coverage
        # curve afterward, by sweeping OUR OWN thresholds over this data.
        try:
            result = diagnose(incident, live=args.live, confidence_threshold=0.0, provider=args.provider)
        except Exception as e:
            # One case failing (e.g. the LLM call ran out of retries) should
            # not lose all the other results from a long --live run --
            # found via testing: a real run crashed entirely on case 3/125
            # with no way to recover the first 2 cases' results.
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: diagnose() failed after retries ({e})")
            continue

        records.append({
            "incident_id": incident["incident_id"],
            "decision": result.decision,
            "root_cause": result.root_cause,
            "confidence": result.confidence,
            "ground_truth": incident["ground_truth"],
            "num_queries": result.num_queries,
        })

        status = "OK" if result.root_cause == incident["ground_truth"] else "WRONG"
        print(f"  [{i + 1}/{len(cases)}] {incident['incident_id']}: "
              f"guessed={result.root_cause} truth={incident['ground_truth']} "
              f"conf={result.confidence:.2f} [{status}]")

        if args.live and args.delay > 0:
            time.sleep(args.delay)

    if not records:
        print("No cases loaded successfully -- nothing to evaluate.")
        return

    # --- Save raw per-case results ---
    results_csv = os.path.join(args.out, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records to {results_csv}")

    # --- Summary metrics ---
    print("\n=== Summary metrics (raw, threshold=0.0 i.e. 'always answer') ===")
    print(f"Accuracy (top-1, all cases answered): {metrics.accuracy_on_answered(records):.3f}")
    print(f"Brier score (lower is better): {metrics.brier_score(records):.3f}")
    print(f"Cost: {metrics.cost_summary(records)}")
    if topology:
        rate = metrics.symptom_as_root_cause_rate(records, topology)
        print(f"Symptom-as-root-cause rate (among wrong answers): {rate:.3f}")

    # --- Risk-coverage curve: THE core result ---
    curve = metrics.risk_coverage_curve(records)
    print("\n=== Risk-coverage curve (sweeping a confidence threshold) ===")
    print("  threshold | coverage (answered) | risk (error rate among answered)")
    for point in curve:
        print(f"    {point['threshold']:.2f}    |        {point['coverage']:.2f}         |        {point['risk']:.3f}")

    plots.plot_risk_coverage(curve, save_path=os.path.join(args.out, "risk_coverage.png"))
    plots.plot_reliability_diagram(records, save_path=os.path.join(args.out, "reliability.png"))
    print(f"\nPlots saved to {args.out}/risk_coverage.png and {args.out}/reliability.png")


if __name__ == "__main__":
    main()