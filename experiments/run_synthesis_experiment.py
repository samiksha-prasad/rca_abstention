"""
Three-way comparison, on the same cases:
  1. RAW:       the LLM's own guess, no evidence checks at all
  2. ARITHMETIC: the existing pipeline (fixed +0.30/-0.35 formula overrides
                 the LLM's guess based on rule-based evidence) -- found to
                 be net-harmful (see ablation results)
  3. SYNTHESIS: same evidence checks run, but instead of a formula
                 overriding the LLM, the evidence is handed back to the
                 LLM as text and IT decides how much to trust each piece,
                 alongside its own original reasoning

This tests whether letting the model interpret evidence (rather than a
rigid rule silently overruling it) preserves more of its original good
judgment while still using evidence when it's genuinely strong.

NOTE: this makes TWO LLM calls per case (one for the initial hypotheses,
one for synthesis) -- roughly double the API usage of run_ablation.py.

Usage:
    python experiments/run_synthesis_experiment.py --dataset re1 --limit 20 --live
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


def run_case(incident, provider):
    """Runs all three approaches on the same incident, reusing the same
    initial hypotheses and evidence-check results for a fair comparison."""
    hypotheses = generate_hypotheses(incident, live=True, provider=provider)
    _normalize(hypotheses)

    raw_top = max(hypotheses, key=lambda h: h.confidence)
    raw_guess, raw_confidence = raw_top.service, raw_top.confidence

    # Run the SAME evidence checks once, recording the full trail, and
    # apply the arithmetic update as the existing pipeline does (for the
    # ARITHMETIC comparison arm) -- on a COPY of confidences so we can
    # still hand the ORIGINAL raw hypotheses to the synthesis stage.
    import copy
    arithmetic_hyps = copy.deepcopy(hypotheses)
    used_checks = {h.id: set() for h in arithmetic_hyps}
    num_queries = 0
    max_queries = max(config.MAX_EVIDENCE_QUERIES, len(arithmetic_hyps) * config.MIN_CHECKS_PER_HYPOTHESIS)

    evidence_trail = []  # shared trail, used by both arithmetic and synthesis arms
    while num_queries < max_queries:
        selection = _select_next_check(arithmetic_hyps, used_checks)
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
        _apply_evidence(hyp, result)
        _normalize(arithmetic_hyps)

    arithmetic_top = max(arithmetic_hyps, key=lambda h: h.confidence)
    arithmetic_guess, arithmetic_confidence = arithmetic_top.service, arithmetic_top.confidence

    # EVIDENCE CONSENSUS: a signal independent of any stated confidence --
    # simply, how many of the checks that ran against the RAW GUESS
    # specifically came back "support" vs "contradict"? This tests whether
    # raw agreement/disagreement among checks predicts correctness better
    # than the model's own stated confidence (RAW or SYNTHESIS).
    raw_guess_supports = sum(
        1 for step in evidence_trail if step["service"] == raw_guess and step["verdict"] == "support"
    )
    raw_guess_contradicts = sum(
        1 for step in evidence_trail if step["service"] == raw_guess and step["verdict"] == "contradict"
    )
    raw_guess_inconclusive = sum(
        1 for step in evidence_trail if step["service"] == raw_guess and step["verdict"] == "inconclusive"
    )
    evidence_consensus_net = raw_guess_supports - raw_guess_contradicts

    # SYNTHESIS: hand the SAME evidence trail to the LLM, on the ORIGINAL
    # (unmodified) hypotheses, and let it decide the final answer itself.
    synthesis = synthesize_final_diagnosis(incident, hypotheses, evidence_trail, provider=provider)

    return {
        "raw_guess": raw_guess, "raw_confidence": raw_confidence,
        "arithmetic_guess": arithmetic_guess, "arithmetic_confidence": arithmetic_confidence,
        "synthesis_guess": synthesis["service"], "synthesis_confidence": synthesis["confidence"],
        "synthesis_reasoning": synthesis["reasoning"],
        "raw_guess_supports": raw_guess_supports,
        "raw_guess_contradicts": raw_guess_contradicts,
        "raw_guess_inconclusive": raw_guess_inconclusive,
        "evidence_consensus_net": evidence_consensus_net,
        "num_queries": num_queries,
    }


def print_arm(records, guess_key, conf_key, label):
    adapted = [
        {"decision": "diagnosed", "root_cause": r[guess_key], "confidence": r[conf_key],
         "ground_truth": r["ground_truth"]}
        for r in records
    ]
    acc = metrics.accuracy_on_answered(adapted)
    brier = metrics.brier_score(adapted)
    print(f"\n=== {label}: accuracy={acc:.3f}, Brier={brier:.3f} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["re1", "re2"], default="re1")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--live", action="store_true", required=False,
                         help="this experiment REQUIRES --live (synthesis has no mock mode); kept as a flag for clarity/consistency with other scripts")
    parser.add_argument("--provider", choices=["groq", "anthropic"], default=None)
    parser.add_argument("--delay", type=float, default=1.5,
                         help="longer default delay than run_ablation.py since this makes 2 LLM calls per case")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if not args.live:
        print("This experiment requires --live (the synthesis stage needs a real LLM). Exiting.")
        return

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
            result = run_case(incident, provider=args.provider)
        except Exception as e:
            print(f"  [{i + 1}/{len(cases)}] SKIP {case_path}: run_case failed ({e})")
            continue

        result["incident_id"] = incident["incident_id"]
        result["ground_truth"] = incident["ground_truth"]
        records.append(result)

        gt = incident["ground_truth"]
        print(f"  [{i + 1}/{len(cases)}] {incident['incident_id']}: "
              f"RAW={result['raw_guess']}({result['raw_confidence']:.2f})"
              f"[{'OK' if result['raw_guess']==gt else 'WRONG'}] | "
              f"ARITH={result['arithmetic_guess']}({result['arithmetic_confidence']:.2f})"
              f"[{'OK' if result['arithmetic_guess']==gt else 'WRONG'}] | "
              f"SYNTH={result['synthesis_guess']}({result['synthesis_confidence']:.2f})"
              f"[{'OK' if result['synthesis_guess']==gt else 'WRONG'}]")

        if args.delay > 0:
            time.sleep(args.delay)

    if not records:
        print("No cases completed.")
        return

    results_csv = os.path.join(args.out, f"synthesis_results_{args.dataset}.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records to {results_csv}")

    print_arm(records, "raw_guess", "raw_confidence", "RAW (LLM alone)")
    print_arm(records, "arithmetic_guess", "arithmetic_confidence", "ARITHMETIC (formula overrides LLM)")
    print_arm(records, "synthesis_guess", "synthesis_confidence", "SYNTHESIS (LLM interprets evidence itself)")


if __name__ == "__main__":
    main()