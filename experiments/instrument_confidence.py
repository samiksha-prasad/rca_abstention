"""
Instruments the exact confidence-update mechanics: for a given incident,
runs the real hypothesis-generation + evidence-verification loop (using the
ACTUAL functions from src/, not a reimplementation), and prints a full
audit trail showing, for every single evidence check applied to every
hypothesis:
    - which check ran, on which hypothesis
    - the verdict (support/contradict/inconclusive)
    - the raw confidence BEFORE the delta
    - the raw confidence AFTER the delta (before renormalization)
    - the confidence AFTER renormalization across all hypotheses

This exists to answer one specific question: is the ~0.5 confidence
ceiling seen on real data a property of the normalization math itself, or
a consequence of competing hypotheses also picking up support/contradict
signals? No API calls needed -- runs entirely in mock mode.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.llm_client import generate_hypotheses
from src.evidence import CHECK_REGISTRY
from src.pipeline import _select_next_check, _apply_evidence, _normalize


def instrumented_diagnose(incident, live=False, max_queries=None):
    hypotheses = generate_hypotheses(incident, live=live)
    _normalize(hypotheses)

    # Match pipeline.diagnose()'s dynamic budget scaling exactly -- this
    # script has its own standalone loop for instrumentation purposes, so
    # it needs to be kept in sync manually. (Found out the hard way: an
    # earlier version of this script used the OLD fixed budget and
    # produced a shorter, non-matching trace vs. the real pipeline.)
    if max_queries is None:
        max_queries = max(config.MAX_EVIDENCE_QUERIES, len(hypotheses) * config.MIN_CHECKS_PER_HYPOTHESIS)

    print(f"\n{'='*100}")
    print(f"Incident: {incident.get('incident_id', 'unknown')}  |  Ground truth: {incident['ground_truth']}")
    print(f"{'='*100}")
    print(f"\nInitial (after first normalize):")
    for h in hypotheses:
        marker = " <-- TRUE CAUSE" if h.service == incident["ground_truth"] else ""
        print(f"  {h.id} ({h.service}): {h.confidence:.4f}{marker}")

    used_checks = {h.id: set() for h in hypotheses}
    num_queries = 0

    print(f"\n{'Step':<5}{'Hyp':<5}{'Service':<25}{'Check':<22}{'Verdict':<13}{'Before':<9}{'RawDelta':<10}{'AfterNorm':<10}")
    print("-" * 100)

    while num_queries < max_queries:
        selection = _select_next_check(hypotheses, used_checks)
        if selection is None:
            break
        hyp, check_name = selection
        check_fn = CHECK_REGISTRY[check_name]
        result = check_fn(hyp, incident)

        used_checks[hyp.id].add(check_name)
        num_queries += 1

        before = hyp.confidence
        _apply_evidence(hyp, result)
        raw_after = hyp.confidence  # after delta, before renormalize
        _normalize(hypotheses)
        after_norm = hyp.confidence  # after renormalize

        marker = "*" if hyp.service == incident["ground_truth"] else " "
        print(f"{num_queries:<5}{hyp.id:<5}{hyp.service:<25}{check_name:<22}{result.verdict:<13}"
              f"{before:<9.4f}{raw_after:<10.4f}{after_norm:<10.4f}{marker}")

    print(f"\nFinal confidences:")
    for h in sorted(hypotheses, key=lambda x: -x.confidence):
        marker = " <-- TRUE CAUSE" if h.service == incident["ground_truth"] else ""
        print(f"  {h.id} ({h.service}): {h.confidence:.4f}{marker}")

    total = sum(h.confidence for h in hypotheses)
    print(f"\nSum of all final confidences: {total:.6f}  (should be 1.0 -- confirms it IS a valid "
          f"probability distribution over hypotheses BY CONSTRUCTION; whether it's a CALIBRATED one "
          f"is a separate, empirical question tested by the risk-coverage curve, not this script)")

    return hypotheses


def make_messy_realistic_incident(seed=7):
    """
    Builds an incident with the SAME shape as load_rcaeval_case's output,
    but constructed directly in Python (no files needed) so we can control
    exactly how messy it is. Unlike the synthetic toy example (4 clean
    hypotheses, one obvious signal), this mimics what real RCAEval data
    actually looks like: ~10 competing services, most of them showing SOME
    bursty noise (like real 'load' metrics do) even though only one is the
    real fault -- this is the condition that was causing confusion on real
    data, so it's the right stress test for the confidence mechanics.
    """
    import random
    from src.topology import ONLINE_BOUTIQUE_TOPOLOGY

    random.seed(seed)
    services = list(ONLINE_BOUTIQUE_TOPOLOGY.keys())
    true_cause = "checkoutservice"
    inject_time = 1050

    metrics = {}
    for svc in services:
        cpu_series = []
        load_series = []
        for t in range(1000, 1100):
            cpu_base = 2.0
            cpu_noise = random.uniform(-0.4, 0.4)
            cpu_spike = 9.0 if (svc == true_cause and t >= inject_time) else 0.0
            cpu_series.append((t, cpu_base + cpu_spike + cpu_noise))

            # Bursty load noise for EVERYONE, pre and post -- like real traffic
            load_base = 50.0
            burst = random.choice([0, 0, 0, 35, -25])
            load_series.append((t, load_base + burst))
        metrics[svc] = {"cpu": cpu_series, "load": load_series}

    return {
        "incident_id": "messy-realistic-001",
        "alert_text": f"Fault detected (type=cpu) around service {true_cause}",
        "topology": ONLINE_BOUTIQUE_TOPOLOGY,
        "telemetry": {"metrics": metrics, "logs": {}},
        "ground_truth": true_cause,
        "inject_time": inject_time,
    }


if __name__ == "__main__":
    import argparse
    from src.data_loader import make_synthetic_incident, load_rcaeval_case, load_re2_case

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, default=None,
                         help="path to a real RE1 or RE2 case folder to instrument, "
                              "e.g. data/RE1/RE1-OB/adservice_cpu/1")
    parser.add_argument("--re2", action="store_true", help="treat --real path as an RE2 case")
    args = parser.parse_args()

    if args.real:
        loader = load_re2_case if args.re2 else load_rcaeval_case
        incident = loader(args.real)
        print(f"### REAL CASE: {args.real} ###")
        instrumented_diagnose(incident)
    else:
        print("### CASE 1: Synthetic toy example (ideal conditions, small distractor pool) ###")
        instrumented_diagnose(make_synthetic_incident())

        print("\n\n### CASE 2: Messy realistic-style data (10 competing hypotheses, bursty noise) ###")
        instrumented_diagnose(make_messy_realistic_incident())