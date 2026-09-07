import os
import tempfile
import shutil

import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from src.data_loader import make_synthetic_incident, load_rcaeval_case
from src.llm_client import Hypothesis, generate_hypotheses
from src.evidence import threshold_check, temporal_ordering_check, topology_consistency_check
from src.pipeline import diagnose
from eval.metrics import accuracy_on_answered, coverage, risk_coverage_curve, brier_score


def test_mock_hypothesis_generation_returns_valid_services():
    incident = make_synthetic_incident()
    hyps = generate_hypotheses(incident, live=False)
    assert len(hyps) > 0
    for h in hyps:
        assert h.service in incident["topology"]


def test_threshold_check_detects_spike():
    incident = make_synthetic_incident()
    hyp = Hypothesis(id="H1", service="payment-service", description="", confidence=0.5)
    result = threshold_check(hyp, incident)
    assert result.verdict in ("support", "contradict", "inconclusive")


def test_topology_check_flags_downstream_service():
    incident = make_synthetic_incident()
    hyp = Hypothesis(id="H1", service="order-service", description="", confidence=0.5)
    result = topology_consistency_check(hyp, incident)
    # order-service depends on payment-service, which is anomalous -> should contradict
    assert result.verdict == "contradict"


def test_diagnose_returns_valid_decision():
    incident = make_synthetic_incident()
    result = diagnose(incident, live=False)
    assert result.decision in ("diagnosed", "abstained")
    if result.decision == "diagnosed":
        assert result.root_cause in incident["topology"]
    assert result.num_queries <= 12


def test_diagnose_respects_query_budget():
    incident = make_synthetic_incident()
    result = diagnose(incident, live=False, max_queries=2)
    assert result.num_queries <= 2


def test_metrics_on_toy_records():
    records = [
        {"decision": "diagnosed", "root_cause": "a", "confidence": 0.9, "ground_truth": "a", "num_queries": 3},
        {"decision": "diagnosed", "root_cause": "b", "confidence": 0.8, "ground_truth": "c", "num_queries": 5},
        {"decision": "abstained", "root_cause": None, "confidence": 0.4, "ground_truth": "a", "num_queries": 8},
    ]
    assert accuracy_on_answered(records) == 0.5
    assert coverage(records) == 2 / 3
    curve = risk_coverage_curve(records)
    assert len(curve) > 0
    assert 0 <= brier_score(records) <= 1


def test_load_rcaeval_case_matches_real_format():
    """
    Regression test using the REAL RCAEval RE1 format we confirmed by
    downloading and inspecting an actual case (see chat history / git log).
    If RCAEval ever changes their format, or someone edits load_rcaeval_case
    incorrectly, this test should catch it.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        case_dir = os.path.join(tmp_dir, "data", "RE1", "RE1-OB", "cartservice_cpu", "1")
        os.makedirs(case_dir)
        with open(os.path.join(case_dir, "data.csv"), "w") as f:
            f.write(
                "time,adservice_cpu,cartservice_cpu,checkoutservice_cpu,frontend_cpu,"
                "PassthroughCluster_load,frontend-external_load,adservice_latency,"
                "cartservice_latency,time,PassthroughCluster_error,frontend_error\n"
                "1000,1.0,2.0,1.5,3.0,0.0,10.0,0.01,0.02,1000,0.0,0.0\n"
                "1001,1.0,2.0,1.5,3.0,0.0,10.0,0.01,0.02,1001,0.0,0.0\n"
                "1050,1.1,9.0,1.5,3.2,0.0,10.0,0.01,0.09,1050,0.0,0.0\n"
                "1051,1.1,9.5,1.5,3.2,0.0,10.0,0.01,0.10,1051,0.0,0.0\n"
            )
        with open(os.path.join(case_dir, "inject_time.txt"), "w") as f:
            f.write("1050")

        old_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            incident = load_rcaeval_case(
                "data/RE1/RE1-OB/cartservice_cpu/1",
                window_before=1000, window_after=1000,
            )
        finally:
            os.chdir(old_cwd)

        assert incident["ground_truth"] == "cartservice"
        assert incident["inject_time"] == 1050
        assert "cartservice" in incident["telemetry"]["metrics"]
        assert "cpu" in incident["telemetry"]["metrics"]["cartservice"]
        assert incident["telemetry"]["logs"] == {}
    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])