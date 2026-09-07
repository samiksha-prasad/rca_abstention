"""
The four evaluation angles from the proposal, as functions over a list of
DiagnosisResult dicts (each must also carry the incident's ground_truth,
attached by the experiment runner).

Expected input shape per record:
{
    "decision": "diagnosed" | "abstained",
    "root_cause": str | None,
    "confidence": float,
    "ground_truth": str,
    "num_queries": int,
}
"""
import numpy as np


def accuracy_on_answered(records: list) -> float:
    """Correctness metric #1: accuracy, but ONLY over cases where the
    system chose to answer (this is the standard 'selective accuracy')."""
    answered = [r for r in records if r["decision"] == "diagnosed"]
    if not answered:
        return float("nan")
    correct = sum(1 for r in answered if r["root_cause"] == r["ground_truth"])
    return correct / len(answered)


def coverage(records: list) -> float:
    """What fraction of incidents did the system choose to answer at all?"""
    if not records:
        return float("nan")
    answered = sum(1 for r in records if r["decision"] == "diagnosed")
    return answered / len(records)


def brier_score(records: list) -> float:
    """
    Calibration metric #2: Brier score over ALL records (including
    abstained ones, treated as confidence-in-nothing = incorrect),
    lower is better.
    """
    scores = []
    for r in records:
        correct = 1.0 if (r["decision"] == "diagnosed" and r["root_cause"] == r["ground_truth"]) else 0.0
        scores.append((r["confidence"] - correct) ** 2)
    return float(np.mean(scores)) if scores else float("nan")


def risk_coverage_curve(records: list, thresholds=None) -> list:
    """
    Sweeps a confidence threshold and recomputes, at each threshold:
        - coverage: fraction of incidents answered at that threshold
        - risk: error rate among those answered

    This is the CORE result figure for the project. Note this assumes
    each record stores the *raw* top-hypothesis confidence regardless of
    whether the pipeline's own internal threshold already caused an
    abstain -- so re-run diagnosis with confidence_threshold=0.0 (always
    answer) when generating records for this curve, then apply thresholds
    here after the fact.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    curve = []
    for t in thresholds:
        answered = [r for r in records if r["confidence"] >= t]
        cov = len(answered) / len(records) if records else float("nan")
        if answered:
            errors = sum(1 for r in answered if r["root_cause"] != r["ground_truth"])
            risk = errors / len(answered)
        else:
            risk = float("nan")
        curve.append({"threshold": float(t), "coverage": cov, "risk": risk})
    return curve


def symptom_as_root_cause_rate(records: list, topology: dict) -> float:
    """
    Error-quality metric: among WRONG diagnoses, what fraction blamed a
    service that is downstream of the true root cause (i.e. a plausible
    symptom) rather than something topologically unrelated?
    """
    wrong = [r for r in records if r["decision"] == "diagnosed" and r["root_cause"] != r["ground_truth"]]
    if not wrong:
        return float("nan")

    def is_downstream(candidate, true_cause, topo, depth=0, seen=None):
        if seen is None:
            seen = set()
        if candidate in seen or depth > 10:
            return False
        seen.add(candidate)
        deps = topo.get(candidate, [])
        if true_cause in deps:
            return True
        return any(is_downstream(candidate, d, topo, depth + 1, seen) for d in deps)

    downstream_blames = sum(
        1 for r in wrong if is_downstream(r["root_cause"], r["ground_truth"], topology)
    )
    return downstream_blames / len(wrong)


def cost_summary(records: list) -> dict:
    """Average investigation cost per incident."""
    if not records:
        return {"avg_queries": float("nan")}
    return {"avg_queries": float(np.mean([r["num_queries"] for r in records]))}
