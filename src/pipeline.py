"""
The core control loop described in the proposal:

    generate hypotheses
        -> select next evidence check
        -> run check
        -> update confidences
        -> diagnose / keep going / abstain

Kept as plain, readable Python -- no agent framework -- so every decision
point is easy to inspect and modify.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .evidence import CHECK_REGISTRY, EvidenceResult
from .llm_client import Hypothesis, generate_hypotheses


@dataclass
class DiagnosisResult:
    incident_id: str
    decision: str  # "diagnosed" or "abstained"
    root_cause: Optional[str]
    confidence: float
    hypotheses: List[dict]
    evidence_trail: List[dict]
    num_queries: int

    def to_dict(self):
        return {
            "incident_id": self.incident_id,
            "decision": self.decision,
            "root_cause": self.root_cause,
            "confidence": round(self.confidence, 3),
            "hypotheses": self.hypotheses,
            "evidence_trail": self.evidence_trail,
            "num_queries": self.num_queries,
        }


def _select_next_check(hypotheses: List[Hypothesis], used_checks: dict) -> Optional[tuple]:
    """
    Evidence-selection heuristic: prioritize whichever hypothesis has been
    checked the FEWEST times so far (ties broken by higher current
    confidence), rather than always chasing the current leader.

    This guarantees round-robin coverage -- every hypothesis gets at least
    one check before any hypothesis gets a second, every hypothesis gets a
    second before any gets a third, and so on. The earlier "always chase
    the leader" version could leave a low-initial-confidence hypothesis
    (including the actual true root cause) completely unchecked if it never
    became the leader -- confirmed as a real failure mode during testing on
    real RCAEval data, made worse once hypothesis generation stopped
    truncating to a top-N list (see llm_client.py).

    Note: with a limited MAX_EVIDENCE_QUERIES budget and many hypotheses,
    not every hypothesis is guaranteed to get checked even with this fix --
    if you have ~10 hypotheses and a budget of 8, at most 8 will get their
    first check. Consider raising MAX_EVIDENCE_QUERIES in config.py if your
    topology has many services.
    """
    candidates = []
    for hyp in hypotheses:
        tried = used_checks.get(hyp.id, set())
        remaining = [c for c in CHECK_REGISTRY if c not in tried]
        if remaining:
            candidates.append((len(tried), -hyp.confidence, hyp, remaining[0]))

    if not candidates:
        return None  # every hypothesis has exhausted every check

    candidates.sort(key=lambda x: (x[0], x[1]))  # fewest checks first, then highest confidence
    _, _, hyp, check_name = candidates[0]
    return hyp, check_name


def _apply_evidence(hyp: Hypothesis, result: EvidenceResult):
    if result.verdict == "support":
        hyp.confidence = min(1.0, hyp.confidence + config.SUPPORT_DELTA)
        hyp.supporting_evidence.append(result.reason)
    elif result.verdict == "contradict":
        hyp.confidence = max(0.0, hyp.confidence + config.CONTRADICT_DELTA)
        hyp.contradicting_evidence.append(result.reason)
    # inconclusive: no change, but you could log it if you want full traceability


def _normalize(hypotheses: List[Hypothesis]):
    total = sum(h.confidence for h in hypotheses)
    if total > 0:
        for h in hypotheses:
            h.confidence = h.confidence / total


def diagnose(incident: dict, live: bool = False,
             confidence_threshold: float = None,
             max_queries: int = None) -> DiagnosisResult:
    """Run the full loop on a single incident."""
    # NOTE: must check "is None", not use `confidence_threshold or config...`,
    # because 0.0 is falsy in Python -- the "or" pattern would silently
    # replace an intentional confidence_threshold=0.0 with the default,
    # which is exactly the bug that caused run_experiment.py to report every
    # case as "abstained" even when explicitly asking the pipeline to always
    # answer.
    confidence_threshold = confidence_threshold if confidence_threshold is not None else config.CONFIDENCE_THRESHOLD
    max_queries = max_queries if max_queries is not None else config.MAX_EVIDENCE_QUERIES

    hypotheses = generate_hypotheses(incident, live=live)
    _normalize(hypotheses)

    used_checks = {h.id: set() for h in hypotheses}
    evidence_trail = []
    num_queries = 0

    while num_queries < max_queries:
        top = max(hypotheses, key=lambda h: h.confidence)
        top_fully_checked = len(used_checks[top.id]) == len(CHECK_REGISTRY)

        # Only allow stopping early if the leader has actually been checked
        # against every available piece of evidence. Otherwise a lucky early
        # "support" result could let an unverified hypothesis win by default
        # (this was the premature-convergence bug found during testing).
        if top.confidence >= confidence_threshold and top_fully_checked:
            break

        selection = _select_next_check(hypotheses, used_checks)
        if selection is None:
            break  # nothing left to check

        hyp, check_name = selection
        check_fn = CHECK_REGISTRY[check_name]
        result = check_fn(hyp, incident)

        used_checks[hyp.id].add(check_name)
        num_queries += 1
        evidence_trail.append({
            "check": result.check_name,
            "hypothesis_id": result.hypothesis_id,
            "verdict": result.verdict,
            "reason": result.reason,
        })

        _apply_evidence(hyp, result)
        _normalize(hypotheses)

    top = max(hypotheses, key=lambda h: h.confidence)
    if top.confidence >= confidence_threshold:
        decision = "diagnosed"
        root_cause = top.service
    else:
        decision = "abstained"
        root_cause = None

    return DiagnosisResult(
        incident_id=incident.get("incident_id", "unknown"),
        decision=decision,
        root_cause=root_cause,
        confidence=top.confidence,
        hypotheses=[h.to_dict() for h in hypotheses],
        evidence_trail=evidence_trail,
        num_queries=num_queries,
    )