"""
Training-free evidence checks. Each check takes a hypothesis + the incident's
telemetry/topology and returns a verdict: "support", "contradict", or
"inconclusive", plus a human-readable reason (this becomes the evidence trail
in your final output).

These are intentionally simple rules, not learned models -- that's the whole
point of the "training-free verification policy" framing in the proposal.
"""
from dataclasses import dataclass
from typing import Literal, Optional

Verdict = Literal["support", "contradict", "inconclusive"]


@dataclass
class EvidenceResult:
    check_name: str
    hypothesis_id: str
    verdict: Verdict
    reason: str


def temporal_ordering_check(hypothesis, incident) -> EvidenceResult:
    """
    Does this hypothesis's service show an anomaly onset *before* other
    services in the propagation path? Early onset = support; late onset
    (i.e. this looks like a downstream symptom, not the source) = contradict.
    """
    metrics = incident.get("telemetry", {}).get("metrics", {})
    svc = hypothesis.service
    svc_series = metrics.get(svc, {})

    onset_times = {}
    for other_svc, series_dict in metrics.items():
        onset_times[other_svc] = _first_anomaly_time(series_dict, incident)

    this_onset = onset_times.get(svc)
    if this_onset is None:
        return EvidenceResult(
            "temporal_ordering", hypothesis.id, "inconclusive",
            f"No anomaly onset detected for {svc}."
        )

    earlier_services = [s for s, t in onset_times.items()
                        if t is not None and t < this_onset and s != svc]

    if not earlier_services:
        return EvidenceResult(
            "temporal_ordering", hypothesis.id, "support",
            f"{svc} shows the earliest anomaly onset among observed services."
        )
    else:
        return EvidenceResult(
            "temporal_ordering", hypothesis.id, "contradict",
            f"{svc} anomaly appears after {earlier_services}, suggesting it's downstream."
        )


def _first_anomaly_time(series_dict, incident, z_thresh=2.0):
    """
    Anomaly onset detector. If the incident has an `inject_time` (real
    RCAEval cases do; the synthetic toy example doesn't), this computes each
    metric's baseline mean/std from ONLY the pre-injection data, then finds
    the first POST-injection point that deviates from that pre-fault normal.

    This matters a lot in practice: the earlier version computed mean/std
    from the WHOLE window (pre- and post-fault mixed together), so a
    service's own post-fault spike inflated its own baseline stats, and
    unrelated services' ordinary pre-fault noise could look just as
    "anomalous" as the real fault. On real RCAEval data this made onset
    detection close to noise (see chat: risk-coverage curve came out flat).
    This mirrors how real RCA baselines like BARO use a known incident
    timestamp as an input, rather than trying to detect it themselves.

    Falls back to the old whole-window approach when no inject_time is
    available (keeps the synthetic toy example working unchanged).
    """
    inject_time = incident.get("inject_time")
    earliest = None

    for metric_name, series in series_dict.items():
        if len(series) < 3:
            continue

        if inject_time is not None:
            pre = [v for t, v in series if t < inject_time]
            post = [(t, v) for t, v in series if t >= inject_time]
            if len(pre) < 2 or not post:
                continue  # not enough pre-fault baseline, or no post-fault data
            mean = sum(pre) / len(pre)
            var = sum((v - mean) ** 2 for v in pre) / len(pre)
            std = var ** 0.5 or 1e-6
            for t, v in post:
                if abs(v - mean) / std > z_thresh:
                    if earliest is None or t < earliest:
                        earliest = t
                    break
        else:
            # Fallback: no known injection time (e.g. synthetic example) --
            # use the original whole-series approach.
            values = [v for _, v in series]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = var ** 0.5 or 1e-6
            for t, v in series:
                if abs(v - mean) / std > z_thresh:
                    if earliest is None or t < earliest:
                        earliest = t
                    break

    return earliest


def threshold_check(hypothesis, incident) -> EvidenceResult:
    """
    Does the hypothesized service actually show anomalous metric values
    AFTER the fault was injected, compared to its own PRE-injection
    baseline (not just "above its own overall average", which was too
    easily satisfied by ordinary noise on real data).
    """
    metrics = incident.get("telemetry", {}).get("metrics", {}).get(hypothesis.service, {})
    if not metrics:
        return EvidenceResult(
            "threshold", hypothesis.id, "inconclusive",
            f"No metrics available for {hypothesis.service}."
        )

    inject_time = incident.get("inject_time")

    for metric_name, series in metrics.items():
        if inject_time is not None:
            pre = [v for t, v in series if t < inject_time]
            post = [v for t, v in series if t >= inject_time]
            if len(pre) < 2 or not post:
                continue
            mean = sum(pre) / len(pre)
            var = sum((v - mean) ** 2 for v in pre) / len(pre)
            std = var ** 0.5 or 1e-6
            max_post_z = max(abs(v - mean) / std for v in post)
            if max_post_z > 2.0:
                return EvidenceResult(
                    "threshold", hypothesis.id, "support",
                    f"{hypothesis.service}.{metric_name} deviates {max_post_z:.1f} std "
                    f"from its pre-injection baseline after the fault."
                )
        else:
            # Fallback (no inject_time, e.g. the synthetic toy example):
            # use the SAME z-score>2.0 criterion as _first_anomaly_time and
            # the hypothesis-ranking heuristic, for consistency. This used
            # to be a different rule ("max > 1.5x mean"), which could
            # contradict a hypothesis that every other part of the pipeline
            # had already correctly flagged as anomalous -- found via
            # testing: a real spike whose mean shifted upward enough that
            # its max never reached 1.5x its own (spike-inflated) mean.
            values = [v for _, v in series]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = var ** 0.5 or 1e-6
            max_z = max(abs(v - mean) / std for v in values)
            if max_z > 2.0:
                return EvidenceResult(
                    "threshold", hypothesis.id, "support",
                    f"{hypothesis.service}.{metric_name} deviates {max_z:.1f} std from its baseline."
                )
    return EvidenceResult(
        "threshold", hypothesis.id, "contradict",
        f"No metric for {hypothesis.service} shows a clear anomalous spike."
    )


def log_pattern_check(hypothesis, incident) -> EvidenceResult:
    """Do this service's logs contain error/failure patterns?"""
    logs = incident.get("telemetry", {}).get("logs", {}).get(hypothesis.service, [])
    if not logs:
        return EvidenceResult(
            "log_pattern", hypothesis.id, "inconclusive",
            f"No logs available for {hypothesis.service}."
        )

    error_keywords = ["error", "exception", "failed", "timeout", "refused"]
    hits = [line for line in logs if any(k in line.lower() for k in error_keywords)]

    if hits:
        return EvidenceResult(
            "log_pattern", hypothesis.id, "support",
            f"Found {len(hits)} error-pattern log lines for {hypothesis.service}."
        )
    return EvidenceResult(
        "log_pattern", hypothesis.id, "contradict",
        f"No error patterns found in {hypothesis.service} logs."
    )


def topology_consistency_check(hypothesis, incident) -> EvidenceResult:
    """
    Is this service's position in the dependency graph consistent with it
    being the *source* of the failure, i.e. does it not depend on any other
    currently-anomalous service? If it depends on another anomalous service,
    it's more likely a downstream symptom.
    """
    topology = incident.get("topology", {})
    metrics = incident.get("telemetry", {}).get("metrics", {})
    svc = hypothesis.service
    dependencies = topology.get(svc, [])

    anomalous_deps = [
        dep for dep in dependencies
        if _first_anomaly_time(metrics.get(dep, {}), incident) is not None
    ]

    if not anomalous_deps:
        return EvidenceResult(
            "topology_consistency", hypothesis.id, "support",
            f"{svc} does not depend on any other currently-anomalous service."
        )
    return EvidenceResult(
        "topology_consistency", hypothesis.id, "contradict",
        f"{svc} depends on anomalous service(s) {anomalous_deps}, "
        f"so it may be a downstream symptom instead."
    )


# Registry used by the pipeline to pick the next check to run.
CHECK_REGISTRY = {
    "temporal_ordering": temporal_ordering_check,
    "threshold": threshold_check,
    "log_pattern": log_pattern_check,
    "topology_consistency": topology_consistency_check,
}