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

from . import config

Verdict = Literal["support", "contradict", "inconclusive"]


@dataclass
class EvidenceResult:
    check_name: str
    hypothesis_id: str
    verdict: Verdict
    reason: str


def _sustained_anomaly(points, mean, std, z_thresh, min_consecutive):
    """
    Shared helper for BOTH _first_anomaly_time and threshold_check: scans
    a list of (t, v) points and looks for a run of at least
    `min_consecutive` CONSECUTIVE points whose z-score exceeds z_thresh --
    not just any single point, which was found to be far too easy to
    trigger by chance on real bursty telemetry (see config.py comment on
    MIN_CONSECUTIVE_ANOMALOUS_POINTS for the real-data evidence).

    Returns (onset_time, max_z_in_run) for the FIRST qualifying run, or
    (None, None) if no run of sufficient length is found.
    """
    streak_start_idx = None
    streak_count = 0
    for i, (t, v) in enumerate(points):
        z = abs(v - mean) / std
        if z > z_thresh:
            if streak_count == 0:
                streak_start_idx = i
            streak_count += 1
            if streak_count >= min_consecutive:
                run = points[streak_start_idx:i + 1]
                max_z = max(abs(vv - mean) / std for _, vv in run)
                return points[streak_start_idx][0], max_z
        else:
            streak_count = 0
    return None, None


def _max_sustained_z(post_values, mean, std, min_consecutive=None):
    """
    Returns the highest z-score that holds for at least `min_consecutive`
    CONSECUTIVE post-injection points, or None if no such run exists.

    This exists because raising the z-threshold alone wasn't enough: real
    telemetry (especially bursty metrics like request "load") can throw a
    single one-off spike well past even a strict threshold, with nothing
    to do with any actual fault. Requiring the deviation to be SUSTAINED
    across several consecutive samples -- not just one lucky/unlucky point
    -- filters out this kind of single-sample noise. Confirmed necessary
    via testing: even after raising ANOMALY_Z_THRESHOLD from 2.0 to 3.5,
    a real case still showed EVERY hypothesis getting "support" from
    threshold_check, because bursty metrics kept throwing single-point
    spikes past the static threshold regardless of which service was
    actually at fault.
    """
    min_consecutive = min_consecutive if min_consecutive is not None else config.MIN_CONSECUTIVE_ANOMALOUS_POINTS
    if std < 1e-9 or len(post_values) < min_consecutive:
        return None

    z_scores = [abs(v - mean) / std for v in post_values]
    best = None
    streak = 0
    for z in z_scores:
        if z > config.ANOMALY_Z_THRESHOLD:
            streak += 1
            if streak >= min_consecutive:
                if best is None or z > best:
                    best = z
        else:
            streak = 0
    return best


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


def _first_anomaly_time(series_dict, incident, z_thresh=None):
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
    z_thresh = z_thresh if z_thresh is not None else config.ANOMALY_Z_THRESHOLD
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
            onset, _ = _sustained_anomaly(post, mean, std, z_thresh, config.MIN_CONSECUTIVE_ANOMALOUS_POINTS)
            if onset is not None and (earliest is None or onset < earliest):
                earliest = onset
        else:
            # Fallback: no known injection time (e.g. synthetic example) --
            # use the original whole-series approach.
            values = [v for _, v in series]
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = var ** 0.5 or 1e-6
            onset, _ = _sustained_anomaly(series, mean, std, z_thresh, config.MIN_CONSECUTIVE_ANOMALOUS_POINTS)
            if onset is not None and (earliest is None or onset < earliest):
                earliest = onset

    return earliest


def threshold_check(hypothesis, incident) -> EvidenceResult:
    """
    Does the hypothesized service actually show anomalous metric values
    AFTER the fault was injected, compared to its own PRE-injection
    baseline, SUSTAINED across several consecutive points (not just one
    spike).

    IMPORTANT: this must use the SAME sustained-anomaly logic as
    _first_anomaly_time (via _sustained_anomaly), not a single-point check.
    An earlier version of this function checked only the single highest
    post-injection z-score, which is exactly what let it return "support"
    for EVERY hypothesis in a real case even after the z-threshold was
    raised to 3.5 -- a real fix to _first_anomaly_time's persistence logic
    was made, but this function was accidentally left behind using the old
    single-point rule, so the underlying bug persisted here even though it
    looked fixed elsewhere. Found via testing: instrumenting a real case
    showed round 2 (threshold) still flooding all 10 hypotheses with
    "support" after round 1 (temporal_ordering, which DID use the fixed
    helper) had already shown real discrimination.
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
            post = [(t, v) for t, v in series if t >= inject_time]
            if len(pre) < 2 or not post:
                continue
            mean = sum(pre) / len(pre)
            var = sum((v - mean) ** 2 for v in pre) / len(pre)
            std = var ** 0.5 or 1e-6
            onset, max_z = _sustained_anomaly(
                post, mean, std, config.ANOMALY_Z_THRESHOLD, config.MIN_CONSECUTIVE_ANOMALOUS_POINTS
            )
            if onset is not None:
                return EvidenceResult(
                    "threshold", hypothesis.id, "support",
                    f"{hypothesis.service}.{metric_name} sustains {max_z:.1f} std "
                    f"above its pre-injection baseline for "
                    f"{config.MIN_CONSECUTIVE_ANOMALOUS_POINTS}+ consecutive points."
                )
        else:
            # Fallback (no inject_time, e.g. the synthetic toy example):
            # same sustained-anomaly logic, whole-series baseline.
            values = [v for _, v in series]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            std = var ** 0.5 or 1e-6
            onset, max_z = _sustained_anomaly(
                series, mean, std, config.ANOMALY_Z_THRESHOLD, config.MIN_CONSECUTIVE_ANOMALOUS_POINTS
            )
            if onset is not None:
                return EvidenceResult(
                    "threshold", hypothesis.id, "support",
                    f"{hypothesis.service}.{metric_name} sustains {max_z:.1f} std above its baseline."
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
#
# ORDER MATTERS: with round-robin selection and a limited budget, if the
# budget doesn't fit every check type for every hypothesis, whichever check
# is LAST in this dict is the one that gets dropped. This was found to
# matter a lot in practice: with 4 hypotheses and MAX_EVIDENCE_QUERIES=12,
# exactly 3 of these 4 checks fit (4 hyps x 3 rounds = 12). The original
# order put log_pattern 3rd and topology_consistency 4th -- so
# topology_consistency (arguably the most discriminating check, since it
# directly flags "this is probably a downstream symptom") NEVER RAN,
# while log_pattern -- which is GUARANTEED to return "inconclusive" on
# RE1 data, since RE1 has no logs at all -- ran for every single
# hypothesis, wasting a third of the entire evidence budget. Putting
# topology_consistency before log_pattern means the useless-on-this-
# dataset check is the one that gets dropped, not the useful one.
CHECK_REGISTRY = {
    "temporal_ordering": temporal_ordering_check,
    "threshold": threshold_check,
    "topology_consistency": topology_consistency_check,
    "log_pattern": log_pattern_check,
}