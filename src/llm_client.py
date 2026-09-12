"""
Thin wrapper around the LLM call used for hypothesis generation.

Two modes:
  - mock: no API key needed, uses a simple heuristic so you can test the
    pipeline mechanics for free.
  - live: calls the real Anthropic API.

Keeping this in one small file means later, if you want to compare models
or add litellm, you only change this file.
"""
import json
import os
from dataclasses import dataclass, field
from typing import List

from . import config


@dataclass
class Hypothesis:
    id: str
    service: str
    description: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    # Set True on exactly one hypothesis (the raw top pick) right after
    # initial normalization in pipeline.diagnose(), so the "strong prior"
    # protection (requiring overwhelming evidence to overturn) can be
    # applied ONLY to that specific hypothesis, not uniformly to everyone.
    # A previous attempt applied the stricter rule to all hypotheses
    # equally, which was a confound: it made every wrong competitor harder
    # to knock down too, not just the LLM's leader -- found via testing,
    # results got WORSE in the medium/high confidence buckets instead of
    # better, the opposite of the intended effect.
    is_raw_leader: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "service": self.service,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
        }


def _compute_service_anomaly_scores(incident: dict) -> List[tuple]:
    """
    Shared anomaly-scoring logic, used by BOTH the mock hypothesis generator
    AND the live-LLM prompt (as of this update -- previously only the mock
    path saw this, which is why the live LLM was guessing from generic
    incident-type priors instead of the actual telemetry, and got the wrong
    answer on payment-service_delay-style cases).

    Returns [(service, z_score), ...] sorted highest-anomaly-first. Uses a
    Z-SCORE (not raw value) so metrics with different units -- memory in
    bytes, cpu in percent, latency in seconds -- are comparable. Uses
    inject_time (when available) to compare POST-injection values against
    a PRE-injection baseline, rather than blending both together.
    """
    inject_time = incident.get("inject_time")
    services = list(incident["topology"].keys())
    scored = []
    for svc in services:
        metrics = incident.get("telemetry", {}).get("metrics", {}).get(svc, {})
        max_z = 0.0
        for series in metrics.values():
            if inject_time is not None:
                pre = [v for t, v in series if t < inject_time]
                post = [v for t, v in series if t >= inject_time]
                if len(pre) < 2 or not post:
                    continue
                mean = sum(pre) / len(pre)
                var = sum((v - mean) ** 2 for v in pre) / len(pre)
                std = var ** 0.5
                if std < 1e-9:
                    continue
                z = max(abs(v - mean) for v in post) / std
            else:
                vals = [v for _, v in series]
                if len(vals) < 2:
                    continue
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                std = var ** 0.5
                if std < 1e-9:
                    continue
                z = max(abs(v - mean) for v in vals) / std
            max_z = max(max_z, z)
        scored.append((svc, max_z))
    scored.sort(key=lambda x: -x[1])
    return scored


def _mock_generate_hypotheses(incident: dict) -> List[Hypothesis]:
    """
    Free, deterministic stand-in for the LLM call. It proposes services
    ranked by how anomalous their metrics look, using _compute_service_anomaly_scores.

    This matters because real telemetry mixes units that aren't comparable
    in magnitude: memory in bytes (~10^7), CPU in percent (~1-10), latency
    in seconds (~0.01), load as raw counts (~10-100). Comparing raw values
    across these would always favor whichever metric happens to use the
    largest unit (e.g. memory), regardless of whether anything is actually
    wrong -- which is exactly the bug found when this was first tried on
    real RCAEval data (a service's ordinary byte-scale memory reading beat
    out another service's genuinely anomalous CPU spike).
    """
    scored = _compute_service_anomaly_scores(incident)

    # IMPORTANT: do NOT truncate to config.NUM_HYPOTHESES here. Truncating
    # based on this same (noisy) z-score heuristic can eliminate the true
    # root cause before the evidence-verification loop ever gets a chance to
    # check it -- found via testing: on real RCAEval data, the actual
    # faulted service didn't make a top-4 cut because other services had
    # higher z-scores from ordinary load/latency noise unrelated to the
    # injected fault. With a small topology (~10 services here), there's no
    # real cost to considering all of them as candidates.
    hyps = []
    for i, (svc, score) in enumerate(scored):
        hyps.append(
            Hypothesis(
                id=f"H{i+1}",
                service=svc,
                description=f"{svc} is the root cause (naive: highest metric z-score={score:.2f})",
                confidence=max(0.05, 1.0 / (i + 1)) * 0.5,
            )
        )
    return hyps


def _build_hypothesis_prompt(incident: dict) -> str:
    """
    Shared prompt used by every live provider, so Anthropic and Groq stay
    in sync -- edit this once, both providers benefit.

    UPDATE: now includes a telemetry summary (top anomalous services by
    z-score), not just the alert text and topology. Without this, the LLM
    had no way to know which service's metrics actually looked wrong, and
    fell back on generic priors about what "usually" causes the described
    symptom (e.g. guessing "database contention" for a slow-checkout alert)
    -- which is exactly what caused it to pick the wrong service despite
    the evidence-verification loop later favoring the correct one.
    """
    topology_str = json.dumps(incident["topology"], indent=2)
    alert_text = incident.get("alert_text", "")

    scored = _compute_service_anomaly_scores(incident)
    telemetry_lines = [
        f"  - {svc}: anomaly z-score = {score:.2f}"
        for svc, score in scored[:6] if score > 0
    ]
    telemetry_summary = (
        "\n".join(telemetry_lines)
        if telemetry_lines
        else "  (no clear metric anomalies detected)"
    )

    return f"""You are diagnosing a distributed systems incident.

Alert: {alert_text}

Service dependency topology (service -> services it depends on):
{topology_str}

Telemetry summary -- services with the most anomalous metrics since the
fault was reported, ranked highest first (higher z-score = more anomalous
relative to that service's own normal baseline):
{telemetry_summary}

Use the telemetry summary as your primary evidence -- it reflects what
actually happened, not just what commonly causes this kind of symptom.
Propose up to {config.NUM_HYPOTHESES} competing root-cause hypotheses.
Keep each description under 12 words -- this is being parsed as JSON, so
a long description risks the output being cut off before it's valid.
Respond ONLY with a JSON array, no other text, in this exact format:
[
  {{"service": "<service name from topology>", "description": "<short reason>", "initial_confidence": <0.0-1.0>}}
]
"""


def _parse_hypothesis_json(text: str) -> List[Hypothesis]:
    """Shared response parser -- strips accidental code fences, then turns
    the JSON array into Hypothesis objects. Used by every live provider."""
    text = text.strip()

    if not text:
        # Happens most often from rate limiting or a transient API hiccup
        # returning an empty body -- surfaced as a clear, specific error
        # instead of a cryptic "Expecting value: line 1 column 1" from
        # json.loads, so callers (like _call_with_retry below) know exactly
        # what went wrong and can retry.
        raise ValueError("LLM returned an empty response (likely rate-limited or a transient API error).")

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM response wasn't valid JSON: {e}. Raw response: {text[:300]!r}") from e

    hyps = []
    for i, item in enumerate(parsed):
        hyps.append(
            Hypothesis(
                id=f"H{i+1}",
                service=item["service"],
                description=item["description"],
                confidence=float(item["initial_confidence"]),
            )
        )
    return hyps


def _call_with_retry(fn, max_retries=3, base_delay=2.0):
    """
    Retries a live LLM call with exponential backoff -- specifically for
    rate limits and transient empty/malformed responses, which are common
    on free-tier API budgets when running many cases back to back (found
    via testing: a real experiment run crashed entirely on case 3 because
    one rate-limited call returned an empty body, and nothing caught it).
    """
    import time

    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    (LLM call failed: {e} -- retrying in {delay:.0f}s, "
                      f"attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
    raise last_error


def _live_generate_hypotheses(incident: dict) -> List[Hypothesis]:
    """Anthropic (Claude) provider."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = _build_hypothesis_prompt(incident)

    def _call():
        response = client.messages.create(
            model=config.MODEL_NAME,
            max_tokens=config.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return _parse_hypothesis_json(text)

    return _call_with_retry(_call)


def _live_generate_hypotheses_groq(incident: dict) -> List[Hypothesis]:
    """
    Groq provider. Groq speaks an OpenAI-compatible chat API (different
    request/response shape from Anthropic's), so this is a separate
    function rather than a branch inside _live_generate_hypotheses --
    but it uses the SAME prompt and SAME JSON parsing as the Anthropic
    path, via the shared helpers above, so the two providers behave
    identically from the rest of the pipeline's point of view.

    Requires: pip install groq, and GROQ_API_KEY set as an environment
    variable (never hardcode a key in this file).
    """
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from the environment automatically
    prompt = _build_hypothesis_prompt(incident)

    def _call():
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            max_tokens=config.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return _parse_hypothesis_json(text)

    return _call_with_retry(_call)


def generate_hypotheses(incident: dict, live: bool = False, provider: str = None) -> List[Hypothesis]:
    """
    provider: optional override for config.LLM_PROVIDER ("groq" or
    "anthropic"). Useful for switching providers per-run (e.g. via a
    --provider CLI flag) without editing config.py -- added after hitting
    Groq's free-tier daily token cap mid-experiment and needing to fall
    back to Anthropic without a code edit.

    IMPORTANT: if live=True is requested but no valid API key is found for
    the selected provider, this used to SILENTLY fall back to mock mode --
    found via testing: a full 125-case "--live" run produced results
    bit-for-bit identical to a mock run, with zero errors printed, because
    the API key wasn't actually set in that shell session. That silent
    fallback wastes real time and can make you think you got a live-LLM
    result when you didn't. Now it raises a clear, loud error instead.
    """
    provider = provider or config.LLM_PROVIDER
    if live:
        if provider == "groq":
            if not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError(
                    "live=True with provider='groq' but GROQ_API_KEY is not set in "
                    "this environment. Either export it, put it in a .env file "
                    "(and confirm load_dotenv() ran), or drop --live to use mock mode "
                    "on purpose."
                )
            return _live_generate_hypotheses_groq(incident)
        if provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "live=True with provider='anthropic' but ANTHROPIC_API_KEY is not "
                    "set in this environment. Either export it, put it in a .env file "
                    "(and confirm load_dotenv() ran), or drop --live to use mock mode "
                    "on purpose."
                )
            return _live_generate_hypotheses(incident)
    return _mock_generate_hypotheses(incident)