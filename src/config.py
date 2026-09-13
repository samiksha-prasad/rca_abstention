"""
Central place for all the knobs of the pipeline.
Change these to run different experimental conditions (Condition F = budget sweep,
threshold sweep for the risk-coverage curve, etc.)
"""

# --- Decision thresholds ---
# Confidence the top hypothesis must reach before we're willing to diagnose.
# Swept across many values to build the risk-coverage curve.
CONFIDENCE_THRESHOLD = 0.7

# How many NET contradictions (contradictions minus supports) the LLM's
# original raw-leader hypothesis must accumulate before any confidence
# penalty is applied to it at all. 1 = old behavior (a single contradict
# immediately penalizes, same as every other hypothesis). Only applies to
# whichever hypothesis is flagged hyp.is_raw_leader -- deliberately NOT
# uniform across all hypotheses (see pipeline._apply_evidence for why a
# uniform version was tried first and found to be a confound).
MIN_CONTRADICTS_TO_OVERTURN = 2

# --- Budget (Condition F: limited investigation budget) ---
# Max number of evidence checks the pipeline may run per incident.
# Raised from 8 to 12: since hypothesis generation now considers every
# service in the topology (not just a top-4 cut), more candidates are
# competing for the same budget, and round-robin coverage in
# _select_next_check needs enough queries to reach lower-ranked hypotheses,
# including potentially the true root cause. This is exactly the Condition F
# knob from your evaluation plan -- feel free to sweep it across values.
MAX_EVIDENCE_QUERIES = 12

# Minimum checks guaranteed per hypothesis when scaling the budget (see
# pipeline.diagnose). With 4 check types available, 2 means every
# hypothesis is guaranteed at least half of them before budget runs out,
# rather than the budget being a fixed number regardless of how many
# hypotheses are competing for it.
MIN_CHECKS_PER_HYPOTHESIS = 2

# --- Anomaly detection threshold (used by temporal_ordering and threshold checks) ---
# How many standard deviations from baseline counts as "anomalous". Raised
# from 2.0 to 3.5 after finding, via instrumentation on real RE1 data, that
# 2.0 was far too lenient: with ~10 services each having multiple metrics,
# even a modest per-test false-positive rate compounds badly -- confirmed
# on real data as threshold_check returning "support" for EVERY SINGLE
# hypothesis in a case, regardless of actual guilt, making the check pure
# noise instead of a discriminator. z=3.5 corresponds to roughly a
# 1-in-2000 chance of a single point crossing by pure randomness, a much
# safer margin given how many checks run per case.
ANOMALY_Z_THRESHOLD = 3.5

# Require the anomaly to hold for several CONSECUTIVE points, not just one
# spike. Found via testing on real data: even at z=3.5, a single bursty
# sample (common in real metrics like "load", which spike from ordinary
# traffic with no fault involved) was enough to make threshold_check
# return "support" for EVERY hypothesis in a case -- one-off spikes are
# common in real telemetry, but a SUSTAINED shift is a much stronger
# signal of an actual state change.
MIN_CONSECUTIVE_ANOMALOUS_POINTS = 3

# --- Hypothesis generation ---
NUM_HYPOTHESES = 4  # how many competing hypotheses the LLM proposes per incident

# --- LLM settings ---
# Which provider to use for --live hypothesis generation: "anthropic" or "groq".
# Set the matching API key as an environment variable (ANTHROPIC_API_KEY or
# GROQ_API_KEY) -- never hardcode a key into this file or any other.
LLM_PROVIDER = "groq"

MODEL_NAME = "claude-sonnet-5"          # used when LLM_PROVIDER == "anthropic"
GROQ_MODEL = "openai/gpt-oss-20b"      # used when LLM_PROVIDER == "groq"
                                        # (fast/cheap; use "openai/gpt-oss-120b"
                                        # for stronger reasoning, at higher cost)
MAX_TOKENS = 1500

# --- Confidence update rule (simple, training-free heuristic) ---
# How much a single "support" or "contradict" result shifts a hypothesis's score.
# This is intentionally simple to start; you can replace it with something
# more principled (e.g. a proper Bayesian update) once the loop works end to end.
SUPPORT_DELTA = 0.30
CONTRADICT_DELTA = -0.35

# Floor after a contradiction is a small POSITIVE number, not exactly 0.0.
# A hard 0.0 can never recover under multiplicative renormalization
# (0 / anything = 0 forever) unless that specific hypothesis later gets its
# own "support" -- but with a limited evidence budget, it might never get
# checked again. Found via testing: the true root cause on a real RE2 case
# got contradicted once, hit 0.0, and was permanently locked out even
# though the check most likely to support it never got the chance to run.
MIN_CONFIDENCE_FLOOR = 0.03