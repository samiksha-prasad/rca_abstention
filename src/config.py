"""
Central place for all the knobs of the pipeline.
Change these to run different experimental conditions (Condition F = budget sweep,
threshold sweep for the risk-coverage curve, etc.)
"""

# --- Decision thresholds ---
# Confidence the top hypothesis must reach before we're willing to diagnose.
# Swept across many values to build the risk-coverage curve.
CONFIDENCE_THRESHOLD = 0.7

# --- Budget (Condition F: limited investigation budget) ---
# Max number of evidence checks the pipeline may run per incident.
# Raised from 8 to 12: since hypothesis generation now considers every
# service in the topology (not just a top-4 cut), more candidates are
# competing for the same budget, and round-robin coverage in
# _select_next_check needs enough queries to reach lower-ranked hypotheses,
# including potentially the true root cause. This is exactly the Condition F
# knob from your evaluation plan -- feel free to sweep it across values.
MAX_EVIDENCE_QUERIES = 12

# --- Hypothesis generation ---
NUM_HYPOTHESES = 4  # how many competing hypotheses the LLM proposes per incident

# --- LLM settings ---
# Which provider to use for --live hypothesis generation: "anthropic" or "groq".
# Set the matching API key as an environment variable (ANTHROPIC_API_KEY or
# GROQ_API_KEY) -- never hardcode a key into this file or any other.
LLM_PROVIDER = "groq"

MODEL_NAME = "claude-sonnet-4-6"       # used when LLM_PROVIDER == "anthropic"
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