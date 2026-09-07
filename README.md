# Evidence-Guided RCA with Calibrated Abstention

A lightweight, training-free LLM pipeline for root cause analysis (RCA) in
distributed systems, evaluated on whether it knows when **not** to answer.

## Project structure

```
rca-abstention/
├── src/
│   ├── config.py          # thresholds, budgets, model settings
│   ├── llm_client.py      # LLM calls (hypothesis generation) + a mock mode
│   ├── evidence.py        # rule-based, training-free evidence checks
│   ├── pipeline.py        # the hypothesize -> select -> verify -> decide loop
│   └── data_loader.py     # loads incidents (synthetic now, RCAEval later)
├── eval/
│   ├── metrics.py         # accuracy, calibration (Brier/ECE), risk-coverage
│   └── plots.py           # risk-coverage curve, reliability diagram
├── experiments/
│   ├── run_toy_example.py # runs the pipeline on synthetic data, no API key needed
│   └── run_experiment.py  # runs over a real dataset (RCAEval), produces results.csv
├── tests/
│   └── test_pipeline.py   # unit tests for evidence checks and decision logic
└── data/                  # put downloaded RCAEval data here (gitignored)
```

## Quickstart (no API key needed yet)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_toy_example.py
```

This runs the full loop — hypothesis generation, evidence checks, confidence
updates, and the diagnose/abstain decision — using a **mock LLM** and a
synthetic incident, so you can see the mechanics work before spending any
API budget.

## Running with a real LLM

Set the relevant API key as an environment variable in your own shell --
NEVER put a real key into this file or any other tracked file (if you ever
do by accident, treat that key as compromised and regenerate it immediately,
since git history keeps it even after you edit it out).

```bash
# Groq (default provider -- see LLM_PROVIDER in src/config.py):
export GROQ_API_KEY=your-key-here
python experiments/run_toy_example.py --live

# Or Anthropic -- set LLM_PROVIDER = "anthropic" in src/config.py first:
export ANTHROPIC_API_KEY=your-key-here
python experiments/run_toy_example.py --live
```

## Next steps (in order)

1. Confirm the toy example runs and makes sense to you (read `pipeline.py` alongside it)
2. Swap in real hypothesis generation with `--live`
3. Download RCAEval RE1, adapt `data_loader.py` to parse it
4. Run `experiments/run_experiment.py` over real incidents
5. Use `eval/metrics.py` + `eval/plots.py` to produce the risk-coverage curve