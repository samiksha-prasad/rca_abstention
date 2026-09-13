# Evidence-Guided RCA with Calibrated Abstention

An LLM-based pipeline for root cause analysis (RCA) in distributed systems,
built to answer a specific research question: **can rule-based evidence
verification make an LLM's confidence trustworthy — and if a rigid formula
doesn't work, does letting the LLM interpret the evidence itself do
better?**

Tested end-to-end on real fault-injection data (RCAEval). See
`results/Full_Final_Documentation.md` for the complete write-up, findings,
and charts.

## Headline finding

A lightweight rule-based evidence-verification formula (fixed +0.30
support / −0.35 contradict, mechanically overriding the LLM's own guess)
**consistently and substantially hurts accuracy** — confirmed across two
datasets, two model providers, and three formula designs. Letting the LLM
interpret the same evidence as text instead — deciding for itself how much
to trust each check — recovers most of the lost accuracy and produces the
best-calibrated confidence of any approach tested.

| Dataset | RAW (LLM alone) | ARITHMETIC (formula) | SYNTHESIS (LLM interprets) |
|---|---|---|---|
| RE1 (125 cases) | 83.2% | 44.8% | 75.2% |
| RE2 (90 cases, real logs) | 94.4% | 45.6% | 78.9% |

## Project structure

```
rca-abstention/
├── src/
│   ├── config.py                  # thresholds, budgets, model/provider settings
│   ├── llm_client.py              # LLM calls: hypothesis generation + evidence synthesis, mock mode
│   ├── evidence.py                # rule-based, training-free evidence checks
│   ├── pipeline.py                # the hypothesize -> select -> verify -> decide loop (ARITHMETIC)
│   ├── data_loader.py             # loaders: synthetic toy example, RCAEval RE1, RCAEval RE2
│   └── topology.py                # known service dependency graphs (Online Boutique, etc.)
├── eval/
│   ├── metrics.py                 # accuracy, Brier score, risk-coverage curve
│   └── plots.py                   # risk-coverage curve, reliability diagram
├── experiments/
│   ├── run_toy_example.py         # single synthetic incident, no API key needed
│   ├── run_experiment.py          # full ARITHMETIC pipeline over a real dataset (RE1 or RE2)
│   ├── run_ablation.py            # RAW vs. ARITHMETIC (final) confidence, same cases
│   ├── run_synthesis_experiment.py# RAW vs. ARITHMETIC vs. SYNTHESIS, three-way comparison
│   ├── instrument_confidence.py   # step-by-step confidence audit trail for one case (debugging)
│   └── analyze_flips.py           # breaks down good/bad flips by raw-confidence bucket
├── tests/
│   └── test_pipeline.py           # unit tests for evidence checks and decision logic
├── data/                          # put downloaded RCAEval data here (gitignored) -- see DOWNLOAD_RCAEVAL.md
├── results/
│   ├── Full_Final_Documentation.md# the complete write-up: problem, findings, charts
│   └── figures/                   # generated comparison charts
├── inspect_case.py                # diagnostic: prints the real file structure of a downloaded case
└── DOWNLOAD_RCAEVAL.md            # how to download RE1/RE2
```

## Quickstart (no API key needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_toy_example.py
```

This runs the full loop on one synthetic incident using a **mock LLM** (a
z-score-based heuristic, not a real model call), so you can see the
mechanics work before spending any API budget.

## Running with a real LLM

Put your key in a local `.env` file (already gitignored) rather than
exporting it manually each session:

```
GROQ_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

Then run any experiment with `--live`, optionally choosing a provider:

```bash
python experiments/run_toy_example.py --live --provider anthropic
```

If `--live` is passed without a valid key for the selected provider, this
fails loudly with a clear error — it will never silently fall back to mock
mode.

## Working with real data (RCAEval)

See `DOWNLOAD_RCAEVAL.md` for exact download commands. Once downloaded,
`inspect_case.py` will print the real file structure of one case (useful
if RCAEval's format ever changes) and the data loaders in `src/data_loader.py`
handle the rest.

```bash
# RE1: metrics only, 125 Online Boutique cases
python experiments/run_experiment.py --dataset re1 --live --provider anthropic

# RE2: metrics + real logs + traces, 90 Online Boutique cases
python experiments/run_experiment.py --dataset re2 --live --provider anthropic
```

## Reproducing the three-way comparison (the main result)

```bash
python experiments/run_synthesis_experiment.py --dataset re1 --live --provider anthropic
python experiments/run_synthesis_experiment.py --dataset re2 --live --provider anthropic
```

Each produces `results/synthesis_results.csv` and prints accuracy + Brier
score for all three approaches (RAW / ARITHMETIC / SYNTHESIS).

## Debugging tools

- `experiments/instrument_confidence.py --real <case_path>` — prints every
  single confidence update, step by step, for one real case. This is how
  every bug in this project's history was actually found and confirmed,
  not guessed at.
- `experiments/analyze_flips.py results/ablation_results.csv` — after
  running `run_ablation.py`, breaks down how often verification helped vs.
  hurt, bucketed by how confident the LLM's raw guess already was.
