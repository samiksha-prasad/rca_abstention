# Downloading RCAEval and inspecting a real case

## 1. Install and download (run this yourself — needs Zenodo access, which
##    isn't available from Claude's sandbox, only from your own machine)

```bash
pip install RCAEval tqdm
python -c "from RCAEval.utility import download_re1_dataset; download_re1_dataset()"
```

This downloads RE1 (metrics-only, ~390MB total across 3 systems) into `data/RE1/`.
It can take a few minutes. If you only want one system to start (recommended,
to save time/disk), you can call the more specific function instead:

```bash
python -c "from RCAEval.utility import download_re1ob_dataset; download_re1ob_dataset()"
```

This gets just Online Boutique (125 cases), which is enough to get the pipeline
working before pulling the other two systems.

## 2. Inspect one real case

Once downloaded, run:

```bash
python inspect_case.py
```

(This script is in the project root — see below.) Paste its output back into
the chat with Claude — that's what will let the data loader get written
correctly on the first real try, instead of guessing at the JSON structure.
