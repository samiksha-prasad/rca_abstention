# Data folder

This is where you'll put the downloaded RCAEval RE1 files. It's gitignored
(see `.gitignore`) so the dataset itself doesn't get committed to the repo --
only this README does.

Steps:
1. Download RCAEval RE1 from its GitHub repo.
2. Extract it here.
3. Inspect one case's file structure, then fill in `src/data_loader.py::load_rcaeval_case`
   to parse it into the shape `make_synthetic_incident()` returns.
