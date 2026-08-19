# Provenance status of this corpus (2026-08-19)

The 30 trials here are HISTORICAL and were produced under two defects
fixed on 2026-08-18 (see LOGBOOK.md):

1. **Early-stop guillotine** — every trial was terminated after exactly
   patience+1 epochs (patience 3-20 sampled) regardless of progress; no
   trial was allowed to converge.
2. **Crash-inheritance scorer** — a trial whose training crashed could be
   scored from the previous trial's metrics file.

Consequently: the best value here (41.86% exact, trial 23) is a FLOOR
obtained under truncated training, and per-trial hyperparameter
conclusions drawn from this corpus are not trustworthy. Keep for the
record; do not tune against it. Re-run the study with the fixed tuner
(`optuna_tuning.py`) to produce a comparable corpus.
