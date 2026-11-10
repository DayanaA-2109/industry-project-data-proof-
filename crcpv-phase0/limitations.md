# Limitations (Phase 0)

1. Single dataset, single column, single synthetic corruption (x2.5). No generalisation claim.
2. Repair is applied through the ground-truth corruption mask (oracle error detection).
3. Clip bounds are fit on CLEAN train data, which a real pipeline would not have.
4. Repair MAE stays ~16 (clipping to p99 only caps outliers); A vs B differ negligibly in AUC.
   Leakage here is a provenance problem, not a large accuracy effect.
5. Statistical audit (reward gap) is noisy with ~16 corrupted rows per split: on seed 42 the
   CLEAN control exceeds the 0.05 threshold (false positive, gap 0.0523). The deterministic
   ledger audit is the reliable detector; the statistical one is reported, not tuned.
6. The leakage assertion checks the ledger (label + data hash). It cannot detect leakage that
   never passes through the repair bank (e.g. test data used in model selection).
7. Replay measures a change in model output for a fixed model, not causal effect on retraining.

## Known false positive in statistical audit signal

On `control_clean_s42`, the audit's statistical val-test-gap signal flags
the pipeline as contaminated (gap=0.0523, threshold=0.05). The ledger
signal — the primary detector — correctly reports `False`. This is a
false positive of the secondary statistical corroboration and demonstrates
why the ledger signal must be the primary mechanism. Phase 1 will
calibrate the threshold on multiple seeds and add a scrambled-key control.