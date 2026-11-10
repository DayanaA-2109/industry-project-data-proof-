# CRCPV Phase-0 Protocol (v0.2, corrected)

## Dataset
- Breast Cancer Wisconsin (Diagnostic), `sklearn.datasets.load_breast_cancer`
- 569 rows, 30 features + binary target (0 = malignant, 1 = benign)
- Corrupted column: **"worst texture"** (single column; v0.1 text wrongly also said "mean radius")

## Split
- Stratified 70/15/15 on target, seed 42 -> 398 / 85 / 86
- Test split is FROZEN: `data/splits/test_hash.txt` holds SHA-256 of the clean test *data*
  (v0.1 hashed only the row indices 0..85, which never changes). `run_demo.py` freezes on
  first run and ABORTS if the hash later differs.

## Corruption
- 20% of cells per split, value x 2.5
- Independent RNG stream per split (seed*10 + {train:0, val:1, test:2}). v0.1 reused one seed
  so all three splits had identical corruption masks.

## Repair
- Clip to p1-p99 of the **clean TRAIN split** (Phase-0 simplification; v0.1 text said "corrupted
  train", but the code used clean train and a p99 of corrupted data would not clip anything useful)
- Repair is applied only to cells flagged by the corruption mask (oracle detector, see limitations)
- `fit_split` and `fit_hash` in the ledger are DERIVED from the data actually passed to the fit
  function, not typed by hand.

## Baselines
- A: leaky (fit on train+val+test) | B: split-aware (train only) | C: no repair

## Metrics
- Test AUC (LogisticRegression trained on repaired train), repair MAE vs clean value,
  leakage PASS/FAIL, per-action replay delta.

## Seeds
- Comparison: 0-4 (split and corruption seed). Leakage demo: 42.

## Audit
- Ledger audit (deterministic): flag if any action has fit_split != train or fit_hash != hash(train).
- Statistical audit (pre-registered): |mean(test reward) - mean(val reward)| > 0.05 -> flag,
  reward = change in probability of the true class caused by the repair. Threshold NOT retuned.

## Out of scope
Surrogate credit models, ReClean reproduction, multi-dataset evaluation, any patentability,
novelty or superiority claim.
