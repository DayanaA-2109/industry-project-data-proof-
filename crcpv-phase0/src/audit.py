import numpy as np

THRESHOLD = 0.05  # pre-registered in protocol.md; not to be retuned


def ledger_audit(bank_rows, train_hash):
    """Deterministic provenance audit. Returns (flag, reason)."""
    bad_label = [r for r in bank_rows if r["fit_split"] != "train"]
    if bad_label:
        return True, f"{len(bad_label)} actions declare fit_split={bad_label[0]['fit_split']}"
    bad_hash = [r for r in bank_rows if r.get("fit_hash") != train_hash]
    if bad_hash:
        return True, f"{len(bad_hash)} actions: label says train but fit_hash != train hash"
    return False, "all actions fit on train (label and hash agree)"


def reward_gap_audit(val_rewards, test_rewards, thr=THRESHOLD):
    """Statistical audit from the protocol: |mean(test) - mean(val)| > thr."""
    gap = abs(float(np.mean(test_rewards)) - float(np.mean(val_rewards)))
    return gap > thr, gap
