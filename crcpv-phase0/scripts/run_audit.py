import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import pipeline
from src.audit import ledger_audit, reward_gap_audit, THRESHOLD

CASES = [  # name, seed, fit_mode, relabel, is_contaminated
    ("control_clean_s42", 42, "train", None, False),
    ("control_clean_s7",   7, "train", None, False),
    ("C1_leaky_honest_label", 42, "all", None, True),
    ("C2_leaky_label_forged_as_train", 42, "all", "train", True),
    ("C3_fit_on_test_only", 42, "test", None, True),
]
rows = []
for name, seed, fm, relabel, bad in CASES:
    r = pipeline.run(seed=seed, fit_mode=fm, relabel=relabel)
    flag, why = ledger_audit(r["bank"], r["hashes"]["train"])
    val = [x["reward"] for x in r["replay"]("val")]
    test = [x["reward"] for x in r["replay"]("test")]
    sflag, gap = reward_gap_audit(val, test)
    rows.append(dict(case=name, contaminated=bad, ledger_flag=flag, ledger_reason=why,
                     reward_gap=round(gap, 4), stat_flag=sflag, threshold=THRESHOLD))
df = pd.DataFrame(rows)
os.makedirs("results/metrics", exist_ok=True)
df.to_csv("results/metrics/audit_results.csv", index=False)
print(df.to_string(index=False))
