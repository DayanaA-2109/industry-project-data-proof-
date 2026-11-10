import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src import pipeline
from src.data import TARGET, CORRUPT_COL

MODES = {"A_leaky": dict(fit_mode="all"), "B_split_aware": dict(fit_mode="train"),
         "C_no_repair": dict(repair=False)}
rows = []
for seed in range(5):
    for name, kw in MODES.items():
        r = pipeline.run(seed=seed, **kw)
        te, f = r["rep"]["test"], r["feats"]
        auc = roc_auc_score(te[TARGET], r["model"].predict_proba(te[f])[:, 1])
        m = r["mask"]["test"]
        mae = float(np.abs(te.loc[m, CORRUPT_COL] - r["sp"]["test"].loc[m, CORRUPT_COL]).mean())
        rows.append(dict(seed=seed, mode=name, test_auc=round(auc, 4), repair_mae=round(mae, 4)))
df = pd.DataFrame(rows)
os.makedirs("results/metrics", exist_ok=True)
df.to_csv("results/metrics/comparison.csv", index=False)
print(df.to_string(index=False))
print(df.groupby("mode")[["test_auc", "repair_mae"]].agg(["mean", "std"]).round(4))
