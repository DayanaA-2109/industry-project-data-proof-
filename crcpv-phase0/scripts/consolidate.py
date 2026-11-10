import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

R = "results"
parts = []
bank = pd.read_csv(f"{R}/repair_bank_split.csv")
parts.append(("bank_rows_clean", len(bank)))
parts.append(("bank_all_fit_split_train", bool((bank.fit_split == "train").all())))
parts.append(("leakage_clean", open(f"{R}/logs/leakage_check_split.txt").read().strip()))
parts.append(("leakage_leaky", open(f"{R}/logs/leakage_check_leaky.txt").read().strip()[:90]))
rp = pd.read_csv(f"{R}/replay_split.csv")
parts.append(("replay_rows", len(rp)))
parts.append(("replay_nonzero", int((rp.delta.abs() > 1e-9).sum())))
parts.append(("replay_mean_abs_delta", round(rp.delta.abs().mean(), 4)))
a = pd.read_csv(f"{R}/metrics/audit_results.csv")
for _, x in a.iterrows():
    parts.append((f"audit_{x.case}_ledger_flag", bool(x.ledger_flag)))
    parts.append((f"audit_{x.case}_stat_flag(gap={x.reward_gap})", bool(x.stat_flag)))
c = pd.read_csv(f"{R}/metrics/comparison.csv").groupby("mode")[["test_auc", "repair_mae"]].mean().round(4)
for m, x in c.iterrows():
    parts.append((f"cmp_{m}_mean_auc", x.test_auc))
    parts.append((f"cmp_{m}_mean_mae", x.repair_mae))
out = pd.DataFrame(parts, columns=["metric", "value"])
out.to_csv(f"{R}/metrics/final_results.csv", index=False)
print(out.to_string(index=False))
