import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src import pipeline
from src.provenance import write_bank
from src.leakage import run_check

OUT = "results"
FROZEN = os.path.join("data", "splits", "test_hash.txt")
for d in (OUT, f"{OUT}/logs", f"{OUT}/metrics", "data/splits"):
    os.makedirs(d, exist_ok=True)


def freeze_or_verify(test_hash):
    """Freeze the test split hash on first run, verify on every later run."""
    if not os.path.exists(FROZEN):
        open(FROZEN, "w").write(test_hash + "\n")
        print("test split FROZEN:", test_hash)
    else:
        frozen = open(FROZEN).read().strip()
        if frozen != test_hash:
            sys.exit(f"ABORT: test split changed!\n frozen={frozen}\n now   ={test_hash}")
        print("test split verified against frozen hash")


def main(leaky=False):
    tag = "leaky" if leaky else "split"
    r = pipeline.run(seed=42, fit_mode="all" if leaky else "train")
    freeze_or_verify(r["hashes"]["test"])
    print("dataset_hash:", r["dataset_hash"])
    print("bounds used :", r["bounds"])

    bank_path = f"{OUT}/repair_bank_{tag}.csv"
    write_bank(bank_path, r["bank"])
    rows = pd.read_csv(bank_path).to_dict("records")
    print("bank rows   :", len(rows))

    ok, msg = run_check(rows, train_hash=r["hashes"]["train"])
    with open(f"{OUT}/logs/leakage_check_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(f"{'PASS' if ok else 'FAIL'}: {msg}\n")
    print("leakage check:", "PASS" if ok else f"FAIL - {msg}")

    rep = pd.DataFrame(r["replay"]("val"))
    rep.to_csv(f"{OUT}/replay_{tag}.csv", index=False)
    print(f"replay rows : {len(rep)}  nonzero: {(rep['delta'].abs() > 1e-9).sum()}  "
          f"mean|delta|: {rep['delta'].abs().mean():.4f}")


if __name__ == "__main__":
    main(leaky="--leaky" in sys.argv)
