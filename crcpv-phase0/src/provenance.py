import csv
import time
import uuid

SCHEMA = [
    "action_id", "timestamp", "dataset_hash", "split_hash",
    "row_index", "column", "method", "source_split", "fit_split",
    "fit_hash",  # NEW: SHA-256 of the data the operator was really fit on
    "code_version", "old_value", "new_value",
]


def init_bank(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=SCHEMA).writeheader()


def log(path, record):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=SCHEMA).writerow(record)


def write_bank(path, records):
    init_bank(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        for r in records:
            w.writerow(r)


def make_record(**kw):
    r = {k: kw.get(k, "") for k in SCHEMA}
    if not r["action_id"]:
        r["action_id"] = str(uuid.uuid4())
    if not r["timestamp"]:
        r["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return r
