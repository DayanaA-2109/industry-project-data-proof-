def assert_no_test_in_fit(bank_rows, train_hash=None):
    """Raise AssertionError if any repair was fit on anything other than train.

    Two independent checks:
      1. the declared fit_split must be 'train'
      2. if train_hash is given, the recorded fit_hash must equal it
         (catches a ledger whose label says 'train' but whose data was not)
    """
    for r in bank_rows:
        assert r["fit_split"] == "train", (
            f"LEAK: action {r['action_id']} fit_split={r['fit_split']} "
            f"(row {r['row_index']}, col {r['column']})"
        )
        if train_hash is not None:
            assert r.get("fit_hash") == train_hash, (
                f"LEAK: action {r['action_id']} labelled fit_split=train but "
                f"fit_hash does not match the train split (label/data mismatch)"
            )


def run_check(bank_rows, train_hash=None):
    """Return (ok: bool, message: str)."""
    try:
        assert_no_test_in_fit(bank_rows, train_hash)
        return True, "PASS"
    except AssertionError as e:
        return False, str(e)
