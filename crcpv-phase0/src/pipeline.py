"""One shared pipeline so demo / audit / comparison cannot drift apart."""
import numpy as np
import pandas as pd

from src.data import (load_clean, split, corrupt_split, hash_df, TARGET, CORRUPT_COL)
from src.repair import fit_clip_bounds_tracked, apply_clip
from src.provenance import make_record
from src.replay import train_model, replay_one

FIT_MODES = {"train": ["train"], "all": ["train", "val", "test"], "test": ["test"]}


def run(seed=42, fit_mode="train", relabel=None, repair=True):
    """fit_mode: which CLEAN splits the clip operator is fit on.
    relabel: lie in the ledger about fit_split (to simulate a tampered bank).
    repair=False: control (no repair)."""
    df = load_clean()
    train, val, test = split(df, seed=seed)
    sp = {"train": train, "val": val, "test": test}
    cor, mask = {}, {}
    for k, d in sp.items():
        cor[k], mask[k] = corrupt_split(d, k, seed=seed)

    d_hash = hash_df(df)
    hashes = {k: hash_df(d) for k, d in sp.items()}

    if repair:
        bounds, label, fit_hash = fit_clip_bounds_tracked(
            {k: sp[k] for k in FIT_MODES[fit_mode]}, CORRUPT_COL)
        if relabel:
            label = relabel
        rep = {k: apply_clip(cor[k], CORRUPT_COL, bounds, mask[k]) for k in sp}
    else:
        bounds, label, fit_hash = None, "none", ""
        rep = {k: cor[k].copy() for k in sp}

    bank = []
    if repair:
        for k in sp:
            for i in np.where(mask[k])[0]:
                bank.append(make_record(
                    dataset_hash=d_hash, split_hash=hashes[k], row_index=int(i),
                    column=CORRUPT_COL, method="p1p99_clip", source_split=k,
                    fit_split=label, fit_hash=fit_hash, code_version="v0.2",
                    old_value=float(cor[k].loc[i, CORRUPT_COL]),
                    new_value=float(rep[k].loc[i, CORRUPT_COL])))

    feats = [c for c in train.columns if c != TARGET]
    col_idx = feats.index(CORRUPT_COL)
    model = train_model(rep["train"][feats], rep["train"][TARGET])

    def replay(k):
        rows = []
        for i in np.where(mask[k])[0]:
            r = replay_one(model, rep[k][feats].iloc[[i]], col_idx,
                           float(rep[k].loc[i, CORRUPT_COL]),
                           float(cor[k].loc[i, CORRUPT_COL]),
                           y_true=sp[k].loc[i, TARGET])
            rows.append({"split": k, "row": int(i), **r})
        return rows

    return dict(sp=sp, cor=cor, rep=rep, mask=mask, bank=bank, bounds=bounds,
                hashes=hashes, dataset_hash=d_hash, model=model, feats=feats,
                col_idx=col_idx, replay=replay)
