import pandas as pd
from src.data import hash_df


def fit_clip_bounds(train_df, col, lo=0.01, hi=0.99):
    """Return (low, high) quantile bounds fitted on train_df."""
    return float(train_df[col].quantile(lo)), float(train_df[col].quantile(hi))


def fit_clip_bounds_tracked(named_frames, col, lo=0.01, hi=0.99):
    """Fit bounds AND report what data was actually used.

    named_frames: dict like {"train": df} or {"train": a, "val": b, "test": c}.
    Returns (bounds, fit_split_label, fit_hash). The label and hash are DERIVED
    from the data passed in, never typed by hand, so the ledger cannot claim
    'train' while the operator was fit on something else.
    """
    fit_df = pd.concat(list(named_frames.values()))
    bounds = fit_clip_bounds(fit_df, col, lo, hi)
    label = "+".join(named_frames.keys()) if len(named_frames) > 1 else next(iter(named_frames))
    if label == "train+val+test":
        label = "all"
    return bounds, label, hash_df(fit_df)


def apply_clip(df, col, bounds, mask=None):
    """Clip cells in `df[col]` to `bounds`. If mask is given, only clip masked rows."""
    out = df.copy()
    lo, hi = bounds
    if mask is None:
        out[col] = out[col].clip(lo, hi)
    else:
        out.loc[mask, col] = out.loc[mask, col].clip(lo, hi)
    return out
