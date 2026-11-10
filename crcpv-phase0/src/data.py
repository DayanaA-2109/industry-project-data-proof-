import hashlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

TARGET = "target"
CORRUPT_COL = "worst texture"
CORRUPT_FRAC = 0.20
OUTLIER_MULT = 2.5
SEED = 42


def load_clean():
    df = load_breast_cancer(as_frame=True).frame
    return df.rename(columns={"target": TARGET})


def split(df, seed=SEED):
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df[TARGET], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[TARGET], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def corrupt(df, col=CORRUPT_COL, frac=CORRUPT_FRAC, mult=OUTLIER_MULT, seed=SEED):
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = rng.random(len(out)) < frac
    out.loc[mask, col] = out.loc[mask, col] * mult
    return out, mask


def hash_df(df):
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# Per-split offsets so each split gets an INDEPENDENT corruption RNG stream
# (previously all three splits reused one seed => identical masks).
SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}


def corrupt_split(df, split_name, seed=SEED, **kw):
    return corrupt(df, seed=seed * 10 + SPLIT_OFFSET[split_name], **kw)


def hash_indices(idx):
    return hashlib.sha256(
        ",".join(map(str, sorted(idx))).encode()
    ).hexdigest()
