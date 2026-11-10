from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model(X, y):
    """Smooth classifier so per-action deltas are observable (RF saturates)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model


def replay_one(model, row_df, col_idx, repaired_val, baseline_val, y_true=None):
    """Predict with the repaired vs the un-repaired cell. If y_true is given
    also return reward = change in probability of the TRUE class."""
    xr, xb = row_df.copy(), row_df.copy()
    xr.iloc[0, col_idx] = repaired_val
    xb.iloc[0, col_idx] = baseline_val
    pr = float(model.predict_proba(xr)[0, 1])
    pb = float(model.predict_proba(xb)[0, 1])
    out = {"p_repaired": pr, "p_baseline": pb, "delta": pr - pb}
    if y_true is not None:
        out["reward"] = (pr - pb) if int(y_true) == 1 else (pb - pr)
    return out
