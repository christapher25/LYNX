"""VANTA gain builder (N10): gain scaling and winsorisation fitted on the training fold only."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINSOR_PCT = (2.0, 98.0)


def build_fold_gains(labels: pd.DataFrame, train_mask, horizon: int) -> pd.Series:
    """g = y_net / (atr_pct * sqrt(horizon)), clipped at the 2nd/98th percentile (numpy linear) of the TRAINING rows' gains.

    All rows (train and test) are clipped with the training bounds; test labels never influence them. Non-finite gains (e.g. atr_pct == 0) become NaN and stay NaN
    and are ignored when fitting the bounds (missing is missing).
    """
    mask = np.asarray(train_mask, bool)
    if len(mask) != len(labels):
        raise ValueError("train_mask must align positionally with labels")
    atr = labels["atr_pct"].where(labels["atr_pct"] > 0)          # zero / negative / NaN ATR is missing, never a gain
    g = labels["y_net"] / (atr * np.sqrt(horizon))
    g = g.where(np.isfinite(g))                                  # atr_pct == 0 / inf gains are missing, never a phantom extreme
    train = g[mask].to_numpy(float)
    train = train[np.isfinite(train)]
    if train.size == 0:
        raise ValueError("no finite training gains to fit winsorisation bounds")
    lo, hi = np.percentile(train, WINSOR_PCT)
    return g.clip(lo, hi)
