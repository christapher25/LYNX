"""
Stage 13B report: filtered regime posterior on the real index/macro data.

Prints the first available date, state occupancy (argmax of the filtered posterior) by year, and the max abs
difference between the filtered posterior at t on the full series vs the series truncated at t for ~20 sampled
dates (must be 0). The HMM is fitted once on rows <= --train-end (default 2019-12-31) as an illustration; the
real per-fold fits end at fold test start minus embargo (vanta.regime.fold_train_end).

    python scripts/report_regime_posterior.py [--train-end 2019-12-31]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vanta import regime as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", default="2019-12-31")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    feats = R.build_regime_features()
    logger.info("features {} rows {} -> {}; non-null share: {}", len(feats), feats.index[0].date(),
                feats.index[-1].date(), feats.notna().mean().round(3).to_dict())
    params = R.fit_hmm(feats, train_end=args.train_end)
    logger.info("fit on {} rows <= {}; state means (log VIX z): {}", params.n_train, args.train_end,
                params.means[:, 0].round(2).tolist())
    logger.info("transmat diag: {}", np.diag(params.transmat).round(3).tolist())

    post = R.filtered_posterior_frame(params, feats)
    first = R.first_available(feats)
    logger.info("first VIX {}; first available posterior {}", feats[R.FEATURES[0]].first_valid_index().date(),
                first.date())

    avail = post.dropna()
    occ = (pd.Series(avail.to_numpy().argmax(axis=1), index=avail.index)
           .groupby(avail.index.year).value_counts(normalize=True).unstack(fill_value=0).round(3))
    logger.info("state occupancy by year (0 calm .. 2 stressed):\n{}", occ.to_string())

    rng = np.random.default_rng(0)
    pos = rng.choice(np.arange(len(feats))[feats.index >= first], size=args.samples, replace=False)
    full = R.filter_posterior(params, feats)
    gap = max(float(np.abs(full[t] - R.filter_posterior(params, feats.iloc[: t + 1])[t]).max()) for t in pos)
    logger.info("max abs diff, full-series vs truncated-at-t filtered posterior over {} dates: {}", len(pos), gap)
    return 0 if gap == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
