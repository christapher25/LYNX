"""
Stage 13C synthetic proof: on EXCHANGEABLE data the RW-CS gate's realised FDR is <= q.

Calibration and candidates are iid draws from one joint (score, y_net) law (score ~ N(0,1), y = 0.5*score + N(0,1) noise scale
0.7 minus 0.1, so roughly half the candidates are nulls y <= 0). Weights are equal (tau huge, one regime): the exactness
setting of the conformal p-value. Realised FDR = mean over replications of V / max(R, 1) (zero-publish counts as 0), with its
standard error; q is an engineering target, so the check allows 2 standard errors of slack and says so. Two deliberately
broken variants must violate it:
  no_plus_one         p-values without the +1 in the numerator
  missing_losers      calibration holds only some of the losers (unmatured / survivor-biased outcomes treated as absent)
No learned model of any kind.

SCOPE (honest): this proof is for EQUAL weights on exchangeable data. The weighted age x regime gate has NO proven FDR control
here: weighted conformal p-values are exact only for true likelihood-ratio weights; q is a target, not a guarantee; with test
weight 1 against calibration weights below 1 the weighted p-values are conservative. Usage: python scripts/verify_stage13c.py [--reps 300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vanta import gate  # noqa: E402

MODEL, FILT = "m@x", "f@x"
T = pd.Timestamp("2024-06-03")
QS = (0.05, 0.1, 0.2)
N_CAL, M_CAND = 200, 40
N_CAL_SMALL = 20          # the +1 matters only when n is small: bias ~ 1/(1+n)


def _draw(rng, n):
    s = rng.normal(size=n)
    return s, 0.5 * s - 0.1 + 0.7 * rng.normal(size=n)


def _cal_frame(s, y):
    d = T - pd.Timedelta(weeks=1)
    n = len(s)
    return pd.DataFrame({"decision_date": d, "symbol": [f"C{i}" for i in range(n)], "horizon": 5, "score": s,
                         "model_version": MODEL, "fold_id": "f", "is_oos": True, "regime_posterior": [[1.0, 0.0, 0.0]] * n,
                         "hmm_version": "h", "y_net": y, "matured_at": d + pd.Timedelta(days=6),
                         "candidate_filter_version": FILT})


def one_rep(rng, q, variant, cap):
    n_cal = N_CAL_SMALL if variant == "no_plus_one" else N_CAL
    cs, cy = _draw(rng, n_cal)
    ts, ty = _draw(rng, M_CAND)
    if variant == "missing_losers":
        keep = (cy > 0) | (rng.random(n_cal) < 0.4)
        cs, cy = cs[keep], cy[keep]
    cal = _cal_frame(cs, cy)
    cands = pd.DataFrame({"symbol": [f"X{i}" for i in range(M_CAND)], "score": ts})
    kw = dict(model_version=MODEL, candidate_filter_version=FILT, tau_weeks=1e9, q=q, cap=cap, ess_floor=0)
    if variant == "no_plus_one":
        w = np.ones(len(cs))
        p = (gate.conformal_pvalues(cs, cy, w, ts) * (1 + w.sum()) - 1.0) / (1 + w.sum())
        pub = gate.bh_select(p, q)
    else:
        pub = gate.run_gate(cands, cal, T, (1.0, 0.0, 0.0), **kw)["published"].to_numpy()
    r = int(pub.sum())
    v = int((pub & (ty <= 0)).sum())
    return v / max(r, 1), r


def run(reps, variant, cap, seed):
    rows = {}
    for q in QS:
        rng = np.random.default_rng(seed)
        res = [one_rep(rng, q, variant, cap) for _ in range(reps)]
        fdp = np.array([a for a, _ in res])
        rows[q] = (fdp.mean(), fdp.std(ddof=1) / np.sqrt(reps), np.mean([b for _, b in res]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    ok = True
    for name, variant, cap in [("gate, cap=40 (BH set)", "good", M_CAND), ("gate, cap=10", "good", 10),
                               ("BROKEN no_plus_one", "no_plus_one", M_CAND), ("BROKEN missing_losers", "missing_losers", M_CAND)]:
        rows = run(a.reps, variant, cap, seed=20260101)
        print(f"\n{name}  (reps={a.reps}, n_cal={N_CAL_SMALL if variant == 'no_plus_one' else N_CAL}, m={M_CAND})")
        for q, (mean, se, nr) in rows.items():
            within = mean <= q + 2 * se
            verdict = "ok" if within else "VIOLATES"
            print(f"  q={q:<5} mean FDR={mean:.4f}  se={se:.4f}  mean published={nr:5.2f}  {verdict}")
            if variant == "good" and not within:
                ok = False
        if variant != "good":
            flagged = all(rows[q][0] > q + 2 * rows[q][1] for q in QS) if variant == "no_plus_one"                 else any(rows[q][0] > q + 2 * rows[q][1] for q in QS)
            print(f"  broken variant flagged: {flagged}")
            ok &= flagged
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
