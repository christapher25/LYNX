"""
tests/invariants/test_fno_feature_invariants.py

The Stage 10 invariants for the F&O block (features/blocks/fno.py):

  1. NO ZERO-FILL      a row flagged missing is NaN in every feature the flag covers; a real 0 stays 0.
  2. NO STALE CARRY    an outage day, or a session with no ban list, never inherits the previous day's OI or ban status.
  3. BAN: UNAVAILABLE  a session with no usable ban list is NaN + flagged, never "nobody banned" (0.0); a published NIL
     IS NOT NONE      list (available, symbol absent) is a real 0.0.
  4. CAUSALITY         truncating the future (or rewriting it) never changes an earlier row: no same-day-after / next-session
                       ban leak, no centred or forward-looking window.
  5. REFERENCE         agrees with a plain-loop recomputation on random worlds with outages, absences, splits and bans.

Each is a violation counter run on the real code (must be 0) and on a deliberately broken implementation of the failure it
exists to prevent (must be > 0), so a counter that cannot fire is itself caught.
"""

import numpy as np
import pandas as pd
import pytest

from features.blocks.fno import MIN_OBS_20D, compute_fno_features

SUMMARY = ["fut_oi_raw", "near_oi", "next_oi", "days_to_expiry", "put_oi", "call_oi"]
OI_FEATURES = ["fut_oi", "fut_oi_chg_1d", "fut_oi_chg_5d", "fut_oi_ratio_20d", "fut_roll_next_share", "fut_days_to_expiry"]
BAN_FEATURES = ["fno_in_ban", "fno_ban_new_entry", "fno_ban_days_20d"]
SEEDS = range(8)


def _world(seed: int, n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=n)})
    obs["volume_adj_factor"] = np.where(np.arange(n) < n // 2, 2.0, 1.0)             # a 2:1 split halfway
    obs["fno_day_available"] = rng.random(n) > 0.06                                  # archive gaps
    present = rng.random(n) > 0.15                                                   # symbol absent from an available day
    level = np.cumsum(rng.integers(-5, 8, n)) + 400.0
    obs["fut_oi_raw"] = np.where(present, np.maximum(level, 0.0) / obs["volume_adj_factor"], np.nan)
    near = np.where(present, np.maximum(level * 0.7, 0), np.nan)
    obs["near_oi"] = near
    obs["next_oi"] = np.where(present & (rng.random(n) > 0.1), np.maximum(level * 0.3, 0), np.nan)
    obs["days_to_expiry"] = np.where(present, rng.integers(0, 30, n), np.nan)
    obs["put_oi"] = np.where(present & (rng.random(n) > 0.2), rng.integers(1, 90, n), np.nan)
    obs["call_oi"] = np.where(present & (rng.random(n) > 0.2), rng.integers(0, 90, n), np.nan)
    obs["ban_available"] = rng.random(n) > 0.08
    obs["in_ban"] = rng.random(n) < 0.2                                              # may be True on an unavailable day: must be ignored
    return obs


def _nan_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return all(np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True) for c in a.columns if c != "symbol")


# ------------------------------------------------------------------ violation counters ------------------------------------------------

def zero_fill_violations(out: pd.DataFrame) -> int:
    """Cells that must be NaN because their flag says missing, but hold a number."""
    bad = out.loc[out["is_missing_fno"], OI_FEATURES].notna().to_numpy().sum()
    bad += out.loc[out["is_missing_opt"], "opt_pcr_oi"].notna().sum()
    bad += out.loc[out["is_missing_ban"], BAN_FEATURES[:2]].notna().to_numpy().sum()
    observed = (~out["is_missing_ban"]).astype(float).rolling(20, min_periods=1).sum()      # lists seen in the trailing 20 rows
    bad += out.loc[observed < MIN_OBS_20D, "fno_ban_days_20d"].notna().sum()               # too few lists: must be NaN, not a count
    return int(bad)


def stale_carry_violations(obs: pd.DataFrame, fn) -> int:
    """Rows on an outage day (no F&O file) or a no-list session that came out with a value."""
    out = fn(obs)
    outage = ~obs["fno_day_available"].to_numpy()
    no_list = ~obs["ban_available"].to_numpy()
    return int(out.loc[outage, OI_FEATURES].notna().to_numpy().sum() + out.loc[no_list, "fno_in_ban"].notna().sum())


def ban_unavailable_violations(obs: pd.DataFrame, fn) -> int:
    """Sessions with no usable list that are reported as a definite ban status (0.0 or 1.0)."""
    out = fn(obs)
    return int(out.loc[~obs["ban_available"].to_numpy(), "fno_in_ban"].notna().sum())


def causality_violations(obs: pd.DataFrame, fn, cuts=(20, 45, 70)) -> int:
    """Rows before each cut that change when everything from the cut onward is removed."""
    full = fn(obs)
    bad = 0
    for cut in cuts:
        head = fn(obs.iloc[:cut])
        for column in head.columns:
            if column in ("symbol", "date"):
                continue
            a, b = full[column].iloc[:cut].astype(float).to_numpy(), head[column].astype(float).to_numpy()
            bad += int((~np.isclose(a, b, equal_nan=True)).sum())
    return bad


def future_rewrite_violations(obs: pd.DataFrame, fn, cut: int = 50) -> int:
    """Rows before the cut that change when every input dated at or after the cut is scrambled."""
    scrambled = obs.copy()
    rng = np.random.default_rng(1)
    for column in SUMMARY + ["in_ban"]:
        values = scrambled[column].to_numpy(copy=True)
        values[cut:] = rng.permutation(values[cut:])
        scrambled[column] = values
    flipped = not (bool(obs["ban_available"].iloc[cut]) and bool(obs["in_ban"].iloc[cut]))
    scrambled.loc[scrambled.index[cut:], "ban_available"] = True
    scrambled.loc[scrambled.index[cut], "in_ban"] = flipped            # the first future list certainly differs from the real one
    before, after = fn(obs).iloc[:cut], fn(scrambled).iloc[:cut]
    return int(sum((~np.isclose(before[c].astype(float), after[c].astype(float), equal_nan=True)).sum()
                   for c in before.columns if c not in ("symbol", "date")))


# ------------------------------------------------------------------ deliberately broken implementations ------------------------------

def broken_zero_fill(obs: pd.DataFrame) -> pd.DataFrame:
    out = compute_fno_features(obs)
    return out.assign(**{c: out[c].fillna(0.0) for c in OI_FEATURES + ["opt_pcr_oi", "fno_in_ban", "fno_ban_new_entry", "fno_ban_days_20d"]})


def broken_stale_carry(obs: pd.DataFrame) -> pd.DataFrame:
    """Treats an outage day / no-list session as available and carries the last known values forward."""
    carried = obs.copy()
    carried[SUMMARY] = carried[SUMMARY].ffill()
    carried["in_ban"] = carried["in_ban"].where(carried["ban_available"]).ffill().fillna(False).astype(bool)
    carried["fno_day_available"] = True
    carried["ban_available"] = True
    return compute_fno_features(carried)


def broken_ban_none_when_unavailable(obs: pd.DataFrame) -> pd.DataFrame:
    return compute_fno_features(obs.assign(ban_available=True, in_ban=obs["in_ban"] & obs["ban_available"]))


def broken_next_session_ban(obs: pd.DataFrame) -> pd.DataFrame:
    """Row t carries the NEXT session's ban status (a list not yet public at t)."""
    out = compute_fno_features(obs)
    out["fno_in_ban"] = out["fno_in_ban"].shift(-1)
    return out


def broken_centered_window(obs: pd.DataFrame) -> pd.DataFrame:
    out = compute_fno_features(obs)
    out["fut_oi_ratio_20d"] = out["fut_oi"] / out["fut_oi"].rolling(20, min_periods=MIN_OBS_20D, center=True).mean()
    return out


def broken_unadjusted(obs: pd.DataFrame) -> pd.DataFrame:
    return compute_fno_features(obs.assign(volume_adj_factor=1.0))


# ------------------------------------------------------------------ plain-loop reference ----------------------------------------------

def reference(obs: pd.DataFrame) -> pd.DataFrame:
    n = len(obs)
    rows = obs.to_dict("records")
    oi = [np.nan] * n
    usable = [False] * n
    for i, r in enumerate(rows):
        usable[i] = bool(r["fno_day_available"]) and not np.isnan(r["fut_oi_raw"]) and r["volume_adj_factor"] > 0
        if usable[i]:
            oi[i] = r["fut_oi_raw"] * r["volume_adj_factor"]
    ban = [float(r["in_ban"]) if r["ban_available"] else np.nan for r in rows]

    def change(i, k):
        if i < k or np.isnan(oi[i]) or np.isnan(oi[i - k]) or not oi[i - k] > 0:
            return np.nan
        return oi[i] / oi[i - k] - 1

    def window(values, i):
        seen = [v for v in values[max(0, i - 19): i + 1] if not np.isnan(v)]
        return seen if len(seen) >= MIN_OBS_20D else None

    out = []
    for i, r in enumerate(rows):
        seen = window(oi, i)
        mean = np.mean(seen) if seen else np.nan
        ratio = oi[i] / mean if seen and mean > 0 and not np.isnan(oi[i]) else np.nan
        denom = r["near_oi"] + r["next_oi"]
        roll = r["next_oi"] / denom if usable[i] and denom > 0 else np.nan             # NaN denom compares False
        dte = r["days_to_expiry"] if usable[i] else np.nan
        opt_ok = bool(r["fno_day_available"]) and not np.isnan(r["put_oi"]) and not np.isnan(r["call_oi"])
        pcr = r["put_oi"] / r["call_oi"] if opt_ok and r["call_oi"] > 0 else np.nan
        entry = np.nan
        if i >= 1 and not np.isnan(ban[i]) and not np.isnan(ban[i - 1]):
            entry = float(ban[i] == 1.0 and ban[i - 1] == 0.0)
        bseen = window(ban, i)
        out.append({"fut_oi": oi[i], "fut_oi_chg_1d": change(i, 1), "fut_oi_chg_5d": change(i, 5), "fut_oi_ratio_20d": ratio,
                    "fut_roll_next_share": roll, "fut_days_to_expiry": dte, "opt_pcr_oi": pcr, "fno_in_ban": ban[i],
                    "fno_ban_new_entry": entry, "fno_ban_days_20d": float(np.sum(bseen)) if bseen else np.nan,
                    "is_missing_fno": not usable[i], "fno_day_unavailable": not bool(r["fno_day_available"]),
                    "is_missing_opt": not opt_ok, "is_missing_ban": not bool(r["ban_available"])})
    return pd.DataFrame(out)


def reference_violations(obs: pd.DataFrame, fn) -> int:
    out, ref = fn(obs).reset_index(drop=True), reference(obs)
    return int(sum((~np.isclose(out[c].astype(float), ref[c].astype(float), equal_nan=True)).sum() for c in ref.columns))


# ------------------------------------------------------------------ the invariants ------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
class TestInvariants:
    def test_no_zero_fill(self, seed):
        obs = _world(seed)
        assert zero_fill_violations(compute_fno_features(obs)) == 0
        assert zero_fill_violations(broken_zero_fill(obs)) > 0

    def test_no_stale_carry_forward(self, seed):
        obs = _world(seed)
        assert stale_carry_violations(obs, compute_fno_features) == 0
        assert stale_carry_violations(obs, broken_stale_carry) > 0

    def test_ban_unavailable_is_not_none(self, seed):
        obs = _world(seed)
        assert ban_unavailable_violations(obs, compute_fno_features) == 0
        assert ban_unavailable_violations(obs, broken_ban_none_when_unavailable) > 0

    def test_nil_list_is_a_real_zero(self, seed):
        obs = _world(seed).assign(ban_available=True, in_ban=False)
        out = compute_fno_features(obs)
        assert (out["fno_in_ban"] == 0.0).all() and not out["is_missing_ban"].any()

    def test_causality_by_truncation(self, seed):
        obs = _world(seed)
        assert causality_violations(obs, compute_fno_features) == 0
        assert causality_violations(obs, broken_next_session_ban) > 0
        assert causality_violations(obs, broken_centered_window) > 0

    def test_causality_by_rewriting_the_future(self, seed):
        obs = _world(seed)
        assert future_rewrite_violations(obs, compute_fno_features) == 0
        assert future_rewrite_violations(obs, broken_next_session_ban) > 0

    def test_agrees_with_plain_loop_reference(self, seed):
        obs = _world(seed)
        assert reference_violations(obs, compute_fno_features) == 0
        assert reference_violations(obs, broken_unadjusted) > 0              # the split makes the factor observable

    def test_split_does_not_register_as_positioning(self, seed):
        obs = _world(seed).assign(fno_day_available=True)
        n = len(obs)
        flat = obs.assign(fut_oi_raw=np.where(np.arange(n) < n // 2, 100.0, 200.0), ban_available=True)
        ex_date = n // 2
        assert compute_fno_features(flat)["fut_oi_chg_1d"].iloc[ex_date] == pytest.approx(0.0)
        assert broken_unadjusted(flat)["fut_oi_chg_1d"].iloc[ex_date] == pytest.approx(1.0)
