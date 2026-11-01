"""
M2 driver (scripts/run_m2.py) and report invariants, from docs/vanta/M2_PREREG.md sections 4, 6, 8 and 9.
Module under test: scripts.run_m2 (does not exist yet; `build_parser()` and the vanta.m2 report functions are INVENTED names).

  M2-P1  frozen parameters   nothing the pre-registration fixes can be changed from the command line (no q, k, tau, cap, ESS floor, seeds,
                             window, liquidity cliff, arm list); only --horizon {5,20,60}, --impact {zero,floor}, --variant {atr,hold}
                             and bookkeeping options exist
  M2-R1  report header       the first paragraph states the status of the evidence (window seen in M1 and M1b, re-test, weaker than fresh
                             data, only the paper-trading log is fresh)
  M2-R2  report footer       realised FDR is an empirical number, not a guarantee; q = 0.30 an engineering target; the GO/NO-GO is the owner's

Each check takes a `parser` (real or toy) or an `api`; real ones skip until the module exists; the toy passes and every broken toy fails.
"""

import argparse

import pytest

from test_m2_toy import toy_api

FROZEN = {"q", "k", "k-s", "k-t", "ks", "kt", "tau", "tau-weeks", "cap", "ess-floor", "ess", "seed", "seeds", "window", "window-weeks",
          "liquidity-cliff", "cliff", "arms", "arm", "floor", "impact-floor", "oos-start", "first-judged-week", "gate", "folds"}


def toy_parser(*bugs):
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[5, 20] if "no_h60" in bugs else [5, 20, 60, 100] if "h_any" in bugs else [5, 20, 60])
    ap.add_argument("--impact", choices=["zero"] if "no_floor" in bugs else ["zero", "floor"], required=True)
    ap.add_argument("--variant", choices=["atr"] if "no_hold" in bugs else ["atr", "hold"], default="atr")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--end", default=None)
    for b, opt in (("opt_q", "--q"), ("opt_cap", "--cap"), ("opt_tau", "--tau-weeks"), ("opt_cliff", "--liquidity-cliff"),
                   ("opt_seed", "--seeds"), ("opt_k", "--k-s")):
        if b in bugs:
            ap.add_argument(opt, default=None)
    return ap


def _choices(parser, dest):
    for a in parser._actions:
        if a.dest == dest:
            return a.choices
    return None


def check_parser_exposes_only_horizon_impact_variant(parser):
    """M2-P1: --horizon accepts exactly 5, 20, 60; --impact exactly zero/floor; --variant exactly atr/hold (A11 hold-to-horizon next to
    ATR k = 1.5 is a diagnostic variant of the same run); no option can change anything the pre-registration fixes."""
    assert list(_choices(parser, "horizon")) == [5, 20, 60]
    assert list(_choices(parser, "impact")) == ["zero", "floor"]
    assert list(_choices(parser, "variant")) == ["atr", "hold"]
    opts = {s.lstrip("-") for a in parser._actions for s in a.option_strings}
    assert not (opts & FROZEN), sorted(opts & FROZEN)


PARSER_CHECKS = [(check_parser_exposes_only_horizon_impact_variant,
                  ["no_h60", "h_any", "no_floor", "no_hold", "opt_q", "opt_cap", "opt_tau", "opt_cliff", "opt_seed", "opt_k"])]


def test_run_m2_parser_real():
    mod = pytest.importorskip("scripts.run_m2")
    check_parser_exposes_only_horizon_impact_variant(mod.build_parser())


def test_run_m2_parser_toy_passes():
    check_parser_exposes_only_horizon_impact_variant(toy_parser())


@pytest.mark.parametrize("bug", PARSER_CHECKS[0][1])
def test_run_m2_parser_mutant_fails(bug):
    with pytest.raises(AssertionError):
        check_parser_exposes_only_horizon_impact_variant(toy_parser(bug))


# ----------------------------------------------------------------------------------------------------------------- report text
def check_report_header_states_status_of_evidence(api):
    """M2-R1 (M2_PREREG section 1, section 8): the first paragraph says the window was already seen in M1 and M1b, that M2 is a re-test
    on an inspected window and not fresh out-of-sample data, that a pass is weaker evidence, and that only forward paper trading is fresh."""
    t = api.report_header().lower()
    for needle in ("m1b", "already", "inspected", "weaker", "paper"):
        assert needle in t, needle
    assert "fresh" in t


def check_report_footer_is_honest_and_stops(api):
    """M2-R2: realised FDR is an empirical number, not a guarantee; q = 0.30 an engineering target; the report stops: the GO/NO-GO is the
    owner's (no verdict sentence)."""
    t = api.report_footer().lower()
    assert "empirical" in t and "not a guarantee" in t and "engineering target" in t and "owner" in t
    assert "is a guarantee" not in t and "verdict is go" not in t and "verdict: go" not in t


REPORT = {"header": (check_report_header_states_status_of_evidence, ["header_silent"]),
          "footer": (check_report_footer_is_honest_and_stops, ["footer_guarantee"])}


@pytest.mark.parametrize("name", list(REPORT))
def test_m2_report_real(name):
    api = pytest.importorskip("vanta.m2")
    REPORT[name][0](api)


@pytest.mark.parametrize("name", list(REPORT))
def test_m2_report_toy_passes(name):
    REPORT[name][0](toy_api())


@pytest.mark.parametrize("name,bug", [(n, b) for n in REPORT for b in REPORT[n][1]])
def test_m2_report_mutant_fails(name, bug):
    with pytest.raises(AssertionError):
        REPORT[name][0](toy_api(bug))
