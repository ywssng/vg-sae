"""Scientific bound prerequisites and exclusions, executed on Colab CPU."""
from fractions import Fraction as F
import math
import json

import pytest

ctx = pytest.importorskip("flint").ctx

from scripts.certify_finite_noise import (
    TAIL_ROUNDING_FLOOR, ball, certify_cell, endpoint, excludes,
    nonnegative_upper, remainder, validate_spec, variance_domain,
    fixed_panel, read_verified_terminal, sha, TERMINAL_VERSION,
    HISTORICAL_RUN_SEAL, HISTORICAL_SOURCE_MANIFEST,
)


def spec(d=2, lam="2"):
    return {"id": "fixture", "q0": "1/10", "d": d, "lambda": lam,
            "I": ["1/20", "1/4"]}


@pytest.fixture(autouse=True)
def precision():
    old = ctx.prec
    ctx.prec = 512
    yield
    ctx.prec = old


@pytest.mark.parametrize("d", [2, 10])
def test_compact_domain_uses_certified_small_ball_and_feasible_reference(d):
    domain = variance_domain(spec(d))
    m = d//2
    t, eta = F(domain["rho_squared"]), F(domain["eta"])
    assert 4*t**m/(2**m*math.factorial(m)) <= F(1, 2)
    assert eta == t/2
    assert F(domain["lower_cutoff_envelope_upper"]) < F(domain["reference_lower"])
    assert 0 < F(domain["v_minus"]) < eta/d < F(domain["v_plus"])


def test_d2_tail_matches_exact_exponential_radial_identity():
    s = spec()
    domain = variance_domain(s)
    check = remainder(s, F(1, 20), domain)
    assert check["applicable"]
    C, vm = -2*ball(F(1, 20)).log(), ball(F(domain["v_minus"]))
    # For d=2, P(R>T)=exp(-T^2/2) and E R^2 1_{R>T}=(T^2+2)exp(-T^2/2).
    t_squared = ball(25)
    exact_tail = (-t_squared/2).exp()*(C+(t_squared+2)/(2*vm))
    assert F(check["tail_upper"]) >= endpoint(exact_tail, upper=False)
    assert abs(float(F(check["tail_upper"])-endpoint(exact_tail, upper=False))) < 1e-100


def test_tail_is_rounded_up_instead_of_underflowing_to_zero():
    tiny = (-ball(10**20)).exp()
    assert nonnegative_upper(tiny) == TAIL_ROUNDING_FLOOR
    assert ball(TAIL_ROUNDING_FLOOR) > tiny


def test_large_noise_and_geometry_do_not_receive_a_certificate():
    check = remainder(spec(lam="16"), F(1, 5))
    assert not check["conditions"]["cross_group_separation_at_least_one"]
    assert not check["applicable"]
    assert "epsilon_upper" not in check
    with pytest.raises(ValueError, match="sigma > 0"):
        remainder(spec(), 0)


def test_unverified_input_cannot_be_promoted_by_plausible_objective_numbers():
    cell = {"spec": spec(), "status": "audit_unresolved", "feasible_lower": "10",
            "outside_upper": {"q_le_r": "0", "q_ge_q0": "0"}}
    result = certify_cell(cell)
    assert result["status"] == "audit_unresolved"
    assert result["sigma_grid"] == []
    assert result["sigma0_q_lt_q0"] is None


def test_strict_margin_and_separate_boundary_claims():
    assert excludes("1", "9/10", "1/100")
    assert not excludes("1", "9/10", "1/10")
    assert not excludes("1", None, "1/100")
    # Artificial certificate arithmetic fixture, not evidence about a real cell.
    cell = {"spec": spec(lam="0"), "status": "verified", "feasible_lower": "1",
            "outside_upper": {"q_le_r": None, "q_ge_q0": "999/1000"}}
    result = certify_cell(cell)
    assert result["sigma0_q_lt_q0"] is not None
    assert result["sigma0_q_gt_r"] is None
    assert result["sigma0_strict_bracket"] is None
    assert not result["at_original_noise_0_05"]["q_lt_q0_certified"]


@pytest.mark.parametrize("change", [{"d": 3}, {"lambda": "-1"}, {"I": ["1/5", "1/4"]}, {"q0": 0.1}])
def test_unsupported_or_inadmissible_model_is_rejected(change):
    with pytest.raises(ValueError):
        validate_spec({**spec(), **change})


def audit_fixture(tmp_path, change=None):
    source = {"version": TERMINAL_VERSION, "source_sha256": "a"*64,
              "historical_run_seal_sha256": HISTORICAL_RUN_SEAL,
              "source_manifest_sha256": HISTORICAL_SOURCE_MANIFEST,
              "denominator": 20, "completed_cells": 0, "passed": False,
              "cells": [{"spec": s, "status": "audit_unresolved"} for s in fixed_panel()]}
    if change:
        change(source)
    path, seal = tmp_path/"TERMINAL_AUDIT.json", tmp_path/"SEAL.json"
    path.write_text(json.dumps(source))
    prefix = tmp_path/"VERIFIED_PREFIX.jsonl"
    prefix.write_text("")
    seal.write_text(json.dumps({"version": TERMINAL_VERSION,
                               "files": {path.name: sha(path), prefix.name: sha(prefix)}}))
    return path, seal


def test_sealed_partial_audit_retains_unresolved_rows(tmp_path):
    path, seal = audit_fixture(tmp_path)
    result = read_verified_terminal(path, seal, "a"*64)
    assert len(result["cells"]) == 20
    assert not result["passed"]
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="seal"):
        read_verified_terminal(path, seal, "a"*64)


@pytest.mark.parametrize("change", [
    lambda x: x["cells"][0]["spec"].update(I=["1/20", "2/5"]),
    lambda x: x["cells"].reverse(),
    lambda x: x["cells"][0]["spec"].update(id="changed"),
    lambda x: x.update(source_sha256="b"*64),
    lambda x: x["cells"][0].update(status="verified"),
])
def test_resealed_but_different_contract_or_source_is_rejected(tmp_path, change):
    path, seal = audit_fixture(tmp_path, change)
    with pytest.raises(ValueError):
        read_verified_terminal(path, seal, "a"*64)
