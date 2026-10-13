"""Regression challenges for the independent terminal-partition verifier.

Run on the authorized Colab CPU runtime; no historical scientific run is needed
for these artificial certificate and analytic-contract fixtures.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json

import pytest

from scripts import verify_population_terminal as audit


def artificial_certificate(ident, lo, hi, upper, lower):
    return {"id": ident, "domain": [str(lo), str(hi)],
            "upper": str(upper), "monotone_upper": str(upper), "mean_value_upper": str(upper),
            "derivative": ["-9", "9"], "witness_c": str((F(lo) + F(hi)) / 2),
            "witness_lower": str(lower), "q": ["1/20", "2/5"], "v": ["1/2", "2"],
            "variance_error": ["-1/2", "1"], "precision_bits": 128}


def artificial_trace(left_upper="1/2"):
    spec = {"id": "q1_10_d2_l2", "q0": "1/10", "d": 2, "lambda": "2", "I": ["1/100", "2/5"]}
    root = artificial_certificate(0, "1", "3", "5", "0")
    left = artificial_certificate(1, "1", "2", left_upper, "0")
    right = artificial_certificate(2, "2", "3", "3", "1")
    events = [
        {"type": "input", "spec": spec, "root": root, "tail_upper": "3",
         "tail_bound": ["5/2", "3"], "incumbent_lower": "0"},
        {"type": "split", "parent": {"id": 0, "domain": ["1", "3"]},
         "children": [left, right], "incumbent_lower": "1"},
        {"type": "reject", "box": deepcopy(left), "incumbent_lower": "1"},
        {"type": "remaining", "active": [deepcopy(right)], "incumbent_lower": "1", "termination": "cell_cpu_cap"},
    ]
    row = {"spec": spec, "tail_upper": "3", "tail_bound": ["5/2", "3"],
           "subdivisions": 1, "rejections": 1, "retained_leaves": 1,
           "global_objective": ["1", "3"], "termination": "cell_cpu_cap"}
    return events, row


def write_trace(tmp_path, events, row):
    payload = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events).encode()
    path = tmp_path / "fixture.trace.jsonl"
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    row["trace"] = {"events": len(events), "sha256": sha}
    return path, sha


def test_terminal_partition_covers_root_including_historical_rejections(tmp_path):
    events, row = artificial_trace()
    path, sha = write_trace(tmp_path, events, row)
    result = audit.parse_partition(path, row, sha)
    assert result.root == (F(1), F(3))
    assert [cert["id"] for cert in result.active] == [2]
    assert [cert["id"] for cert in result.rejected] == [1]
    assert result.witness_candidates[-1] == F(5, 2)
    assert result.incumbent == 1


@pytest.mark.parametrize("field,value", [
    ("upper", "4"), ("q", ["1/10", "1/5"]), ("v", ["1", "2"]),
    ("variance_error", ["0", "1"]), ("derivative", ["-1", "1"]),
    ("witness_lower", "2"), ("precision_bits", 256),
])
def test_terminal_same_id_and_domain_cannot_hide_changed_fields(tmp_path, field, value):
    events, row = artificial_trace()
    events[-1]["active"][0][field] = value
    path, sha = write_trace(tmp_path, events, row)
    with pytest.raises(audit.CertificateError, match="terminal certificate fields"):
        audit.parse_partition(path, row, sha)


def test_final_field_binding_uses_latest_precision_refresh(tmp_path):
    events, row = artificial_trace()
    refreshed = deepcopy(events[-1]["active"][0])
    refreshed["precision_bits"] = 256
    refreshed["upper"] = refreshed["monotone_upper"] = refreshed["mean_value_upper"] = "5/2"
    events.insert(-1, {"type": "precision_raise", "active": [refreshed],
                       "precision_bits": 256, "incumbent_lower": "1"})
    path, sha = write_trace(tmp_path, events, row)
    with pytest.raises(audit.CertificateError, match="terminal certificate fields"):
        audit.parse_partition(path, row, sha)
    events[-1]["active"] = [deepcopy(refreshed)]
    path, sha = write_trace(tmp_path, events, row)
    assert audit.parse_partition(path, row, sha).events == 5


def test_nonstrict_objective_tie_is_not_a_valid_rejection(tmp_path):
    events, row = artificial_trace(left_upper="1")
    path, sha = write_trace(tmp_path, events, row)
    with pytest.raises(audit.CertificateError, match="nonstrict recorded rejection"):
        audit.parse_partition(path, row, sha)


def test_split_cannot_skip_an_interval(tmp_path):
    events, row = artificial_trace()
    events[1]["children"][1]["domain"][0] = "9/4"
    path, sha = write_trace(tmp_path, events, row)
    with pytest.raises(audit.CertificateError, match="exactly cover"):
        audit.parse_partition(path, row, sha)


def test_no_unrecorded_leaf_disappearance(tmp_path):
    events, row = artificial_trace()
    del events[2]
    path, sha = write_trace(tmp_path, events, row)
    with pytest.raises(audit.CertificateError, match="terminal leaf set"):
        audit.parse_partition(path, row, sha)


def test_trace_hash_is_checked_even_when_geometry_is_valid(tmp_path):
    events, row = artificial_trace()
    path, _ = write_trace(tmp_path, events, row)
    with pytest.raises(audit.CertificateError, match="trace SHA"):
        audit.parse_partition(path, row, "0" * 64)


@pytest.mark.parametrize("domains", [
    [("1", "3/2"), ("2", "3")],
    [("1", "5/2"), ("2", "3")],
    [("1", "2")],
])
def test_exact_cover_rejects_gaps_overlaps_and_missing_tail(domains):
    certificates = [artificial_certificate(i, lo, hi, "3", "0") for i, (lo, hi) in enumerate(domains)]
    with pytest.raises(audit.CertificateError):
        audit.check_exact_cover((F(1), F(3)), certificates)


def test_fixed_panel_contains_all_twenty_declared_cases():
    panel = audit.expected_panel()
    assert len(panel) == len({spec["id"] for spec in panel}) == 20
    assert sum(spec["lambda"] == "0" for spec in panel) == 4
    assert {(spec["q0"], spec["d"]) for spec in panel} == {
        ("1/10", 2), ("1/10", 10), ("3/10", 2), ("3/10", 10)}


def test_documented_parent_receipt_fields_do_not_break_cell_binding():
    cell = {"spec": {"id": "fixture"}, "global_objective": ["1", "2"]}
    row = dict(cell, parent_all_in_cell_cpu_seconds=2.5, exit_code=0)
    audit.validate_summary_row(row, cell)


def test_parent_metadata_exception_does_not_hide_changed_scientific_value():
    cell = {"spec": {"id": "fixture"}, "global_objective": ["1", "2"]}
    row = dict(cell, parent_all_in_cell_cpu_seconds=2.5)
    row["global_objective"] = ["2", "2"]
    with pytest.raises(audit.CertificateError, match="summary/cell result"):
        audit.validate_summary_row(row, cell)


def test_unknown_summary_only_field_is_not_silently_ignored():
    with pytest.raises(audit.CertificateError, match="unexpected summary-only"):
        audit.validate_summary_row({"value": 1, "new_scientific_claim": True}, {"value": 1})


@pytest.fixture
def backend():
    pytest.importorskip("flint")
    return audit.load_backend()


def anchor(q0="1/10", d=2):
    q = F(q0)
    r = q * (1 - q)
    values = {"q": r, "v": F(1), "variance_error": F(0), "q_minus_r": F(0),
              "q0_minus_q": q * q, "relative_activity_bias": q}
    return {"spec": {"q0": q0, "d": d, "lambda": "0", "I": ["1/100", "2/5"], "id": "fixture"},
            "status": "exact_analytic_anchor", "objective_gap_upper": "0",
            "components": [{key: [str(value), str(value)] for key, value in values.items()}]}


@pytest.mark.parametrize("q0,d", [("1/10", 2), ("1/10", 10), ("3/10", 2), ("3/10", 10)])
def test_anchor_has_strict_upper_exclusion_but_no_strict_lower_claim(backend, q0, d):
    result = audit.verify_anchor(anchor(q0, d))
    assert result["status"] == "verified"
    assert result["outside_upper"]["q_le_r"] is None
    assert F(result["feasible_lower"]) > F(result["outside_upper"]["q_ge_q0"])
    assert result["strict_lower_activity_margin_claimed"] is False


def test_anchor_variance_must_equal_one(backend):
    row = anchor()
    row["components"][0]["v"] = ["1", "1001/1000"]
    with pytest.raises(audit.CertificateError, match="anchor v"):
        audit.verify_anchor(row)


@pytest.mark.parametrize("spec,domain", [
    ({"q0": "1/10", "d": 2, "lambda": "2", "I": ["1/100", "2/5"]}, ["1", "4"]),
    ({"q0": "3/10", "d": 2, "lambda": "1/2", "I": ["1/100", "2/5"]}, ["1/4", "3/4"]),
    ({"q0": "3/10", "d": 10, "lambda": "16", "I": ["1/100", "2/5"]}, ["8", "9"]),
])
def test_box_and_restricted_objective_bounds_cover_fixed_points(backend, spec, domain):
    """Challenges unclipped, clipping-transition and tiny-tail regimes."""
    profile = audit.IndependentProfile(spec)
    box = profile.box(domain)
    lower_penalty, upper_penalty = profile.outside_penalties(box["u"])
    full_upper = audit.fraction(box["upper"])
    lo, hi = map(F, domain)
    for i in range(9):
        point = profile.point(lo + (hi - lo) * i / 8)
        assert audit.fraction(point["P"].hi) <= full_upper
        for key in ("u", "q", "v", "variance_error"):
            assert box[key].lo <= point[key].lo <= point[key].hi <= box[key].hi
        residual = F(profile.d, 2) * point["v"].log() + F(profile.d, 2)
        restricted_low = profile.count_objective(point["u"], profile.r) - residual
        # Maximizing over q in [q0,beta] is a clipped count update.
        q_high = audit.clip(point["u"], profile.q0, profile.beta)
        restricted_high = profile.count_objective(point["u"], q_high) - residual
        assert audit.fraction(restricted_low.hi) <= full_upper - lower_penalty
        assert audit.fraction(restricted_high.hi) <= full_upper - upper_penalty


def test_upper_penalty_vanishes_when_leaf_includes_true_activity(backend):
    profile = audit.IndependentProfile({"q0": "3/10", "d": 2, "lambda": "2", "I": ["1/100", "2/5"]})
    u = audit.Interval(audit.ball(F(1, 4)), audit.ball(F(3, 8)))
    _, upper_penalty = profile.outside_penalties(u)
    assert upper_penalty == 0


def test_lower_penalty_accounts_for_upper_activity_clipping(backend):
    profile = audit.IndependentProfile({"q0": "3/10", "d": 2, "lambda": "1/2", "I": ["1/100", "2/5"]})
    u = audit.Interval(audit.ball(F(7, 16)), audit.ball(F(15, 32)))
    lower_penalty, _ = profile.outside_penalties(u)
    assert lower_penalty > 0
    value = audit.interval(F(7, 16))
    # Independent closed form: q_I=beta here, not q_I=u.
    expected = 2 * (value * audit.interval(profile.beta / profile.r).log()
                    + (1 - value) * audit.interval((1 - profile.beta) / (1 - profile.r)).log())
    assert lower_penalty <= audit.fraction(expected.hi)
    assert audit.fraction(expected.hi) - lower_penalty < F(1, 10**100)


def test_rounded_down_stored_upper_cannot_pass_independent_arithmetic(backend):
    profile = audit.IndependentProfile({"q0": "1/10", "d": 2, "lambda": "2", "I": ["1/100", "2/5"]})
    box = profile.box(["1", "2"])
    cert = artificial_certificate(0, "1", "2", "0", "-100")
    cert.update({key: list(map(str, box[key].endpoints())) for key in ("q", "v", "variance_error", "derivative")})
    cert.update({key: str(audit.fraction(box[key])) for key in ("upper", "monotone_upper", "mean_value_upper")})
    cert["witness_lower"] = str(audit.fraction(box["witness"]))
    cert["upper"] = str(F(cert["upper"]) - F(1, 1000))
    with pytest.raises(audit.ArithmeticUnresolved, match="stored upper"):
        audit.validate_leaf(profile, cert)


def broad_original_component(domain, old_q):
    return {"threshold": domain, "leaf_ids": [0], "raw_q": ["0", "1/2"],
            "v": ["0", "100"], "variance_error": ["-100", "100"],
            "raw_q_minus_r": ["-1", "1"], "raw_q0_minus_u": ["-1", "1"],
            "q": old_q, "q_minus_r": ["0", "1/100"], "q0_minus_q": ["0", "1/100"],
            "relative_activity_bias": ["0", "1/10"],
            "activity_width_target_met": False, "variance_width_target_met": False,
            "theorem_intersection_empty": old_q is None}


def test_old_wide_optimizer_image_may_be_independently_excluded(backend):
    profile = audit.IndependentProfile({"q0": "1/10", "d": 2, "lambda": "2", "I": ["1/100", "2/5"]})
    domain = ["1", "5/4"]  # Count update is wholly above q0 here.
    row = {"status": "certified_outer_unresolved",
           "components": [broad_original_component(domain, ["9/100", "1/10"])]}
    active = [{"id": 0, "domain": domain}]
    assert audit.summarize_components(profile, active)[0]["theorem_intersection_empty"]
    audit.validate_original_components(profile, row, active)


def test_old_empty_optimizer_image_must_remain_independently_empty(backend):
    profile = audit.IndependentProfile({"q0": "1/10", "d": 2, "lambda": "2", "I": ["1/100", "2/5"]})
    domain = ["3", "4"]  # Count updates intersect (r,q0).
    row = {"status": "certified_outer_unresolved", "components": [broad_original_component(domain, None)]}
    active = [{"id": 0, "domain": domain}]
    assert not audit.summarize_components(profile, active)[0]["theorem_intersection_empty"]
    with pytest.raises(audit.CertificateError, match="unproved theorem-empty"):
        audit.validate_original_components(profile, row, active)


def test_entrypoint_refuses_host_execution_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.delenv("VG_SAE_COLAB_CPU_ONLY", raising=False)
    destination = tmp_path / "must-not-be-created"
    with pytest.raises(audit.CertificateError, match="Colab CPU runtime marker"):
        audit.main(["--input-dir", str(tmp_path), "--output-dir", str(destination)])
    assert not destination.exists()
