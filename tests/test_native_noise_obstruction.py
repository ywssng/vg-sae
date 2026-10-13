"""Bounded artificial verification, to run only on Colab CPU."""
from fractions import Fraction as F
import itertools
import math

import pytest

pytest.importorskip("flint", reason="Optional certificate tests require python-flint")

from scripts.certify_native_noise_obstruction import (
    CENTER_SHIFT, DECODER, DIMENSION, EPSILON, SIGMA, Case,
    ball, certify_case, clean_class_lower_bound, dense_model_parameters,
    dense_second_moment, endpoint, exact, frozen_cases, good_class_witness,
    initialize_arithmetic, orthogonal_decoder_check, retain_budget_remainder,
    softplus_residual_second_moment_bound, strict_separation,
    validate_runtime, write_outputs,
)


@pytest.fixture(scope="module", autouse=True)
def exact_arithmetic_context():
    from flint import ctx
    old_precision, old_threads = ctx.prec, ctx.threads
    metadata = initialize_arithmetic()
    assert metadata["arb_precision_bits"] == 512 and metadata["flint_threads"] == 1
    yield
    ctx.prec, ctx.threads = old_precision, old_threads


def test_frozen_panel_has_twelve_cases_and_rejects_float_scientific_inputs():
    cases = frozen_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 12
    assert {(case.q, case.beta, case.gamma) for case in cases} == set(itertools.product(
        ("1/10", "3/10"), ("400", "1600", "25600"), ("1", "16")
    ))
    for bad in (0.1, True):
        with pytest.raises(ValueError, match="exact"):
            exact(bad)
    with pytest.raises(ValueError, match="frozen"):
        certify_case(Case("1/10", "800", "1"))


def test_rational_geometry_and_second_moment_against_four_support_states():
    assert orthogonal_decoder_check()
    assert sum(map(sum, DECODER)) == F(6, 5)
    for q in (F(1, 10), F(3, 10)):
        direct = F(0)
        for state in itertools.product((0, 1), repeat=2):
            probability = q**sum(state) * (1 - q) ** (2 - sum(state))
            rotated = [CENTER_SHIFT + sum(DECODER[i][j] * state[i] for i in range(2))
                       for j in range(2)]
            direct += probability * sum(value**2 for value in rotated)
        direct += DIMENSION * SIGMA**2
        assert dense_second_moment(q) == direct == 200 + 26 * q + F(1, 200)
    model = dense_model_parameters(Case("1/10", "400", "1"))
    assert model["parameters"]["pre_bias"] == ["2", "-14"]
    assert model["parameters"]["amplitude_encoder.weight"] == [["3/5", "4/5"], ["-4/5", "3/5"]]


def test_column_exponential_moment_factorization_matches_enumeration():
    q = F(3, 10)
    bound, columns, gaussian_shift = softplus_residual_second_moment_bound(q)
    enumerated = []
    for j in range(2):
        value = ball(0)
        for state in itertools.product((0, 1), repeat=2):
            probability = q**sum(state) * (1 - q) ** (2 - sum(state))
            exponent = -2 * sum(DECODER[i][j] * state[i] for i in range(2))
            value += ball(probability) * ball(exponent).exp()
        enumerated.append(value)
        assert columns[j].overlaps(value)
    assert bound.overlaps(gaussian_shift * sum(enumerated, ball(0)))


def test_stochastic_risk_identity_by_exact_mask_enumeration():
    # Independent polynomial test of the identity, including the full squared
    # risk convention and the orthogonal rotation.
    t, a, m = (F(1, 3), F(-2, 5)), (F(7, 4), F(9, 5)), F(3, 7)
    direct = F(0)
    for mask in itertools.product((0, 1), repeat=2):
        probability = m**sum(mask) * (1 - m) ** (2 - sum(mask))
        error = [t[j] - mask[j] * a[j] for j in range(2)]
        rotated = [sum(DECODER[i][j] * error[j] for j in range(2)) for i in range(2)]
        direct += probability * sum(value**2 for value in rotated)
    identity = (1 - m) * sum(value**2 for value in t) + m * sum((a[j] - t[j])**2 for j in range(2))
    assert direct == identity


def test_clean_class_lower_bound_and_nonempty_finite_model():
    assert EPSILON == F(1, 20000)
    case = Case("1/10", "1600", "1")
    good, certified = good_class_witness(case)
    assert certified and good["exponent_exact"] == "-50"
    assert F(good["clean_risk_upper_expression"]["upper"]) < EPSILON
    lower = clean_class_lower_bound(case.beta)
    formula = ball(F(case.beta)) / 2 * (
        ball(SIGMA) * ball(2).sqrt() - ball(EPSILON).sqrt()
    )**2
    assert endpoint(formula, upper=False) <= lower <= endpoint(formula, upper=True)
    assert lower == F(81, 40000) * F(case.beta)
    assert good["parameters"]["gate_encoder.weight"] == [["200", "0"], ["0", "200"]]


def test_strict_certificate_requires_separation_and_retains_matched_precision():
    assert not strict_separation("1", "1")
    assert not strict_separation("1001/1000", "1")
    assert strict_separation("999/1000", "1")
    matched = certify_case(Case("1/10", "400", "1"))
    assert matched["status"] == "bound_inconclusive"
    assert matched["beta_over_true_precision_exact"] == "1"
    overprecision = certify_case(Case("1/10", "25600", "16"))
    assert overprecision["status"] == "obstruction_certified"
    assert F(overprecision["bounds"]["objective_gap_certified_lower"]) > 0
    assert overprecision["checks"]["nonempty_good_class"]


def test_optional_native_free_energy_parity_on_artificial_inputs():
    # Supplemental float64 source-implementation check only. The theorem's
    # rational-weight certificate is not transferred to these rounded tensors.
    torch = pytest.importorskip("torch")
    from src.sae_model import VGSAEConfig, VariationalGarroteSAE
    with torch.device("meta"):
        model = VariationalGarroteSAE(VGSAEConfig(
            input_dim=2, n_latents=2, beta=1600, lambda_sparsity=1,
            beta_mode="learned", dtype=torch.float64,
        ))
    model.to_empty(device="cpu")
    with torch.no_grad():
        decoder = torch.tensor([[float(value) for value in row] for row in DECODER], dtype=torch.float64)
        model.decoder.weight.copy_(decoder)
        model.pre_bias.copy_(torch.tensor([2.0, -14.0], dtype=torch.float64))
        model.amplitude_encoder.weight.copy_(decoder.T)
        model.amplitude_encoder.bias.zero_()
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.fill_(32)
        model.log_beta.fill_(math.log(1600))
        inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float64)
        result = model.free_energy(inputs)
        t = (inputs - model.pre_bias) @ decoder
        amplitude = torch.nn.functional.softplus(t)
        gate = torch.sigmoid(torch.tensor(32.0, dtype=torch.float64))
        risk = ((1 - gate) * t.pow(2) + gate * (amplitude - t).pow(2)).sum(dim=1).mean()
        entropy = (torch.nn.functional.softplus(torch.tensor(-32.0, dtype=torch.float64))
                   + 32 * torch.sigmoid(torch.tensor(-32.0, dtype=torch.float64)))
        kl = gate + torch.nn.functional.softplus(torch.tensor(-1.0, dtype=torch.float64)) - entropy
        beta = model.log_beta.exp()
        direct = beta * risk / 2 - torch.log(beta / (2 * math.pi)) + 2 * kl
    assert result["loss"].item() == pytest.approx(direct.item(), abs=2e-9)


def test_runtime_and_existing_results_are_guarded_before_writing(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(RuntimeError, match="VG_SAE_COLAB_CPU_ONLY"):
        validate_runtime(output, environ={}, gpu_visible=False)
    with pytest.raises(RuntimeError, match="CPU-only"):
        validate_runtime(output, environ={"VG_SAE_COLAB_CPU_ONLY": "1"}, gpu_visible=True)
    assert not output.exists()
    output.mkdir()
    artifact = output / "native_noise_obstruction_results.json"
    original = b'{"status":"preserved failed attempt"}\n'
    artifact.write_bytes(original)
    with pytest.raises(FileExistsError, match="Preserve existing"):
        validate_runtime(output, environ={"VG_SAE_COLAB_CPU_ONLY": "1"}, gpu_visible=False)
    with pytest.raises(FileExistsError):
        write_outputs(output, {})
    assert artifact.read_bytes() == original


def test_budget_remainder_keeps_all_frozen_ids_without_numeric_results():
    first = frozen_cases()[0]
    rows = [{"case_id": first.case_id, "status": "completed-fixture"}]
    retain_budget_remainder(rows, "artificial timeout", interrupted=True)
    assert len(rows) == 12
    assert [row["case_id"] for row in rows] == [case.case_id for case in frozen_cases()]
    assert rows[1]["status"] == "budget_unresolved_interrupted"
    assert all(row["status"] == "budget_unresolved_not_started" for row in rows[2:])
    assert all(not row["numeric_result_available"] for row in rows[1:])
