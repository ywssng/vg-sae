"""Artificial deterministic checks; execute on Colab CPU, not the host."""

from decimal import Decimal, localcontext
import itertools
import math

import pytest
import torch

from scripts.check_native_bias_witness import (
    DECIMAL_PRECISION, SUPPORTS, WitnessCase, analytic_design,
    construct_native_model, evaluate_case, evaluate_native_population,
    frozen_cases, population_weights, profile_derivative, profile_value,
    retain_budget_remainder, state_as_json, tensor_population_reference,
    validate_runtime, _write_outputs,
)


def test_frozen_population_panel_has_all_cross_product_cases():
    cases = frozen_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 16
    assert {(case.q, case.beta, case.gamma) for case in cases} == set(itertools.product(
        ("0.1", "0.3"), ("1", "4", "40", "400"), ("1", "16")
    ))
    weights = population_weights(Decimal("0.1"))
    assert weights == (Decimal("0.81"), Decimal("0.09"), Decimal("0.09"), Decimal("0.01"))
    assert sum(weights) == 1
    assert sum(probability * sum(state) for probability, state in zip(weights, SUPPORTS)) == Decimal("0.2")


def test_native_parameter_witness_matches_two_target_values_without_rng_draws():
    case = WitnessCase("0.3", "4", "1")
    design = analytic_design(case)
    rng_before = torch.random.get_rng_state().clone()
    model = construct_native_model(case, design["candidate_bias"], design["inactive_amplitude_floor"])
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.equal(model.decoder.weight, torch.eye(2, dtype=torch.float64))
    assert all(parameter.device.type == "cpu" and not parameter.requires_grad
               for parameter in model.parameters())
    inputs = torch.tensor(SUPPORTS, dtype=torch.float64)
    m, a, _ = model.encode(inputs)
    b = float(design["candidate_bias"])
    floor = float(design["inactive_amplitude_floor"])
    target_a = inputs * (1 - b) + (1 - inputs) * floor
    target_logits = inputs * (4 * (1 - b) ** 2 / 2 - 1) - (1 - inputs)
    assert torch.allclose(a, target_a, atol=3e-15, rtol=3e-14)
    assert torch.allclose(m, torch.sigmoid(target_logits), atol=3e-15, rtol=3e-14)


def test_native_free_energy_matches_direct_bernoulli_mask_enumeration():
    # Directly enumerate stochastic support draws instead of using the variance
    # identity that is implemented by free_energy.
    case = WitnessCase("0.1", "4", "1")
    design = analytic_design(case)
    model = construct_native_model(case, design["candidate_bias"], design["inactive_amplitude_floor"])
    native = evaluate_native_population(model, Decimal(case.q))
    exact_sum = 0.0
    beta = model.log_beta.exp().item()
    prior_pi = 1 / (1 + math.exp(float(case.gamma)))
    for state, weight in zip(SUPPORTS, population_weights(Decimal(case.q))):
        x = torch.tensor([state], dtype=torch.float64)
        m, a, _ = model.encode(x)
        probabilities = m[0].tolist()
        stochastic_squared_error = 0.0
        for mask in SUPPORTS:
            mass = math.prod(probabilities[j] if mask[j] else 1 - probabilities[j]
                             for j in range(2))
            z = a * torch.tensor([mask], dtype=torch.float64)
            reconstruction = model.decode(z)
            stochastic_squared_error += mass * (x - reconstruction).pow(2).sum().item()
        kl = sum(p * math.log(p / prior_pi)
                 + (1 - p) * math.log((1 - p) / (1 - prior_pi))
                 for p in probabilities)
        value = beta * stochastic_squared_error / 2 - math.log(beta / (2 * math.pi)) + kl
        exact_sum += float(weight) * value
    assert native["objective"] == pytest.approx(exact_sum, abs=2e-13)
    reference = tensor_population_reference(state_as_json(model), Decimal(case.q), Decimal(case.gamma))
    assert native["objective"] == pytest.approx(float(reference["objective"]), abs=2e-13)


def test_analytic_choice_and_finite_floor_preserve_descent():
    case = WitnessCase("0.1", "4", "1")
    design = analytic_design(case)
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        q, beta, gamma = map(Decimal, (case.q, case.beta, case.gamma))
        assert profile_derivative(q, beta, gamma, Decimal(0)) < 0
        assert profile_derivative(q, beta, gamma, q) > 0
        assert design["closure_delta_scalar"] >= design["closure_delta_lower_scalar"] > 0
        assert design["finite_delta_scalar"] >= design["finite_delta_lower_scalar"] > 0
        assert design["finite_candidate_closure_defect_scalar"] > design["finite_baseline_closure_defect_scalar"]
        step = Decimal("1e-30")
        b = design["candidate_bias"]
        numeric_derivative = (profile_value(q, beta, gamma, b + step)
                              - profile_value(q, beta, gamma, b - step)) / (2 * step)
        assert abs(numeric_derivative - profile_derivative(q, beta, gamma, b)) < Decimal("1e-50")
    row = evaluate_case(case)
    assert row["status"] == "finite_native_decrease_resolved_float64"
    assert row["population_objective_difference_baseline_minus_candidate"]["native_minus_guard"] > 0


def test_exact_threshold_tie_is_not_an_active_native_gate():
    # Technical boundary fixture, not an additional research-panel case.
    case = WitnessCase("0.1", "2", "1")
    design = analytic_design(case)
    model = construct_native_model(case, Decimal(0), design["inactive_amplitude_floor"])
    native = evaluate_native_population(model, Decimal(case.q))
    assert design["all_off_entire_restricted_profile"]
    assert native["expected_hard_density"] == 0
    assert native["states"][-1]["gates"] == [0.5, 0.5]


def test_exponentially_small_positive_gap_is_retained_as_unresolved():
    row = evaluate_case(WitnessCase("0.1", "400", "16"))
    assert row["status"] == "unresolved_float64_difference"
    difference = row["population_objective_difference_baseline_minus_candidate"]
    assert Decimal(difference["ideal_finite_family_analytic_lower"]) > 0
    assert float(difference["ideal_finite_family_analytic_lower"]) < row["checks"]["float64_resolution_guard"]
    assert not difference["formal_numerical_certificate"]


def test_invalid_finite_witness_does_not_silently_clip_amplitudes():
    with pytest.raises(ValueError, match="floor"):
        construct_native_model(WitnessCase("0.1", "4", "1"), Decimal(0), Decimal(0))
    with pytest.raises(ValueError, match="bias"):
        profile_value(Decimal("0.1"), Decimal(4), Decimal(1), Decimal("1.1"))


def test_runtime_guard_rejects_missing_flag_and_cuda_before_any_output(tmp_path):
    output = tmp_path / "new-output"
    for environment in ({}, {"VG_SAE_COLAB_CPU_ONLY": "0"}):
        with pytest.raises(RuntimeError, match="VG_SAE_COLAB_CPU_ONLY"):
            validate_runtime(output, environ=environment, cuda_available=False)
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_runtime(output, environ={"VG_SAE_COLAB_CPU_ONLY": "1"},
                         cuda_available=True)
    assert not output.exists()
    validate_runtime(output, environ={"VG_SAE_COLAB_CPU_ONLY": "1"},
                     cuda_available=False)
    assert not output.exists()


def test_existing_attempt_is_preserved_by_preflight_and_exclusive_writer(tmp_path):
    artifact = tmp_path / "native_bias_witness_results.json"
    original = b'{"status":"previous-failed-attempt"}\n'
    artifact.write_bytes(original)
    with pytest.raises(FileExistsError, match="Preserve existing attempt"):
        validate_runtime(tmp_path, environ={"VG_SAE_COLAB_CPU_ONLY": "1"},
                         cuda_available=False)
    # The writer also refuses overwrite if a file appeared after preflight.
    with pytest.raises(FileExistsError):
        _write_outputs(tmp_path, {})
    assert artifact.read_bytes() == original
    assert not (tmp_path / "native_bias_witness_summary.csv").exists()


def test_budget_termination_retains_every_unexecuted_frozen_case():
    first = frozen_cases()[0]
    rows = [{"case_id": first.case_id, "status": "completed-fixture"}]
    retain_budget_remainder(rows, "artificial timeout", interrupted_case=True)
    assert len(rows) == 16
    assert [row["case_id"] for row in rows] == [case.case_id for case in frozen_cases()]
    assert rows[0]["status"] == "completed-fixture"
    assert rows[1]["status"] == "budget_unresolved_interrupted"
    assert all(row["status"] == "budget_unresolved_not_started" for row in rows[2:])
    assert all(not row["numeric_result_available"] for row in rows[1:])
