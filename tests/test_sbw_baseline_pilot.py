"""Selection and no-truth amplitude controls for the paper-inspired P1 pilot."""
from dataclasses import FrozenInstanceError

import pytest
import torch

from scripts.run_sbw_baseline_pilot import fit_amplitude_rescaling, select_calibration


def candidate(name, l0, recovery, f1=.8, mse=.01):
    return {"candidate_id": name, "completed_updates": 500, "control_index": 0,
            "calibration": {"hard_l0": l0, "matched_positive_cosine": recovery,
                            "support_f1": f1, "hard_mse": mse, "c_dec": 1-recovery,
                            "mixing_energy": 1-recovery}}


def test_empty_l0_coverage_is_failure_instead_of_nearest_candidate():
    result = select_calibration([candidate("sparse", 1.0, 1.), candidate("dense", 3., 1.)], 2.)
    assert result.status == "coverage_failure"
    assert result.candidate_id is None


def test_selection_prioritizes_identity_over_mse_and_is_frozen():
    candidates = [candidate("correct", 2., .99, mse=.1), candidate("mixed", 2., .8, mse=.001)]
    result = select_calibration(candidates, 2.)
    assert result.candidate_id == "correct"
    candidates[0]["calibration"]["matched_positive_cosine"] = 0
    assert result.candidate_id == "correct"
    with pytest.raises(FrozenInstanceError):
        result.candidate_id = "mixed"


def test_deployment_selector_ignores_ground_truth_quality():
    a, b = candidate("a", 2., .9), candidate("b", 2., .8)
    a["calibration"]["c_dec"], b["calibration"]["c_dec"] = .4, .1
    selected = select_calibration([a, b], None, "deployment_c_dec")
    assert selected.candidate_id == "b"
    a["calibration"]["matched_positive_cosine"] = 1000
    a["calibration"]["support_f1"] = 1000
    assert select_calibration([a, b], None, "deployment_c_dec") == selected


def test_primary_matching_uses_calibration_density_not_identity():
    near = candidate("near", 2.02, .5)
    farther = candidate("farther", 1.90, 1.)
    assert select_calibration([near, farther], 2., "matched_l0").candidate_id == "near"


def test_amplitude_control_learns_input_fit_and_preserves_fixed_test_support():
    dictionary = torch.eye(2)
    calibration_code = torch.tensor([[.5, 0.], [0., 2.], [.5, 2.]])
    calibration_inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    gains = fit_amplitude_rescaling(calibration_code, dictionary, calibration_inputs)
    torch.testing.assert_close(gains, torch.tensor([2., .5]))
    test_code = torch.tensor([[0., 3.], [.7, 0.]])
    before = gains.clone()
    corrected = test_code * gains
    assert torch.equal(corrected > 0, test_code > 0)
    # Applying the frozen gains accepts no test observations or truth to refit.
    torch.testing.assert_close(gains, before)
    torch.testing.assert_close(corrected, torch.tensor([[0., 1.5], [1.4, 0.]]))
