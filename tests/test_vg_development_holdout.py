"""Protocol-level safeguards for fresh holdout selection of frozen SAEs."""
import numpy as np
import torch

from scripts.run_vg_development_holdout import (
    resolve_l1_threshold, sample_fixed_dictionary, select_calibration_frontier,
)


def test_fresh_examples_keep_original_dictionary_and_separate_rng_streams():
    dictionary = np.eye(3, dtype=np.float64)
    original = dictionary.copy()
    kwargs = dict(dictionary=dictionary, probabilities=np.full(3, .5), n_samples=128,
                  amplitude_mode="exponential", amplitude_scale=1., noise_std=0., source_offset=0)
    cal = sample_fixed_dictionary(**kwargs, base_seed=2026092101)
    repeat = sample_fixed_dictionary(**kwargs, base_seed=2026092101)
    test = sample_fixed_dictionary(**kwargs, base_seed=2026092102)
    np.testing.assert_array_equal(dictionary, original)
    np.testing.assert_array_equal(cal["x"], cal["z"])
    np.testing.assert_array_equal(cal["x"], repeat["x"])
    assert not np.array_equal(cal["support"], test["support"])
    assert not np.array_equal(cal["z"], test["z"])


def test_l1_threshold_uses_saved_train_fit_or_original_train_only(monkeypatch):
    def forbidden():
        raise AssertionError("A saved threshold must not fit on any fresh activations")
    assert resolve_l1_threshold({"l1_gmm_threshold": ".25"}, forbidden) == (.25, "saved_train_fit_gmm")
    original_train = torch.tensor([[0., 1.], [0., 2.]])
    seen = []
    def train_fit(activations):
        seen.append(activations)
        return .75
    monkeypatch.setattr("scripts.run_vg_development_holdout._l1_threshold", train_fit)
    assert resolve_l1_threshold({}, lambda: original_train) == (.75, "refit_original_train_only")
    assert seen == [original_train]


def test_cap_frontier_selection_ignores_test_and_has_frozen_tie_order():
    def row(run, l0, error, mse, order, test_error):
        return dict(condition="exponential", method="vgsae", run_id=run, checkpoint=run,
                    control_value=order, control_order=order, cal_average_l0=l0,
                    cal_hard_generalization_error=error, cal_hard_reconstruction_mse=mse,
                    cal_hard_support_f1=.5, test_hard_generalization_error=test_error)
    rows = [row("ineligible", 4.1, .01, .01, 0, .01),
            row("first", 3., .5, .1, 1, 999.), row("second", 3., .5, .1, 2, 0.),
            row("higher_mse", 2., .5, .2, 3, 0.)]
    selected = select_calibration_frontier(rows, caps=(1., 4.))
    assert not selected[0]["supported"]
    assert selected[1]["selected_run_id"] == "first"
    assert selected[1]["eligible_control_count"] == 3
    for r in rows:
        r["test_hard_generalization_error"] = -r["test_hard_generalization_error"]
    assert select_calibration_frontier(rows, caps=(1., 4.)) == selected
