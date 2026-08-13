from __future__ import annotations

import pytest

from runs.run_CustomData_ablation_matrix import (
    CONDITIONS,
    selected_beta_modes,
    selected_conditions,
    sweep_root,
)


def test_ablation_matrix_contains_the_requested_five_conditions() -> None:
    assert [
        (condition.name, condition.amplitude_mode, condition.frequency_skew)
        for condition in CONDITIONS
    ] == [
        ("ablation2_constant", "constant", 0.5),
        ("ablation2_uniform", "uniform", 0.5),
        ("ablation3_uniformfreq", "exponential", 0.0),
        ("ablation23_constant_uniformfreq", "constant", 0.0),
        ("ablation23_uniform_uniformfreq", "uniform", 0.0),
    ]


def test_ablation_matrix_selectors_and_roots_are_unambiguous() -> None:
    selected = selected_conditions("ablation2_uniform,ablation3_uniformfreq")
    assert [condition.name for condition in selected] == [
        "ablation2_uniform",
        "ablation3_uniformfreq",
    ]
    assert selected_beta_modes("learned,profiled") == ["learned", "profiled"]
    assert sweep_root(selected[0], "learned").name == (
        "stage1_ablation2_uniform_beta_learned"
        "_din128_gt1024_sae1024_sd001_seed0"
    )

    with pytest.raises(ValueError, match="Unknown conditions"):
        selected_conditions("missing")
    with pytest.raises(ValueError, match="profiled and/or learned"):
        selected_beta_modes("fixed")
    with pytest.raises(ValueError, match="duplicate"):
        selected_beta_modes("learned,learned")
