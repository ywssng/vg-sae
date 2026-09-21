"""Regression checks for the paper-inspired generator and truth-based metrics."""
import math
import torch

from scripts.sbw_pilot_utils import geometry_metrics, orthogonal_dictionary, sample_toy


def test_copula_changes_firing_correlation_not_dictionary_geometry():
    d = orthogonal_dictionary(91)
    torch.testing.assert_close(d.T @ d, torch.eye(5), atol=2e-7, rtol=0)
    for correlation in [-.4, 0., .4]:
        data = sample_toy(d, 60000, 903, correlation)
        torch.testing.assert_close(data.support.float().mean(0), torch.full((5,), .4), atol=.009, rtol=0)
        corr = torch.corrcoef(data.support.float().T)
        actual_hub = float(corr[0, 1:].mean())
        assert abs(actual_hub) < .01 if correlation == 0 else actual_hub * correlation > .04
        assert abs(actual_hub-correlation) > .08 if correlation else True
        torch.testing.assert_close(data.x, data.z @ d.T)


def test_independent_split_and_amplitude_rng_preserve_matched_support():
    d = orthogonal_dictionary(8)
    first = sample_toy(d, 4096, 900, .4)
    repeat = sample_toy(d, 4096, 900, .4)
    heldout = sample_toy(d, 4096, 901, .4)
    broad = sample_toy(d, 4096, 900, .4, amplitude="exponential")
    assert torch.equal(first.x, repeat.x)
    assert not torch.equal(first.support, heldout.support)
    assert torch.equal(first.support, broad.support)
    assert torch.equal(first.dictionary, heldout.dictionary)


def test_low_decoder_cosine_is_not_a_truth_recovery_certificate():
    d = orthogonal_dictionary(33)
    rotated = d.clone()
    rotated[:, 0] = (d[:, 0]+d[:, 1])/math.sqrt(2)
    rotated[:, 1] = (d[:, 0]-d[:, 1])/math.sqrt(2)
    correct, _ = geometry_metrics(d, d)
    mixed, _ = geometry_metrics(rotated, d)
    assert correct["c_dec"] < 1e-7 and mixed["c_dec"] < 1e-7
    assert correct["mixing_energy"] < 1e-7
    assert mixed["mixing_energy"] > .19
    assert mixed["matched_abs_cosine"] < .89
    assert mixed["recovered_fraction_cosine_095"] == .6


def test_signed_primary_matching_does_not_treat_negative_atoms_as_correct():
    d = orthogonal_dictionary(19)
    perm = torch.tensor([2, 4, 0, 1, 3])
    signed = d[:, perm] * torch.tensor([2., -3., .5, 4., 1.])
    score, (li, ti, signs) = geometry_metrics(signed, d)
    assert score["matched_abs_cosine"] > .999999
    assert score["matched_positive_cosine"] < .7
    assert score["recovered_fraction_cosine_095"] < 1
    torch.testing.assert_close(signs.float(), torch.ones(5))
    mixed = d.clone();mixed[:, 0] = (d[:, 0]-.4*d[:, 1])/math.sqrt(1.16)
    score, _ = geometry_metrics(mixed, d)
    assert score["negative_mixing_energy"] > .02


def test_shared_batch_order_and_completed_update_snapshots_across_trainers():
    from scripts.sbw_pilot_utils import train_shared_batches
    from src.sae_train import build_sae

    data = sample_toy(orthogonal_dictionary(3), 64, 19)
    seen, hashes = [], []
    for method in ["vg", "l1"]:
        kwargs = {} if method == "vg" else {"normalize_activations": "none"}
        model = build_sae(method, 20, 5, **kwargs)
        updates = []
        result = train_shared_batches(model, data.x, steps=4, seed=92, batch_size=16,
            checkpoints=(1,4), snapshot=lambda _, step: updates.append(step))
        seen.append(updates)
        hashes.append(result["first_batch_sha256"])
        assert result["sample_presentations"] == 64
        assert result["completed_updates"] == 4
    assert seen == [[1,4],[1,4]]
    assert hashes[0] == hashes[1]
