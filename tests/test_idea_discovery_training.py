"""Regression checks for exploratory readout measurements."""
import torch

from scripts.idea_discovery_training_pilot import exact_count_mask, support_f1
from scripts.analyze_idea_discovery_pilots import ablate
from src.sae_model import VGSAEConfig, VariationalGarroteSAE


def test_exact_counts_include_zero_full_and_stable_ties():
    score = torch.ones(3, 4)
    selected = exact_count_mask(score, torch.tensor([0, 2, 4]))
    assert selected.sum(1).tolist() == [0, 2, 4]
    assert selected[1].tolist() == [True, True, False, False]


def test_unmatched_learned_atoms_count_as_false_positives():
    truth_d = torch.eye(2)
    learned_d = torch.tensor([[1., 0., 0.6], [0., 1., 0.8]])
    truth = torch.tensor([[1., 0.]])
    selected = torch.tensor([[True, False, True]])
    assert abs(support_f1(selected, truth, learned_d, truth_d)-2/3) < 1e-12


def test_hardening_gap_needs_the_residual_cross_term():
    target = torch.tensor([[2., 1.]], dtype=torch.float64)
    mean = torch.tensor([[1., 0.]], dtype=torch.float64)
    hard = torch.tensor([[0., 2.]], dtype=torch.float64)
    delta, residual = hard-mean, target-mean
    observed = (target-hard).square().sum()-(target-mean).square().sum()
    decomposed = delta.square().sum()-2*(residual*delta).sum()
    torch.testing.assert_close(observed, decomposed)
    assert observed != delta.square().sum()


def test_mean_preserving_ablation_does_not_call_a_constant_group_information():
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=2, n_latents=3,
        lambda_sparsity=2., dtype=torch.float64))
    with torch.no_grad():
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.fill_(-2.)
        model.amplitude_encoder.weight.zero_()
        model.amplitude_encoder.bias.fill_(0.)
    x = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    rows = {r['group']:r for r in ablate(model, x, x)}
    row = rows['near_prior_logodds025']
    assert row['mean_count'] == 3
    assert row['decoded_variation_ratio'] == 0
    assert abs(row['train_mean_preserving_deletion_ev_loss']) < 1e-12
