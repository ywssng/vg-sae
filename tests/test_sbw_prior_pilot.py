"""Independent objective identities for the trainable normalized prior."""
import torch

from src.sae_model import VariationalGarroteSAE, VGSAEConfig
from scripts.run_sbw_prior_pilot import LearnedPriorVG


def test_learned_prior_matches_base_objective_and_nonprior_gradients_at_same_gamma():
    torch.manual_seed(42)
    cfg=VGSAEConfig(input_dim=4,n_latents=3,lambda_sparsity=2.,dtype='float64')
    base=VariationalGarroteSAE(cfg);learned=LearnedPriorVG(cfg)
    learned.load_state_dict(base.state_dict(),strict=False)
    x=torch.randn(7,4,dtype=torch.float64)
    base_loss=base.free_energy(x)['loss'];new_loss=learned.free_energy(x)['loss']
    torch.testing.assert_close(base_loss,new_loss,atol=1e-12,rtol=0)
    base_loss.backward();new_loss.backward()
    for name,param in base.named_parameters():
        torch.testing.assert_close(param.grad,dict(learned.named_parameters())[name].grad,atol=1e-10,rtol=1e-10)


def test_normalized_prior_gradient_has_bernoulli_self_consistency_not_count_oracle():
    torch.manual_seed(19)
    model=LearnedPriorVG(VGSAEConfig(input_dim=4,n_latents=3,lambda_sparsity=-1.,dtype='float64'))
    x=torch.randn(9,4,dtype=torch.float64);terms=model.free_energy(x)
    derivative=torch.autograd.grad(terms['loss'],model.gamma)[0]
    expected=terms['sparsity'].detach()-3*torch.sigmoid(-model.gamma.detach())
    torch.testing.assert_close(derivative,expected,atol=1e-12,rtol=0)
    # For arbitrary fixed encoder probabilities, optimizing gamma centers the
    # prior on their mean; no ground-truth labels enter this stationary point.
    mean_gate=float(model.encode(x)[0].mean().detach())
    with torch.no_grad():model.gamma.fill_(torch.logit(torch.tensor(1-mean_gate,dtype=torch.float64)))
    derivative=torch.autograd.grad(model.free_energy(x)['loss'],model.gamma)[0]
    assert abs(float(derivative))<1e-12


def test_full_loss_gamma_gradient_matches_centered_finite_differences():
    torch.manual_seed(902)
    model=LearnedPriorVG(VGSAEConfig(input_dim=4,n_latents=3,dtype='float64'))
    x=torch.randn(11,4,dtype=torch.float64)
    for gamma in [-2.,2.,6.]:
        with torch.no_grad():model.gamma.fill_(gamma)
        derivative=torch.autograd.grad(model.free_energy(x)['loss'],model.gamma)[0]
        values=[]
        for shift in [-1e-5,1e-5]:
            with torch.no_grad():
                model.gamma.fill_(gamma+shift)
                values.append(float(model.free_energy(x)['loss']))
        difference=(values[1]-values[0])/2e-5
        torch.testing.assert_close(derivative,torch.tensor(difference,dtype=torch.float64),atol=1e-8,rtol=1e-8)
