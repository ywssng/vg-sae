"""A constructed objective witness, not a fourth training experiment."""
import math

import torch

from src.sae_model import VGSAEConfig, VariationalGarroteSAE


def make_witness(shift, mode):
    generator = torch.Generator().manual_seed(44)
    d = torch.linalg.qr(torch.randn(7,3,generator=generator,dtype=torch.float64))[0]
    z = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.1,0.],[0.,0.,.9],[1.,1.,0.]],dtype=torch.float64)
    model = VariationalGarroteSAE(VGSAEConfig(input_dim=7,n_latents=3,beta=10,
        beta_mode=mode,lambda_sparsity=4,dtype='float64',loss_eps=1e-8))
    if model.log_beta is not None:model.log_beta.requires_grad_(False)
    with torch.no_grad():
        model.decoder.weight.copy_(d)
        model.pre_bias.copy_(-shift*d.sum(1))
        model.amplitude_encoder.weight.copy_(d.T)
        model.amplitude_encoder.bias.zero_()
        model.gate_encoder.weight.zero_()
        model.gate_encoder.bias.fill_(4*shift+4)
    return model,z@d.T,d


def test_profiled_objective_can_improve_with_unchanged_true_atoms_and_dense_codes():
    losses=[]
    for shift in [0.,2.,4.,6.]:
        model,x,d=make_witness(shift,'profiled')
        terms=model.free_energy(x);m,a,_=model.encode(x)
        assert torch.all(((m>.5)*a)>0)
        torch.testing.assert_close(model.decoder.weight,d)
        # At these points the production epsilon floor is not active.
        assert float(2*terms['energy'].detach()/7)>model.config.loss_eps
        losses.append(float(terms['loss'].detach()))
    assert all(y<x-1 for x,y in zip(losses,losses[1:]))


def test_finite_beta_retains_the_gaussian_constant_lower_bound():
    bound=-.5*7*math.log(10/(2*math.pi))
    values=[]
    for shift in [0.,2.,4.,6.]:
        model,x,_=make_witness(shift,'learned')
        assert not model.log_beta.requires_grad
        value=float(model.free_energy(x)['loss'].detach())
        assert value>=bound-1e-10
        values.append(value)
    # A finite asymptote for this family is not a feature-recovery guarantee.
    limit=bound+3*torch.nn.functional.softplus(torch.tensor(4.,dtype=torch.float64)).item()
    assert abs(values[-1]-limit)<.001
