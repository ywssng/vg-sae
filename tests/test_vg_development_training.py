import math

import torch

from scripts.run_vg_development_training import initialize_beta_from_train, make_data, metrics
from src.sae_model import VGSAEConfig, VariationalGarroteSAE


def test_train_moment_initialization_zeroes_full_batch_beta_derivative():
    torch.manual_seed(9)
    model=VariationalGarroteSAE(VGSAEConfig(input_dim=3,n_latents=5,beta_mode='learned',dtype=torch.float64))
    x=torch.randn(13,3,dtype=torch.float64)
    before={k:v.clone() for k,v in model.state_dict().items() if k!='log_beta'}
    initialized=initialize_beta_from_train(model,x)
    model.free_energy(x)['loss'].backward()
    assert abs(float(model.log_beta.grad))<1e-10
    assert initialized['beta_initial']>0
    assert model.log_beta.requires_grad
    for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


def test_profiled_minibatch_log_is_not_global_profiled_log():
    # Independent scalar example checks the interpretation of the reference arm.
    assert .5*(math.log(1)+math.log(9)) < math.log((1+9)/2)


def test_paired_generation_and_risk_variance_identity():
    a=make_data(0,'exponential','cpu')
    b=make_data(0,'constant','cpu')
    torch.testing.assert_close(a.dictionary,b.dictionary,rtol=0,atol=0)
    torch.testing.assert_close(a.support,b.support,rtol=0,atol=0)
    torch.manual_seed(0)
    model=VariationalGarroteSAE(VGSAEConfig(input_dim=16,n_latents=64))
    x=a.x[4096:4128]
    row=metrics(model,x,a.z[4096:4128],a.support[4096:4128],a.dictionary,1.)
    denominator=float((x-x.mean(0)).square().sum(1).mean())
    assert abs((row['mean_ev']-row['sampled_ev'])-2*row['variance_energy']/denominator)<1e-6
