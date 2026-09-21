"""Independent numerical gates for the bounded P3 inference pilot."""
import itertools
import math

import numpy as np
import torch

from scripts.run_vg_development_refinement import (
    exact_count_mask, fixed_objective, sequential_refinement, support_nnls,
)


def problem():
    dtype = torch.float64
    x = torch.tensor([[1.1, -.2], [-.3, 1.7]], dtype=dtype)
    d = torch.tensor([[1.5, .4, -.6], [.2, .9, .7]], dtype=dtype)
    bias = torch.tensor([.1, -.25], dtype=dtype)
    a = torch.tensor([[.8, .0, 1.2], [.3, 1.4, .7]], dtype=dtype)
    m = torch.tensor([[.2, .8, .4], [.7, .15, .6]], dtype=dtype)
    return x, m, a, d, bias, 2.3, .7


def enumeration_free_energy(x, m, a, d, bias, beta, gamma):
    """Use exact joint support probabilities, no analytic variance formula."""
    total = torch.zeros(len(x), dtype=m.dtype)
    for bits in itertools.product((0., 1.), repeat=m.shape[1]):
        s = m.new_tensor(bits)
        q = torch.where(s.bool(), m, 1-m).prod(1)
        err = x-bias-(s*a)@d.T
        energy = .5*err.square().sum(1)
        # E log q - E log p: prior P(s_j=1)=sigmoid(-gamma).
        log_prior = (-gamma*s-torch.nn.functional.softplus(m.new_tensor(-gamma))).sum()
        total = total + beta*q*energy + torch.special.xlogy(q,q) - q*log_prior
    return total - .5*x.shape[1]*math.log(beta/(2*math.pi))


def test_each_coordinate_decreases_exact_enumerated_free_energy():
    x,m,a,d,b,beta,gamma = problem()
    previous = enumeration_free_energy(x,m,a,d,b,beta,gamma)
    analytic = fixed_objective(x,m,a,d,b,beta,gamma)["free_energy"]
    torch.testing.assert_close(analytic, previous, atol=2e-14, rtol=2e-14)
    visited = []
    def check(sweep, j, current, residual):
        nonlocal previous
        exact = enumeration_free_energy(x,current,a,d,b,beta,gamma)
        assert torch.all(exact <= previous+2e-13)
        torch.testing.assert_close(residual, x-b-(current*a)@d.T, atol=2e-14, rtol=2e-14)
        previous = exact
        visited.append((sweep,j))
    sequential_refinement(x,m,a,d,b,beta,gamma,3,after_coordinate=check)
    assert len(visited) == 9
    # Boundary probabilities must stay finite in stable xlogy objective.
    boundary = torch.tensor([[0.,1.,0.],[1.,0.,1.]],dtype=m.dtype)
    torch.testing.assert_close(fixed_objective(x,boundary,a,d,b,beta,gamma)["free_energy"],
                               enumeration_free_energy(x,boundary,a,d,b,beta,gamma))


def test_last_coordinate_is_stationary_with_consistent_residual():
    x,m,a,d,b,beta,gamma = problem()
    refined,_,residual = sequential_refinement(x,m,a,d,b,beta,gamma,1)
    torch.testing.assert_close(residual,x-b-(refined*a)@d.T,atol=2e-14,rtol=2e-14)
    variable = refined.detach().requires_grad_()
    # Exact enumeration supplies a gradient oracle independent of the update.
    gradient = torch.autograd.grad(enumeration_free_energy(x,variable,a,d,b,beta,gamma).sum(),variable)[0]
    torch.testing.assert_close(gradient[:,-1],torch.zeros(len(x),dtype=m.dtype),atol=2e-13,rtol=0)


def test_exact_count_mask_handles_zero_full_ties_and_saturation():
    scores=torch.tensor([[2.,2.,1.,0.],[100.,120.,110.,-5.],[3.,3.,3.,3.],[4.,4.,2.,2.]])
    counts=torch.tensor([0,1,4,2])
    mask=exact_count_mask(scores,counts)
    torch.testing.assert_close(mask.sum(1),counts)
    assert mask.tolist()==[[False]*4,[False,True,False,False],[True]*4,[True,True,False,False]]
    torch.testing.assert_close(mask,exact_count_mask(scores,counts))
    # Sigmoid saturation would tie columns 0,1,2, but raw logit ranks preserve 1.
    assert bool((scores[1,:3].sigmoid()==1).all())


def test_nnls_preserves_support_and_improves_feasible_source_sse():
    d=np.eye(3)
    b=np.array([.1,-.2,.3])
    x=np.array([[2.,-1.,3.],[-2.,1.,4.],[0.,2.,1.]])+b
    support=np.array([[True,True,False],[False,False,False],[True,True,True]])
    original=np.array([[.5,.2,0.],[0.,0.,0.],[.5,.5,.5]])
    fitted=support_nnls(x,d,b,support)
    np.testing.assert_allclose(fitted,np.array([[2.,0.,0.],[0.,0.,0.],[0.,2.,1.]]),atol=1e-14)
    assert np.all(fitted>=0)
    assert np.all(fitted[~support]==0)
    assert np.all(np.sum((x-b-fitted@d.T)**2,axis=1)<=np.sum((x-b-original@d.T)**2,axis=1)+1e-13)
    deficient=np.array([[1.,1.],[0.,0.]])
    fitted=support_nnls(np.array([[2.,.5]]),deficient,np.zeros(2),np.ones((1,2),bool))
    assert np.isfinite(fitted).all()
    np.testing.assert_allclose(fitted@deficient.T,[[2.,0.]],atol=1e-13)
