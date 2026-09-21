"""Independent small numerical checks of exploratory parallel P3b updates."""
import itertools

import torch

from scripts.run_vg_development_parallel_refinement import (
    parallel_refinement, support_projected_gradient,
)


def fixture():
    dtype=torch.float64
    x=torch.tensor([[1.1,-.2],[-.3,1.7]],dtype=dtype)
    d=torch.tensor([[1.5,.4,-.6],[.2,.9,.7]],dtype=dtype)
    b=torch.tensor([.1,-.25],dtype=dtype)
    a=torch.tensor([[.8,0.,1.2],[.3,1.4,.7]],dtype=dtype)
    m=torch.tensor([[.2,.8,.4],[.7,.15,.6]],dtype=dtype)
    return x,m,a,d,b


def test_eta_zero_preserves_probability_and_raw_score_exactly():
    x,m,a,d,b=fixture()
    scores=torch.logit(m)
    result,rank=parallel_refinement(x,m,a,d,b,2.3,.7,0,scores=scores)
    torch.testing.assert_close(result,m,atol=0,rtol=0)
    torch.testing.assert_close(rank,scores,atol=0,rtol=0)


def test_parallel_update_matches_independent_excluded_atom_sums_and_mixture_rank():
    x,m,a,d,b=fixture()
    target=torch.zeros_like(m)
    # Every coordinate uses ORIGINAL m: no sequential updates inside this oracle.
    for i in range(len(x)):
        for j in range(m.shape[1]):
            excluded=x[i]-b
            for k in range(m.shape[1]):
                if k != j:
                    excluded=excluded-m[i,k]*a[i,k]*d[:,k]
            score=2.3*(a[i,j]*torch.dot(d[:,j],excluded)-.5*a[i,j]**2*torch.dot(d[:,j],d[:,j]))-.7
            target[i,j]=score.sigmoid()
    for eta in (.25,.5,1.):
        result,rank=parallel_refinement(x,m,a,d,b,2.3,.7,eta,scores=torch.logit(m))
        expected=(1-eta)*m+eta*target
        torch.testing.assert_close(result,expected,atol=2e-15,rtol=2e-15)
        torch.testing.assert_close(rank.sigmoid(),expected,atol=2e-15,rtol=2e-15)
    # Orthogonal atoms give independently specified target logits; probabilities
    # all round to one, but true mixture survival probability sets the ranking.
    old=torch.tensor([[100.,120.,110.]],dtype=torch.float64)
    wanted=torch.tensor([[130.,80.,100.]],dtype=torch.float64)
    result,rank=parallel_refinement(wanted+.5,old.sigmoid(),torch.ones_like(old),
        torch.eye(3,dtype=torch.float64),torch.zeros(3,dtype=torch.float64),1.,0.,.25,scores=old)
    assert bool((result==1).all())
    assert rank.argsort(descending=True).tolist()==[[2,0,1]]


def test_simultaneous_damped_updates_have_no_monotonicity_guarantee():
    x=torch.tensor([[1.]],dtype=torch.float64)
    d=torch.ones((1,2),dtype=torch.float64)
    a=torch.ones((1,2),dtype=torch.float64)
    b=torch.zeros(1,dtype=torch.float64)
    m=torch.full((1,2),.6,dtype=torch.float64)
    def enumerated_cost(prob):
        cost=prob.new_zeros(())
        for bits in itertools.product((0.,1.),repeat=2):
            s=prob.new_tensor(bits)
            q=torch.where(s.bool(),prob[0],1-prob[0]).prod()
            energy=.5*(1-s.sum()).square()
            # Gaussian/prior constants cancel in this comparison (gamma0).
            cost=cost+20*q*energy+torch.special.xlogy(q,q)
        return cost
    original=enumerated_cost(m)
    for eta in (.5,1.):
        refined,_=parallel_refinement(x,m,a,d,b,20.,0.,eta)
        assert enumerated_cost(refined)>original+.2


def test_support_trace_pg_preserves_support_and_reduces_each_hard_sse():
    dtype=torch.float64
    d=torch.tensor([[1.,1.,0.],[0.,1.,2.]],dtype=dtype)
    bias=torch.tensor([.1,-.2],dtype=dtype)
    centered=torch.tensor([[2.,-1.],[-2.,1.],[0.,2.],[-2.,-1.]],dtype=dtype)
    support=torch.tensor([[1,1,0],[0,0,0],[0,0,1],[1,1,0]],dtype=torch.bool)
    code=torch.tensor([[.5,.2,0.],[0.,0.,0.],[0.,0.,.5],[.1,.1,0.]],dtype=dtype)
    fitted=support_projected_gradient(centered+bias,code,d,bias,support)
    expected=torch.tensor([[14/15,7/30,0.],[0.,0.,0.],[0.,0.,1.],[0.,0.,0.]],dtype=dtype)
    torch.testing.assert_close(fitted,expected,atol=2e-15,rtol=2e-15)
    assert bool((fitted>=0).all())
    assert bool((fitted[~support]==0).all())
    before=(centered-code@d.T).square().sum(1)
    after=(centered-fitted@d.T).square().sum(1)
    assert bool((after<=before+2e-14).all())
