"""Focused mathematical/native correspondence checks; Colab CPU only."""
import math

import pytest

pytest.importorskip("flint", reason="Optional Arb certificate dependency")
from scripts import certify_matched_amplitude as cert


def test_frozen_cases_have_strict_outward_bound_certificates():
    cert.support.initialize_arithmetic()
    assert cert.Q_VALUES == ("1/10", "3/10")
    for q in cert.Q_VALUES:
        row = cert.certify_case(q)
        assert row["status"] == "pair_certified"
        assert all(row["checks"].values())
    with pytest.raises(ValueError, match="frozen"):
        cert.certify_case("1/2")


@pytest.mark.parametrize("q", [0.1, 0.3])
def test_native_pair_preserves_support_and_has_full_stochastic_descent(q):
    torch = pytest.importorskip("torch")
    from src.sae_model import VGSAEConfig, VariationalGarroteSAE
    gamma, beta = math.log((1-q)/q), 400
    u0 = math.log(math.expm1(1))
    x = torch.tensor([[-0.2, 0.0], [0.49, 0.5], [0.95, 1.05], [1.0, 1.4]], dtype=torch.float64)
    pairs = []
    for t in (0.0, 0.5):
        with torch.device("meta"):
            model = VariationalGarroteSAE(VGSAEConfig(
                input_dim=2, n_latents=2, beta=beta, beta_mode="learned",
                lambda_sparsity=gamma, dtype=torch.float64))
        model.to_empty(device="cpu")
        with torch.no_grad():
            model.decoder.weight.copy_(torch.eye(2, dtype=torch.float64))
            model.pre_bias.zero_()
            model.gate_encoder.weight.copy_(beta*torch.eye(2, dtype=torch.float64))
            model.gate_encoder.bias.fill_(-beta/2-gamma)
            model.amplitude_encoder.weight.copy_(t*torch.eye(2, dtype=torch.float64))
            model.amplitude_encoder.bias.fill_(u0-t)
            model.log_beta.fill_(math.log(beta))
            m, a, _ = model.encode(x)
            result = model.free_energy(x)
            explicit_risk = ((1-m)*x.square()+m*(x-a).square()).sum(dim=1).mean()
        assert 2*result["energy"].item() == pytest.approx(explicit_risk.item(), abs=1e-14)
        pairs.append((m, a, result, model.decoder.weight.detach()))
    m0, a0, f0, d0 = pairs[0]
    m1, a1, f1, d1 = pairs[1]
    assert torch.equal(d0, d1) and torch.equal(m0, m1)
    assert torch.equal(m0 > .5, m1 > .5)
    assert torch.allclose(a0, torch.ones_like(x), atol=1e-15, rtol=0)
    assert torch.all((x-a1).square() <= (x-a0).square()+1e-15)
    gain = (m0*((x-a0).square()-(x-a1).square())).sum(dim=1).mean()*beta/2
    assert gain.item() > 0
    assert (f0["loss"]-f1["loss"]).item() == pytest.approx(gain.item(), rel=1e-11, abs=1e-12)
