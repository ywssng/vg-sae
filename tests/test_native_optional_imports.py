"""The native Colab path must not require optional SAELens dependencies."""
import subprocess
import sys


def test_public_native_import_and_backward_without_saelens():
    script = '''
import importlib.abc
import sys
class NoSAELens(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sae_lens" or fullname.startswith("sae_lens."):
            raise ModuleNotFoundError("SAELens intentionally unavailable")
sys.meta_path.insert(0, NoSAELens())
import torch
import src
from src import VGSAEConfig, VariationalGarroteSAE
from src.sae_model import VGSAEConfig as DirectConfig
assert VGSAEConfig is DirectConfig
assert "TopKSAE" in dir(src)
assert not any(name.startswith("sae_lens") for name in sys.modules)
torch.manual_seed(0)
model = VariationalGarroteSAE(VGSAEConfig(input_dim=4, n_latents=6))
loss = model.free_energy(torch.arange(32, dtype=torch.float32).reshape(8, 4)/32)["loss"]
loss.backward()
assert torch.isfinite(loss)
assert model.gate_encoder.weight.grad is not None
try:
    src.TopKSAE
except ModuleNotFoundError as error:
    assert "SAELens intentionally unavailable" in str(error)
else:
    raise AssertionError("Optional baseline unexpectedly available")
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               text=True, timeout=45)
    assert completed.returncode == 0, completed.stderr


def test_lazy_baselines_preserve_public_alias_identity(monkeypatch):
    import types
    import src
    import src.sae_model as model
    fake = types.ModuleType("src.sae_baselines")
    for name in model._BASELINE_EXPORTS:
        setattr(fake, name, type(name, (), {}))
        monkeypatch.delitem(model.__dict__, name, raising=False)
        monkeypatch.delitem(src.__dict__, name, raising=False)
    monkeypatch.setitem(sys.modules, "src.sae_baselines", fake)
    monkeypatch.setattr(src, "sae_baselines", fake, raising=False)
    for name in model._BASELINE_EXPORTS:
        assert getattr(model, name) is getattr(fake, name)
        assert getattr(src, name) is getattr(fake, name)
        delattr(model, name)
        delattr(src, name)
