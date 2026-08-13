from __future__ import annotations

import pytest
import torch

from src.sae_evaluate import (
    decoder_atoms_from_model,
    decoder_pairwise_cosine_similarity,
)
from src.sae_model import VGSAEConfig, VariationalGarroteSAE


def test_decoder_pairwise_cosine_matches_sparse_but_wrong_eq4() -> None:
    atoms = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, -3.0],
            [2.0, 2.0],
        ]
    )

    value = decoder_pairwise_cosine_similarity(atoms, block_size=2)

    assert value == pytest.approx(2**0.5 / 3.0)


def test_decoder_pairwise_cosine_is_sign_and_scale_invariant() -> None:
    atoms = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [3.0, 4.0]])
    transformed = atoms * torch.tensor([[2.0], [-4.0], [0.5]])

    assert decoder_pairwise_cosine_similarity(
        transformed, block_size=1
    ) == pytest.approx(decoder_pairwise_cosine_similarity(atoms, block_size=3))


def test_decoder_pairwise_cosine_matches_paper_zero_row_convention() -> None:
    atoms = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    assert decoder_pairwise_cosine_similarity(atoms) == pytest.approx(
        2**0.5 / 6.0
    )


def test_decoder_atoms_from_model_uses_each_model_orientation() -> None:
    vg = VariationalGarroteSAE(VGSAEConfig(input_dim=2, n_latents=3))
    local_weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with torch.no_grad():
        vg.decoder.weight.copy_(local_weight)
    saelens_like = torch.nn.Module()
    saelens_like.W_dec = local_weight.T.clone()

    torch.testing.assert_close(decoder_atoms_from_model(vg), local_weight.T)
    torch.testing.assert_close(
        decoder_atoms_from_model(saelens_like), local_weight.T
    )


@pytest.mark.parametrize(
    "atoms",
    [
        torch.ones(2),
        torch.ones(1, 2),
        torch.tensor([[float("nan"), 0.0], [1.0, 0.0]]),
    ],
)
def test_decoder_pairwise_cosine_rejects_invalid_atoms(atoms: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        decoder_pairwise_cosine_similarity(atoms)
