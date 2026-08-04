"""Evaluation metrics and baselines for Variational Garrote."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import VariationalGarrote


@dataclass
class EvaluationResult:
    rho_model: float
    generalization_error: float
    selection_error: float | None = None


def model_sparsity(mask: torch.Tensor) -> torch.Tensor:
    return mask.mean()


def generalization_error_torch(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    numerator = (y_pred - y_true).pow(2).sum()
    denominator = y_true.pow(2).sum().clamp_min(eps)
    return torch.sqrt(numerator / denominator)


@torch.no_grad()
def generalization_error(
    model: VariationalGarrote,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    eps: float = 1.0e-12,
) -> float:
    model.eval()
    return float(generalization_error_torch(model(x_test), y_test, eps=eps).detach().cpu())


def generalization_error_numpy(
    w: np.ndarray,
    m: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    eps: float = 1.0e-12,
) -> float:
    y_pred = x_test @ (m * w)
    numerator = np.sum((y_pred - y_test) ** 2)
    denominator = max(float(np.sum(y_test ** 2)), eps)
    return float(np.sqrt(numerator / denominator))


def compute_rho_model_vg(model: VariationalGarrote) -> float:
    return model.get_sparsity()


def selection_error(m: np.ndarray | torch.Tensor, s_true: np.ndarray | torch.Tensor) -> float:
    m_np = np.asarray(m.detach().cpu() if isinstance(m, torch.Tensor) else m, dtype=float)
    s_np = np.asarray(s_true.detach().cpu() if isinstance(s_true, torch.Tensor) else s_true, dtype=float)
    return float(np.mean(s_np * (1.0 - m_np) + (1.0 - s_np) * m_np))


def selection_uncertainty(mask_values_ensemble: np.ndarray | torch.Tensor) -> float:
    values = np.asarray(
        mask_values_ensemble.detach().cpu() if isinstance(mask_values_ensemble, torch.Tensor) else mask_values_ensemble,
        dtype=float,
    )
    mean_mask = values if values.ndim == 1 else values.mean(axis=0)
    return float(np.mean(mean_mask * (1.0 - mean_mask)))


def mean_field_selection_error(rho_model: np.ndarray | float, rho_data: float) -> np.ndarray:
    return np.abs(np.asarray(rho_model, dtype=np.float64) - float(rho_data))


def theoretical_e_sel(rho_model: float, rho_data: float) -> float:
    return float(abs(rho_model - rho_data))


def mean_field_uncertainty_kernel(
    rho_model: np.ndarray | float,
    rho_data: float,
    eps: float = 1.0e-12,
) -> np.ndarray:
    rho_model_arr = np.asarray(rho_model, dtype=np.float64)
    rho_data_safe = max(float(rho_data), eps)
    one_minus_rho_data = max(1.0 - float(rho_data), eps)
    under = (rho_model_arr / rho_data_safe) * (rho_data_safe - rho_model_arr)
    over = (rho_model_arr - rho_data_safe) * ((1.0 - rho_model_arr) / one_minus_rho_data)
    return np.where(rho_model_arr <= rho_data_safe, under, over)


def theoretical_sigma_sel(rho_model: float, rho_data: float, N: int | None = None) -> float:
    del N
    return float(mean_field_uncertainty_kernel(rho_model, rho_data))


def _projected_gradient_nnls(
    templates: np.ndarray,
    observed: np.ndarray,
    steps: int,
    lr: float,
) -> np.ndarray:
    coeffs = np.full(templates.shape[1], 1.0 / max(templates.shape[1], 1), dtype=np.float64)
    for _ in range(steps):
        residual = templates @ coeffs - observed
        grad = 2.0 * templates.T @ residual / max(observed.shape[0], 1)
        coeffs = np.maximum(coeffs - lr * grad, 0.0)
    return coeffs


def infer_data_sparsity(
    rho_model_values: np.ndarray,
    sigma_values: np.ndarray,
    candidate_rhos: np.ndarray,
    steps: int = 2000,
    lr: float = 0.05,
    prefer_scipy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rho_model_values = np.asarray(rho_model_values, dtype=np.float64)
    sigma_values = np.asarray(sigma_values, dtype=np.float64)
    candidate_rhos = np.asarray(candidate_rhos, dtype=np.float64)
    templates = np.stack(
        [mean_field_uncertainty_kernel(rho_model_values, rho_data) for rho_data in candidate_rhos],
        axis=1,
    )

    coeffs: np.ndarray
    if prefer_scipy:
        try:
            from scipy.optimize import nnls

            coeffs, _ = nnls(templates, sigma_values)
        except Exception:
            coeffs = _projected_gradient_nnls(templates, sigma_values, steps=steps, lr=lr)
    else:
        coeffs = _projected_gradient_nnls(templates, sigma_values, steps=steps, lr=lr)

    total = coeffs.sum()
    posterior = np.full_like(coeffs, 1.0 / max(coeffs.shape[0], 1)) if total <= 0.0 else coeffs / total
    return candidate_rhos, posterior


def infer_rho_data(
    sigma_sel_observed: np.ndarray,
    rho_model_values: np.ndarray,
    N: int | None = None,
    rho_data_candidates: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    del N
    if rho_data_candidates is None:
        rho_data_candidates = np.linspace(0.001, 0.5, 200)
    candidates, posterior = infer_data_sparsity(
        rho_model_values=rho_model_values,
        sigma_values=sigma_sel_observed,
        candidate_rhos=rho_data_candidates,
    )
    return posterior, float(candidates[np.argmax(posterior)])


def extract_lasso_mask(coef: np.ndarray, n_components: int = 2) -> np.ndarray:
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError("scikit-learn is required for extract_lasso_mask.") from exc

    abs_coef = np.abs(coef).reshape(-1, 1)
    if np.allclose(abs_coef, abs_coef[0]):
        return (abs_coef[:, 0] > 0.0).astype(float)
    gmm = GaussianMixture(
        n_components=n_components,
        means_init=np.array([[0.0], [float(abs_coef.max()) / 2.0]]),
        covariance_type="full",
        random_state=0,
    )
    labels = gmm.fit_predict(abs_coef)
    spike_component = int(np.argmin(gmm.means_.flatten()))
    return (labels != spike_component).astype(float)


def extract_ridge_mask(coef: np.ndarray, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    w_ols, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    lower_bound = float(np.min(np.abs(w_ols)))
    return (np.abs(coef) >= lower_bound).astype(float)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0):
    try:
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise ImportError("scikit-learn is required for fit_ridge.") from exc

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x_train, y_train)
    return model.coef_, model


def fit_lasso(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.01):
    try:
        from sklearn.linear_model import Lasso
    except ImportError as exc:
        raise ImportError("scikit-learn is required for fit_lasso.") from exc

    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(x_train, y_train)
    return model.coef_, model


@torch.no_grad()
def evaluate_model(
    model: VariationalGarrote,
    x: torch.Tensor,
    y: torch.Tensor,
    true_selection: torch.Tensor | None = None,
) -> EvaluationResult:
    mask = model.mask()
    sel_error = None if true_selection is None else selection_error(mask, true_selection)
    return EvaluationResult(
        rho_model=float(mask.mean().detach().cpu()),
        generalization_error=generalization_error(model, x, y),
        selection_error=sel_error,
    )
