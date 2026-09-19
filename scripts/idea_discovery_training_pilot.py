"""Bounded exploratory VG-SAE width/prior and readout pilots.

Run from the project root. This deliberately reuses the current model and
trainer without changing public inference or the named Stage-1 protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_train import fit_sae


def exact_count_mask(score: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Select exactly counts[b] atoms; stable ties favor the lower atom index."""
    order = torch.argsort(score, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(score.shape[1], device=score.device).expand_as(order))
    return ranks < counts[:, None]


def support_f1(mask: torch.Tensor, truth: torch.Tensor, dictionary: torch.Tensor,
               true_dictionary: torch.Tensor) -> float:
    """One-to-one atom alignment, including unmatched learned atoms as FP."""
    cos = torch.nn.functional.normalize(dictionary, dim=0).T @ torch.nn.functional.normalize(true_dictionary, dim=0)
    li, ti = linear_sum_assignment(-cos.abs().cpu().numpy())
    matched = mask[:, li]
    target = truth[:, ti].bool()
    tp = (matched & target).sum().item()
    # Includes unmatched learned/true atoms in denominators.
    return 2 * tp / max(mask.sum().item() + truth.sum().item(), 1)


def evaluate(model: VariationalGarroteSAE, x: torch.Tensor, truth: torch.Tensor,
             true_dictionary: torch.Tensor) -> tuple[dict, list[dict]]:
    with torch.no_grad():
        terms = model.free_energy(x)
        m, a, h = terms['m'], terms['a'], terms['h']
        d = model.decoder.weight
        mask = m > 0.5
        count = mask.sum(1)
        fixed_count = torch.full_like(count, 2)
        mean = model.decode(h)
        hard = model.decode(mask * a)
        residual = x - mean
        delta = hard - mean
        variance_x = (x - x.mean(0)).square().sum(1).mean().item()
        pi = 1 / (1 + math.exp(model.config.lambda_sparsity))
        # Float64 avoids cancellation and saturated float32 logarithms.
        p = m.double().clamp(1e-15, 1 - 1e-15)
        kl = (p * (p.log() - math.log(pi)) + (1-p) * (torch.log1p(-p) - math.log1p(-pi))).sum(1)
        gram = torch.nn.functional.normalize(d, dim=0).T @ torch.nn.functional.normalize(d, dim=0)
        gram.fill_diagonal_(-1)
        stats = {
            'expected_l0': m.sum(1).mean().item(), 'hard_l0': count.float().mean().item(),
            'prior_expected_l0': pi * m.shape[1], 'kl_to_prior': kl.mean().item(),
            'variance_energy': terms['variance'].item(), 'beta': terms['beta_eff'].item(),
            'expected_ev': 1 - residual.square().sum(1).mean().item()/variance_x,
            'hard_ev': 1 - (x-hard).square().sum(1).mean().item()/variance_x,
            'hardening_distortion': delta.square().sum(1).mean().item(),
            'hardening_residual_cross': (-2 * residual * delta).sum(1).mean().item(),
            'decoder_mean_max_positive_cosine': gram.max(1).values.mean().item(),
            'decoder_pair_fraction_gt095': (gram > 0.95).sum().item()/(m.shape[1]*(m.shape[1]-1)),
            'test_centered_energy': variance_x,
            'ma_effective_participation': (h.sum(1).square()/h.square().sum(1).clamp_min(1e-20)).mean().item(),
        }
        m_top = exact_count_mask(m, fixed_count)
        h_top = exact_count_mask(h, fixed_count)
        h_adapt = exact_count_mask(h, count)
        codes = {
            'default_hard': (mask*a, mask),
            'same_support_mean': (mask*h, mask),
            'top_ma_same_count': (h_adapt*h, h_adapt),
            'top_m_k2_amplitude': (m_top*a, m_top),
            'top_m_k2_mean': (m_top*h, m_top),
            'top_ma_k2_mean': (h_top*h, h_top),
        }
        readouts = []
        for name, (code, selected) in codes.items():
            mse_sum = (x-model.decode(code)).square().sum(1).mean().item()
            readouts.append({
                'readout': name, 'ev': 1-mse_sum/variance_x, 'mse_per_dimension': mse_sum/x.shape[1],
                'l0': selected.sum(1).float().mean().item(),
                'support_f1': support_f1(selected, truth, d, true_dictionary),
            })
        return stats, readouts


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda'):
        # The CLI sets CUBLAS_WORKSPACE_CONFIG before Python starts.
        torch.backends.cuda.matmul.allow_tf32 = False
    cfg = SyntheticSparseCodingConfig(input_dim=16, ground_truth_num_features=32,
        n_samples=12288, support_density=2/32, noise_std=0.05, seed=20260919)
    data = make_synthetic_sparse_coding(cfg, device=args.device)
    train, calibration, test = data.x[:8192], data.x[8192:10240], data.x[10240:]
    output_dir = Path(args.output_dir) / f'seed{args.seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, readout_rows, histories = [], [], []
    base_gamma = math.log(15)
    for width in args.widths:
        for prior_mode in ('fixed_pi', 'fixed_count'):
            if width == 32 and prior_mode == 'fixed_count':
                continue  # Same configuration, not an independent replication.
            gamma = base_gamma if prior_mode == 'fixed_pi' else math.log((width-2)/2)
            torch.manual_seed(args.seed)
            model = VariationalGarroteSAE(VGSAEConfig(input_dim=16, n_latents=width,
                lambda_sparsity=gamma, beta_mode='learned')).to(args.device)
            with torch.no_grad():
                model.pre_bias.copy_(train.mean(0))
            before = time.perf_counter()
            fitted = fit_sae(model, train, max_steps=args.steps, lr=0.003, batch_size=256,
                history_every=200, seed=args.seed)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter()-before
            stats, readouts = evaluate(model, test, data.support[10240:], data.dictionary)
            identity = {'seed': args.seed, 'width': width, 'prior_mode': prior_mode, 'gamma': gamma}
            row = {**identity, **stats, 'train_seconds': elapsed}
            summary.append(row)
            readout_rows.extend({**identity, **r} for r in readouts)
            histories.append({**identity, 'history': fitted.history})
            torch.save({'config': vars(model.config), 'model_state': model.state_dict()},
                       output_dir / f'{prior_mode}_w{width}.pt')
            print(json.dumps(row), flush=True)
    config = {
        'seed': args.seed, 'widths': args.widths, 'steps': args.steps,
        'data_seed': 20260919, 'input_dim': 16, 'true_features': 32, 'true_density': 2/32,
        'n_train': len(train), 'n_calibration': len(calibration), 'n_test': len(test),
        'calibration_used': False, 'noise_std': 0.05, 'amplitude_mode': 'exponential',
        'batch_size': 256, 'learning_rate': 0.003, 'checkpoint': 'last',
        'beta_mode': 'learned', 'weight_decay': 0, 'decoder_bias': True,
        'initial_gate_bias': -2, 'initial_beta': 1, 'entropy_weight': 1,
        'torch': torch.__version__, 'python': platform.python_version(), 'device': args.device,
        'gpu_name': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
    }
    result = {'config': config, 'summary': summary, 'readouts': readout_rows,
              'histories': histories, 'wall_seconds': time.perf_counter()-start}
    (output_dir/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, rows in [('summary', summary), ('readouts', readout_rows)]:
        with (output_dir/f'{name}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--widths', type=int, nargs='+', default=[32,64,128])
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--output-dir', default='outputs/idea_discovery_20260919/training')
    args = parser.parse_args()
    if args.steps < 1 or any(w < 3 for w in args.widths):
        parser.error('positive steps and widths >= 3 required')
    run(args)


if __name__ == '__main__':
    main()
