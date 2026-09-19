"""Aggregate pilot evidence and apply explicitly exploratory group ablations."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.idea_discovery_training_pilot import exact_count_mask
from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.sae_model import VGSAEConfig, VariationalGarroteSAE


@torch.no_grad()
def ablate(model, x_train, x_test):
    train = model.free_energy(x_train)
    test = model.free_energy(x_test)
    pi_logit = -model.config.lambda_sparsity
    seed = torch.Generator().manual_seed(20260919)
    result = []
    masks = {}
    for name, terms in [('train', train), ('test', test)]:
        h, m = terms['h'], terms['m']
        near = (terms['gate_logits']-pi_logit).abs() <= 0.25
        masks[name] = {
            'near_prior_logodds025': near,
            'all_subthreshold': m <= 0.5,
            'random_same_count': exact_count_mask(torch.rand(h.shape, generator=seed), near.sum(1)),
            'smallest_ma_same_count': exact_count_mask(-h, near.sum(1)),
        }
    base = test['x_hat']
    residual = x_test-base
    energy = (x_test-x_test.mean(0)).square().sum(1).mean()
    for group in masks['test']:
        mt, me = masks['train'][group], masks['test'][group]
        vt = model.decoder(mt*train['h'])
        ve = model.decoder(me*test['h'])
        # Mean-preserving deletion estimates the replacement ONLY from train.
        centered_removal = ve-vt.mean(0)
        raw_delta = ((residual+ve).square()-residual.square()).sum(1).mean()
        mean_delta = ((residual+centered_removal).square()-residual.square()).sum(1).mean()
        result.append({
            'group': group, 'mean_count': me.sum(1).float().mean().item(),
            'expected_count_removed': (me*test['m']).sum(1).mean().item(),
            'raw_decoded_energy_ratio': (ve.square().sum(1).mean()/energy).item(),
            'decoded_variation_ratio': ((ve-ve.mean(0)).square().sum(1).mean()/energy).item(),
            'raw_deletion_ev_loss': (raw_delta/energy).item(),
            'train_mean_preserving_deletion_ev_loss': (mean_delta/energy).item(),
        })
    return result


def main():
    torch.set_num_threads(2)
    source = Path('outputs/idea_discovery_20260919')
    dest = Path('idea-stage/evidence/pilots')
    dest.mkdir(parents=True, exist_ok=True)
    summaries, readouts, ablations = [], [], []
    data = make_synthetic_sparse_coding(SyntheticSparseCodingConfig(input_dim=16,
        ground_truth_num_features=32, n_samples=12288, support_density=2/32,
        noise_std=0.05, seed=20260919))
    total_seconds = 0
    for seed in (0,1):
        path = source/'training'/f'seed{seed}'/'results.json'
        result = json.loads(path.read_text())
        shutil.copyfile(path, dest/f'training_seed{seed}.json')
        summaries.extend(result['summary'])
        readouts.extend(result['readouts'])
        total_seconds += result['wall_seconds']
        for row in result['summary']:
            path = source/'training'/f'seed{seed}'/f"{row['prior_mode']}_w{row['width']}.pt"
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model = VariationalGarroteSAE(VGSAEConfig(**checkpoint['config']))
            model.load_state_dict(checkpoint['model_state'])
            identity = {k:row[k] for k in ('seed','width','prior_mode','gamma')}
            ablations.extend({**identity, **r} for r in ablate(model, data.x[:8192], data.x[10240:]))
    for name, rows in [('training_summary',summaries), ('readouts',readouts), ('exploratory_group_ablation',ablations)]:
        with (dest/f'{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name in ('results.json','results.csv','verification.json'):
        shutil.copyfile(source/'replication'/name, dest/f'replication_{name}')
    summary = {
        'training_gpu_process_wall_hours': total_seconds/3600,
        'resource_accounting': 'sum of two one-GPU process wall times; excludes import/startup and CPU analysis',
        'group_ablation_status': 'exploratory follow-up fixed after inspecting width/readout pilot, not preregistered',
        'group_ablation_threshold_logodds': 0.25,
        'group_ablation_replacement_mean_split': 'training only',
        'group_ablation_random_seed': 20260919,
        'near_prior_is_not_proof_of_statistical_independence': True,
        'summary': summaries, 'readouts': readouts, 'group_ablation': ablations,
    }
    (dest/'pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'training_gpu_process_wall_hours':total_seconds/3600,'models':len(summaries),'group_rows':len(ablations)}))


if __name__ == '__main__':
    main()
