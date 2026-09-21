"""Calibrate VG-SAE training recipes, then evaluate a frozen recipe selection."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sae_data import SyntheticSparseCodingConfig, make_synthetic_sparse_coding
from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_train import fit_sae

RECIPES = ('learned', 'learned_moment', 'profiled')


def make_data(seed, amplitude, device):
    cfg = SyntheticSparseCodingConfig(input_dim=16, ground_truth_num_features=64,
        n_samples=8192, support_density=4/64, noise_std=.05, frequency_skew=0,
        amplitude_mode=amplitude, seed=20261021+seed)
    return make_synthetic_sparse_coding(cfg, device=device)


@torch.no_grad()
def initialize_beta_from_train(model, train):
    """Initialize, but do not freeze, learned precision using training energy."""
    if model.log_beta is None:
        raise ValueError('Moment initialization requires learned beta.')
    energy = float(model.free_energy(train)['energy'])
    raw = model.config.input_dim / (2 * max(energy, model.config.loss_eps))
    beta = min(max(raw, 1e-4), 1e4)
    model.log_beta.fill_(math.log(beta))
    return {'train_energy_at_init': energy, 'raw_beta_initial': raw, 'beta_initial': beta}


@torch.no_grad()
def metrics(model, x, z, support, true_dictionary, beta_reference):
    out = model.free_energy(x)
    m, a = out['m'], out['a']
    mask = m > .5
    hard = mask * a
    mean_recon, hard_recon = out['x_hat'], model.decode(hard)
    denominator = (x-x.mean(0)).square().sum(1).mean().clamp_min(1e-12)
    mean_sse = (x-mean_recon).square().sum(1).mean()
    hard_sse = (x-hard_recon).square().sum(1).mean()
    variance = out['variance']
    decoder = torch.nn.functional.normalize(model.decoder.weight, dim=0)
    truth = torch.nn.functional.normalize(true_dictionary, dim=0)
    cosine = decoder.T @ truth
    learned, true = linear_sum_assignment(-cosine.abs().cpu().numpy())
    sign = cosine[learned, true].sign()
    sign[sign == 0] = 1
    aligned = torch.zeros_like(z)
    aligned[:, true] = hard[:, learned]*sign
    tp = (mask[:, learned] & support[:, true].bool()).sum()
    f1 = 2*tp/(mask.sum()+support.sum()).clamp_min(1)
    code_error = ((aligned-z).square().sum()/z.square().sum().clamp_min(1e-12)).sqrt()
    prior_minus_entropy = out['prior']-out['entropy']
    return {
        'hard_l0': float(mask.sum(1).float().mean()),
        'expected_l0': float(m.sum(1).mean()),
        'hard_latent_relative_error': float(code_error),
        'hard_support_f1': float(f1),
        'decoder_recovery_cosine': float(cosine[learned,true].abs().mean()),
        'mean_ev': float(1-mean_sse/denominator),
        'sampled_ev': float(1-(mean_sse+2*variance)/denominator),
        'hard_ev': float(1-hard_sse/denominator),
        'hard_mse': float(hard_sse/x.shape[1]),
        'variance_energy': float(variance),
        'bernoulli_kl': float(prior_minus_entropy),
        'beta_reference_train': beta_reference,
        'conditional_full_free_energy_at_train_beta': float(
            beta_reference*out['energy'] - .5*x.shape[1]*math.log(beta_reference/(2*math.pi))
            + prior_minus_entropy),
    }


def identity(seed, amplitude, gamma, recipe):
    return {'seed':seed, 'amplitude':amplitude, 'gamma':gamma, 'recipe':recipe}


def train(args):
    started = time.perf_counter()
    dest = Path(args.output_dir)/f'seed{args.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    cal_rows, histories = [], []
    for amplitude in args.amplitudes:
        data = make_data(args.seed, amplitude, args.device)
        x_train = data.x[:2048]
        for gamma in args.gammas:
            for recipe in args.recipes:
                torch.manual_seed(args.seed)
                cfg = VGSAEConfig(input_dim=16, n_latents=64, lambda_sparsity=gamma,
                                  beta_mode='profiled' if recipe=='profiled' else 'learned')
                model = VariationalGarroteSAE(cfg).to(args.device)
                with torch.no_grad():
                    model.pre_bias.copy_(x_train.mean(0))
                initial = {'beta_initial':1.0}
                if recipe == 'learned_moment':
                    initial = initialize_beta_from_train(model, x_train)
                info = identity(args.seed, amplitude, gamma, recipe)
                stem = f'{amplitude}_g{gamma:g}_{recipe}'
                path = dest/f'{stem}_u{args.steps}.pt'
                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite completed pilot: {path}')
                before = time.perf_counter()

                def snapshot(row):
                    updates = int(row['step'])+1
                    if updates not in (1001,args.steps):
                        return
                    with torch.no_grad():
                        training = model.free_energy(x_train)
                        beta = float(training['beta_eff'])
                        stationary = cfg.input_dim/(2*float(training['energy']))
                        row_metrics = metrics(model, data.x[2048:4096], data.z[2048:4096],
                            data.support[2048:4096], data.dictionary, beta)
                    checkpoint = dest/f'{stem}_u{updates}.pt'
                    torch.save({'model_config':dataclasses.asdict(cfg),
                        'model_state':model.state_dict(), 'identity':info,
                        'updates':updates, 'beta_reference_train':beta}, checkpoint)
                    cal_rows.append({**info, **initial, 'updates':updates, **row_metrics,
                        'beta_train_stationary':stationary,
                        'beta_to_stationary_ratio':beta/stationary,
                        'checkpoint':str(checkpoint), 'split':'calibration',
                        'elapsed_seconds':time.perf_counter()-before})

                fitted = fit_sae(model, x_train, lr=.003, batch_size=256,
                    max_steps=args.steps, history_every=1000, seed=args.seed,
                    history_callback=snapshot)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize(args.device)
                seconds = time.perf_counter()-before
                histories.append({**info, **initial, 'training_seconds':seconds,
                                  'history':fitted.history})
                print(json.dumps({**info,'seconds':round(seconds,3),
                    'cal_final':cal_rows[-1]['hard_latent_relative_error']}),flush=True)
                # Durable recovery after each completed model, no test metrics.
                (dest/'calibration.json').write_text(json.dumps(cal_rows,indent=2)+'\n')
                (dest/'histories.json').write_text(json.dumps(histories,indent=2)+'\n')
    result = {'seed':args.seed,'data_seed':20261021+args.seed,
        'configuration':{'input_dim':16,'true_width':64,'learned_width':64,'true_mean_l0':4,
            'train':2048,'cal':2048,'test':4096,'noise_std':.05,'frequency_skew':0,
            'amplitudes':args.amplitudes,'recipes':args.recipes,'gammas':args.gammas,
            'lr':.003,'batch_size':256,'steps':args.steps,'seed':args.seed,
            'pre_bias_initialization':'training mean','test_evaluated':False},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,
            'device':args.device,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None},
        'calibration':cal_rows,'histories':histories,'elapsed_seconds':time.perf_counter()-started}
    (dest/'training_results.json').write_text(json.dumps(result,indent=2)+'\n')


def evaluate(args):
    selection_path = Path(args.evaluate_frozen_selection)
    selection = json.loads(selection_path.read_text())
    if selection.get('frozen_before_test') is not True:
        raise ValueError('A frozen calibration-only selection is required.')
    dest = Path(args.output_dir)/f'seed{args.seed}'
    source = json.loads((dest/'training_results.json').read_text())
    rows, data_cache = [], {}
    started = time.perf_counter()
    for row in source['calibration']:
        key=row['amplitude']
        if key not in data_cache:
            data_cache[key]=make_data(args.seed,key,args.device)
        data=data_cache[key]
        checkpoint=torch.load(row['checkpoint'],map_location=args.device,weights_only=True)
        model=VariationalGarroteSAE(VGSAEConfig(**checkpoint['model_config'])).to(args.device)
        model.load_state_dict(checkpoint['model_state']);model.eval()
        score=metrics(model,data.x[4096:],data.z[4096:],data.support[4096:],data.dictionary,
                      checkpoint['beta_reference_train'])
        selected=selection['selected'][f"{row['amplitude']}|{row['gamma']:g}"]['recipe']
        rows.append({**identity(args.seed,row['amplitude'],row['gamma'],row['recipe']),
            'updates':row['updates'],**score,'selected_by_calibration':row['recipe']==selected,
            'checkpoint':row['checkpoint'],'split':'test'})
    result={'selection_file':str(selection_path),'seed':args.seed,'metrics':rows,
            'elapsed_seconds':time.perf_counter()-started}
    (dest/'test_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'seed':args.seed,'test_rows':len(rows),'seconds':result['elapsed_seconds']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--amplitudes',nargs='+',choices=['exponential','constant'],default=['exponential','constant'])
    p.add_argument('--gammas',type=float,nargs='+',default=[2.,6.])
    p.add_argument('--recipes',nargs='+',choices=RECIPES,default=list(RECIPES))
    p.add_argument('--output-dir',default='outputs/vg_sae_development_20260921/training')
    p.add_argument('--evaluate-frozen-selection')
    args=p.parse_args()
    if args.steps<1:p.error('steps must be positive')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.evaluate_frozen_selection:evaluate(args)
    else:train(args)


if __name__=='__main__':
    main()
