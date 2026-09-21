"""Bounded empirical-Bayes scalar-prior experiment for the Sparse-but-Wrong run.

An experimental subclass keeps the public VG implementation unchanged. Learning
the normalized prior is tested as a hypothesis, not assumed to identify true L0.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sae_model import VariationalGarroteSAE, VGSAEConfig
from scripts.sbw_pilot_utils import (
    evaluate_model, geometry_metrics, orthogonal_dictionary, sample_toy, tensor_hash, train_shared_batches,
)


class LearnedPriorVG(VariationalGarroteSAE):
    def __init__(self, config: VGSAEConfig):
        super().__init__(config)
        self.gamma = torch.nn.Parameter(torch.tensor(config.lambda_sparsity, dtype=config.torch_dtype))

    def free_energy(self, x, **kwargs):
        if "lambda_sparsity" in kwargs:
            raise ValueError("The experimental learned prior cannot take a fixed-gamma override.")
        out = super().free_energy(x, **kwargs)
        prior = self.gamma*out["sparsity"] + self.config.n_latents*F.softplus(-self.gamma)
        out["loss"] = out["loss"] - out["prior"] + prior
        out["prior"] = prior
        out["gamma_eff"] = self.gamma
        return out


def make_model(world: int, gamma: float, learned: bool, device: str):
    torch.manual_seed(world)
    cfg = VGSAEConfig(input_dim=20, n_latents=5, lambda_sparsity=gamma, beta_mode="profiled")
    cls = LearnedPriorVG if learned else VariationalGarroteSAE
    return cls(cfg).to(device)


def prior_metrics(model, data):
    score = evaluate_model(model, data)
    gamma = float(model.gamma.detach()) if isinstance(model, LearnedPriorVG) else model.config.lambda_sparsity
    probability = float(torch.sigmoid(torch.tensor(-gamma)))
    m, a, h = model.encode(data.x)
    code = a*(m > .5)
    _, (li,ti,_) = geometry_metrics(model.decoder.weight,data.dictionary)
    aligned = torch.zeros_like(data.z);aligned[:,ti] = code[:,li]
    predicted = aligned > 0
    true_positive = (predicted & data.support).sum(0).float()
    predicted_count = predicted.sum(0).float();true_count = data.support.sum(0).float()
    feature_f1 = 2*true_positive/(predicted_count+true_count).clamp_min(1)
    tp_mask = predicted & data.support
    terms = model.free_energy(data.x)
    score.update({"gamma_final": gamma, "prior_probability": probability,
                  "mean_gate_probability": float(m.mean()),
                  "prior_stationarity_residual": float(m.sum(1).mean())-5*probability,
                  "amplitude_rms": float(a.square().mean().sqrt()),
                  "mean_code_rms": float(h.square().mean().sqrt()),
                  "prior_endpoint": probability < .005 or probability > .995,
                  "support_f1_macro": float(feature_f1.mean()),
                  "per_feature_f1": feature_f1.tolist(),
                  "per_feature_recall": (true_positive/true_count.clamp_min(1)).tolist(),
                  "per_feature_precision": (true_positive/predicted_count.clamp_min(1)).tolist(),
                  "true_positive_coefficient_bias": float((aligned-data.z)[tp_mask].mean()) if tp_mask.any() else None,
                  "true_positive_coefficient_mae": float((aligned-data.z)[tp_mask].abs().mean()) if tp_mask.any() else None,
                  "profiled_energy_to_floor_ratio_on_evaluation": float(2*terms['energy']/model.config.input_dim/model.config.loss_eps),
                  "profiled_floor_active_on_evaluation": float(2*terms['energy']/model.config.input_dim) <= model.config.loss_eps})
    return score


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def train(args):
    dest = Path(args.output_dir)/f'world{args.world}'
    dest.mkdir(parents=True, exist_ok=True)
    done = dest/'training_results.json'
    if done.exists():
        raise FileExistsError(f'Refusing to overwrite existing completed training: {done}')
    started = time.perf_counter()
    dictionary = orthogonal_dictionary(args.world)
    rows, train_records = [], []
    for probability in [.2,.4,.6]:
        for correlation in [-.4,.4]:
            train_data = sample_toy(dictionary,32768,10_000+args.world,correlation,
                firing_probability=probability,device=args.device)
            cal = sample_toy(dictionary,8192,20_000+args.world,correlation,
                firing_probability=probability,device=args.device)
            arms = [('learned',g) for g in [-2.,2.,6.]]
            if probability != .4 or args.retrain_fixed_source:
                arms.insert(0,('fixed',2.))
            for kind,gamma in arms:
                model = make_model(args.world,gamma,kind=='learned',args.device)
                info={'world':args.world,'firing_probability':probability,
                      'correlation':correlation,'prior_kind':kind,'gamma_initial':gamma}
                stem=f'p{probability:g}_c{correlation:g}_{kind}_g{gamma:g}'
                bounds={'clipped_updates':0, 'boundary_updates':0}
                def project_prior(current,updates):
                    if isinstance(current,LearnedPriorVG):
                        with torch.no_grad():
                            if abs(float(current.gamma))>8:bounds['clipped_updates']+=1
                            current.gamma.clamp_(-8,8)
                            if abs(float(current.gamma))>=7.999:bounds['boundary_updates']+=1
                def snapshot(current,updates):
                    checkpoint=dest/f'{stem}_u{updates}.pt'
                    if checkpoint.exists():raise FileExistsError(checkpoint)
                    with torch.no_grad():score=prior_metrics(current,cal)
                    torch.save({'model_config':dataclasses.asdict(current.config),
                        'model_state':current.state_dict(),'identity':info,
                        'completed_updates':updates},checkpoint)
                    rows.append({**info,'completed_updates':updates,'checkpoint':str(checkpoint),
                                 'split':'calibration',**score,**bounds})
                    save_json(dest/'calibration.json',rows)
                log=train_shared_batches(model,train_data.x,steps=args.steps,seed=40_000+args.world,
                    lr=.003,batch_size=256,checkpoints=tuple(x for x in [500,1500,args.steps] if x<=args.steps),snapshot=snapshot,after_step=project_prior)
                train_records.append({**info,**log,**bounds,'train_sha256':tensor_hash(train_data.x),
                                      'cal_sha256':tensor_hash(cal.x)})
                print(json.dumps({**info,'seconds':round(log['training_and_snapshot_seconds'],2),
                                  'gamma':rows[-1]['gamma_final']}),flush=True)
                save_json(dest/'partial_training.json',{'calibration':rows,'train_records':train_records})
    save_json(done,{'world':args.world,'configuration':{'steps':args.steps,'batch_size':256,
        'lr':.003,'train':32768,'calibration':8192,'test':16384,
        'fixed_probability_04':'same-device retraining of gamma=2' if args.retrain_fixed_source else 'reuse P1 gamma=2',
        'test_generated':False},'calibration':rows,'train_records':train_records,
        'elapsed_seconds':time.perf_counter()-started,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')})


def evaluate(args):
    selection_path=Path(args.frozen_manifest)
    frozen=json.loads(selection_path.read_text())
    if not frozen.get('frozen_before_test'):raise ValueError('Frozen manifest required.')
    dest=Path(args.output_dir)/f'world{args.world}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'test_results.json').exists():raise FileExistsError(dest/'test_results.json')
    dictionary=orthogonal_dictionary(args.world);datasets={};rows=[];start=time.perf_counter()
    for record in frozen['checkpoints']:
        if record['world'] != args.world:continue
        path=Path(record['checkpoint'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Checkpoint changed.')
        ckpt=torch.load(path,map_location=args.device,weights_only=True)
        learned=record['prior_kind']=='learned'
        cls=LearnedPriorVG if learned else VariationalGarroteSAE
        model=cls(VGSAEConfig(**ckpt['model_config'])).to(args.device)
        model.load_state_dict(ckpt['model_state']);model.eval()
        key=(record['firing_probability'],record['correlation'])
        if key not in datasets:datasets[key]=sample_toy(dictionary,16384,30_000+args.world,
            key[1],firing_probability=key[0],device=args.device)
        with torch.no_grad():score=prior_metrics(model,datasets[key])
        rows.append({**record,**score,'split':'test'})
    save_json(dest/'test_results.json',{'world':args.world,'frozen_manifest':str(selection_path),
        'metrics':rows,'elapsed_seconds':time.perf_counter()-start})
    print(json.dumps({'world':args.world,'test_models':len(rows),'seconds':time.perf_counter()-start}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world',type=int,required=True)
    parser.add_argument('--phase',choices=['train','evaluate'],default='train')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--output-dir',default='outputs/sbw_20260922/prior')
    parser.add_argument('--frozen-manifest')
    parser.add_argument('--retrain-fixed-source',action='store_true',
        help='Repeat the six source-density controls when hardware differs from P1.')
    args=parser.parse_args();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.phase=='train':train(args)
    else:
        if not args.frozen_manifest:parser.error('--frozen-manifest is required for evaluation')
        evaluate(args)


if __name__=='__main__':main()
