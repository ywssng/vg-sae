"""Bounded, preregistered VG-SAE first-principles campaign.

Run from the repository root with .venv/bin/python. Exact inference is a tiny
known-model reference, never a scalable SAE or a source of training labels.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import sys
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runs.gpu_scheduler import activate_worker_device
from scripts.sbw_pilot_utils import train_shared_batches
from src.sae_baselines import to_inference_sae
from src.sae_data import make_unit_dictionary
from src.sae_inference import (
    best_found_mean_field,
    enumerate_support_posterior,
    factorized_free_energy,
    mean_field_fixed_point,
)
from src.sae_model import VGSAEConfig, VariationalGarroteSAE
from src.sae_train import build_sae


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("Predeclared worker budget exhausted; retain partial outputs.")


def known_dictionary(width: int, coherence: float) -> torch.Tensor:
    d = torch.eye(width, dtype=torch.float64)
    d[:, 1] = 0
    d[0, 1] = coherence
    d[1, 1] = math.sqrt(1 - coherence**2)
    return d


def sample_data(d: torch.Tensor, count: int, seed: int, probability: float,
                noise_std: float, amplitude_scale: float | None = None) -> dict:
    """Independent role-seeded data with a shared, explicitly supplied dictionary."""
    generator = torch.Generator().manual_seed(seed)
    support = torch.rand(count, d.shape[1], generator=generator,
                         dtype=torch.float64) < probability
    a = torch.ones_like(support, dtype=torch.float64)
    if amplitude_scale is not None:
        a = -torch.log1p(-torch.rand(a.shape, generator=generator,
                                    dtype=torch.float64)) * amplitude_scale
    z = support * a
    clean = z @ d.double().T
    x = clean + noise_std * torch.randn(clean.shape, generator=generator,
                                        dtype=torch.float64)
    return dict(x=x, z=z, support=support, clean=clean, dictionary=d.double())


def starts_for(x: torch.Tensor, width: int, gamma: float, seed: int,
               encoder: torch.Tensor | None = None) -> list[torch.Tensor]:
    shape = (len(x), width)
    starts = [x.new_full(shape, torch.sigmoid(x.new_tensor(-gamma)).item()),
              x.new_full(shape, .5), x.new_full(shape, .01), x.new_full(shape, .99)]
    starts.append(torch.rand(shape, dtype=x.dtype,
                             generator=torch.Generator().manual_seed(seed)).to(x.device))
    if encoder is not None:
        starts.append(encoder.clone())
    return starts


def support_scores(m: torch.Tensor, truth: torch.Tensor) -> dict:
    target = truth.to(m)
    safe = m.clamp(1e-12, 1 - 1e-12)
    hard = m > .5
    tp = (hard & truth.bool()).sum()
    return dict(brier=float((m - target).square().mean()),
                marginal_nll=float(-(target * safe.log() + (1-target)*torch.log1p(-safe)).mean()),
                support_f1=float(2*tp/(hard.sum()+truth.sum()).clamp_min(1)))


def exact_block(cfg: dict, seed: int, out: Path) -> None:
    rows, arrays = [], {}
    c = cfg["exact"]
    gamma_data = math.log((1-c["data_probability"])/c["data_probability"])
    for coherence in c["coherences"]:
        d = known_dictionary(c["width"], coherence)
        data = sample_data(d, c["samples"], seed*100+1, c["data_probability"],
                           c["data_beta"]**-.5)
        x, a = data["x"], torch.ones(c["samples"], c["width"], dtype=torch.float64)
        for beta in c["betas"]:
            for gamma in c["gammas"]:
                p = enumerate_support_posterior(x, d, a, beta=beta, gamma=gamma)
                mf = best_found_mean_field(x, d, a, beta=beta, gamma=gamma,
                    starts=starts_for(x, c["width"], gamma, seed+9),
                    max_sweeps=c["max_sweeps"], tolerance=c["tolerance"])
                counts = p["states"].sum(1)
                mean_n = p["p"] @ counts
                variance_n = p["p"] @ counts.square() - mean_n.square()
                h = c["field_difference_step"]
                plus = enumerate_support_posterior(x,d,a,beta=beta,gamma=gamma+h)
                minus = enumerate_support_posterior(x,d,a,beta=beta,gamma=gamma-h)
                response = -(plus["m"].sum(1)-minus["m"].sum(1))/(2*h)
                logq = (torch.special.xlogy(p["states"][None], mf["m"][:,None]) +
                        torch.special.xlogy(1-p["states"][None], 1-mf["m"][:,None])).sum(-1)
                q = logq.exp()
                kl_direct = (torch.special.xlogy(q, q)-q*p["log_p"]).sum(1)
                kl = mf["free_energy"] + p["log_evidence"]
                analytic = torch.sigmoid(beta*(x@d-.5*d.square().sum(0))-gamma)
                pair_cov = p["p"] @ (p["states"][:,0]*p["states"][:,1])-p["m"][:,0]*p["m"][:,1]
                row = dict(seed=seed, coherence=coherence, beta=beta, gamma=gamma,
                    model_matched=abs(beta-c["data_beta"])<1e-12 and abs(gamma-gamma_data)<1e-12,
                    exact_expected_l0=float(mean_n.mean()), mf_expected_l0=float(mf["m"].sum(1).mean()),
                    exact_negative_log_evidence=float(-p["log_evidence"].mean()),
                    exact_count_variance=float(variance_n.mean()),
                    exact_entropy=float(-(p["p"]*p["log_p"]).sum(1).mean()),
                    exact_pair_covariance=float(pair_cov.mean()), mf_reverse_kl=float(kl.mean()),
                    mf_marginal_mse=float((mf["m"]-p["m"]).square().mean()),
                    converged_fraction=float((mf["residual"]<=1e-6).double().mean()),
                    max_residual=float(mf["residual"].max()),
                    response_identity_max_error=float((response-variance_n).abs().max()),
                    free_energy_identity_max_error=float((kl-kl_direct).abs().max()),
                    orthogonal_marginal_max_error=float((p["m"]-analytic).abs().max()) if coherence==0 else None)
                if row["model_matched"]:
                    row.update({"exact_"+k:v for k,v in support_scores(p["m"],data["support"]).items()})
                    row.update({"mf_"+k:v for k,v in support_scores(mf["m"],data["support"]).items()})
                rows.append(row)
                key=f'c{coherence}_b{beta}_g{gamma}'
                arrays[key+"_exact_m"]=p["m"].numpy()
                arrays[key+"_mf_m"]=mf["m"].numpy()
                arrays[key+"_kl"]=kl.numpy()
                arrays[key+"_residual"]=mf["residual"].numpy()
                arrays[key+"_negative_log_evidence"]=-p["log_evidence"].numpy()
                arrays[key+"_count_variance"]=variance_n.numpy()
                arrays[key+"_pair_covariance"]=pair_cov.numpy()
        arrays[f'c{coherence}_x']=x.numpy()
        arrays[f'c{coherence}_support']=data["support"].numpy()
        arrays[f'c{coherence}_dictionary']=d.numpy()
    valid = all(r["response_identity_max_error"]<1e-5 and r["free_energy_identity_max_error"]<1e-8
        and (r["orthogonal_marginal_max_error"] is None or r["orthogonal_marginal_max_error"]<1e-8) for r in rows)
    write_csv(out/"summary.csv",rows)
    np.savez_compressed(out/"samples.npz",**arrays)
    write_json(out/"result.json",dict(block="exact",seed=seed,control_valid=valid,rows=rows))
    if not valid:
        raise RuntimeError("Exact control invalid; inspect summary before launching training.")


def frozen_block(cfg: dict, seed: int, out: Path, device: str, deadline: float=math.inf) -> None:
    c=cfg["frozen"]
    gamma=math.log((1-c["probability"])/c["probability"])
    for coherence in c["coherences"]:
        check_deadline(deadline)
        dest=out/f"coherence_{coherence}"
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/"result.json").exists():
            continue
        d=known_dictionary(c["width"],coherence)
        data={role:sample_data(d,c[role+"_samples"],seed*100+offset,c["probability"],c["beta"]**-.5)
              for role,offset in [("train",11),("calibration",12),("test",13)]}
        torch.manual_seed(seed+10000)
        model=VariationalGarroteSAE(VGSAEConfig(input_dim=c["width"],n_latents=c["width"],
            beta_mode="learned",beta=c["beta"],lambda_sparsity=gamma,decoder_bias=False,
            normalize_decoder=False)).to(device)
        with torch.no_grad():
            model.decoder.weight.copy_(d)
            model.amplitude_encoder.weight.zero_()
            model.amplitude_encoder.bias.fill_(math.log(math.expm1(1)))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.gate_encoder.parameters():
            parameter.requires_grad_(True)
        history=[]
        def snapshot(current, step):
            with torch.no_grad():
                value=current.free_energy(data["calibration"]["x"].to(device=device,dtype=torch.float32))
            history.append(dict(step=step,calibration_free_energy=float(value["loss"]),
                                expected_l0=float(value["sparsity"])))
        write_json(dest/"status.json",dict(status="running",seed=seed,coherence=coherence))
        timing=train_shared_batches(model,data["train"]["x"].to(device=device,dtype=torch.float32),
            steps=c["steps"],seed=seed+20000,lr=c["lr"],batch_size=c["batch_size"],
            checkpoints=tuple(sorted(set([max(1,c["steps"]//2),c["steps"]]))),snapshot=snapshot,
            before_step=lambda _model,_step: check_deadline(deadline))
        torch.save(model.state_dict(),dest/"final.pt")
        with torch.no_grad():
            encoder=model.encode(data["test"]["x"].to(device=device,dtype=torch.float32))[0].cpu().double()
        x=data["test"]["x"]
        a=torch.ones(len(x),c["width"],dtype=torch.float64)
        p=enumerate_support_posterior(x,d,a,beta=c["beta"],gamma=gamma)
        mf=best_found_mean_field(x,d,a,beta=c["beta"],gamma=gamma,
            starts=starts_for(x,c["width"],gamma,seed+19,encoder),
            max_sweeps=c["max_sweeps"],tolerance=1e-9)
        f_encoder=factorized_free_energy(x,d,a,encoder,beta=c["beta"],gamma=gamma)
        drop=f_encoder-mf["free_energy"]
        encoder_residual=(encoder-mean_field_fixed_point(x,d,a,encoder,beta=c["beta"],gamma=gamma)).abs().amax(1)
        rows=[]
        for name,m in [("exact",p["m"]),("mean_field",mf["m"]),("encoder",encoder)]:
            f=-p["log_evidence"] if name=="exact" else factorized_free_energy(x,d,a,m,beta=c["beta"],gamma=gamma)
            rows.append(dict(seed=seed,coherence=coherence,method=name,
                **support_scores(m,data["test"]["support"]),
                reverse_kl=float((f+p["log_evidence"]).mean()),free_energy=float(f.mean()),
                marginal_mse_to_exact=float((m-p["m"]).square().mean()),expected_l0=float(m.sum(1).mean())))
        checks=dict(decoder_max_error=float((model.decoder.weight.detach().cpu().double()-d).abs().max()),
            amplitude_weight_max=float(model.amplitude_encoder.weight.detach().abs().max()),
            amplitude_max_error=float((F.softplus(model.amplitude_encoder.bias.detach())-1).abs().max()),
            beta_error=abs(float(model.log_beta.detach().exp())-c["beta"]))
        if checks["decoder_max_error"]>1e-6 or max(checks[k] for k in checks if k!="decoder_max_error")>1e-6:
            raise RuntimeError("Frozen target changed.")
        result=dict(block="frozen",seed=seed,coherence=coherence,rows=rows,timing=timing,
            frozen_checks=checks,mean_refinement_drop=float(drop.mean()),min_refinement_drop=float(drop.min()),
            mf_converged_fraction=float((mf["residual"]<1e-6).double().mean()),
            encoder_mean_residual=float(encoder_residual.mean()),encoder_max_residual=float(encoder_residual.max()),
            mf_max_residual=float(mf["residual"].max()),history=history)
        np.savez_compressed(dest/"samples.npz",x=x.numpy(),truth=data["test"]["support"].numpy(),
            exact=p["m"].numpy(),mean_field=mf["m"].numpy(),encoder=encoder.numpy(),drop=drop.numpy(),
            residual=mf["residual"].numpy(),encoder_residual=encoder_residual.numpy())
        write_json(dest/"result.json",result)
        write_json(dest/"status.json",dict(status="completed"))
        print(json.dumps(dict(block="frozen",seed=seed,coherence=coherence,drop=result["mean_refinement_drop"])),flush=True)


@torch.no_grad()
def joint_metrics(model: torch.nn.Module, data: dict, device: str,
                  effective_threshold: float=1e-6) -> dict:
    x=data["x"].to(device=device,dtype=torch.float32)
    if isinstance(model,VariationalGarroteSAE):
        m,a,h=model.encode(x)
        code=(m>.5)*a
        recon=model.decode(code)
        mean_mse=(model.decode(h)-x).square().mean()
        variance=.5*(m*(1-m)*a.square()*model.decoder_column_sqnorms()).sum(1).mean()
        terms=model.free_energy(x)
        decoder=model.decoder.weight
        extra=dict(expected_mask_count=float(m.sum(1).mean()),hard_mask_count=float((m>.5).float().sum(1).mean()),
            effective_mean_code_count=float((h.abs()>effective_threshold).float().sum(1).mean()),
            mean_mse=float(mean_mse),full_stochastic_mse=float(mean_mse+2*variance/x.shape[1]),
            full_variance_energy=float(variance),optimized_variance_energy=float(terms["variance"]),
            entropy_per_sample=float(terms["entropy"]),beta_eff=float(terms["beta_eff"]),
            optimized_free_energy=float(terms["loss"]),prior_cost=float(terms["prior"]),
            mask_uncertainty=float((m*(1-m)).sum(1).mean()))
    else:
        inference=to_inference_sae(model,fold_decoder_norm=True).to(device)
        code=inference.encode(x)
        recon=inference.decode(code)
        decoder=inference.W_dec.T
        extra={}
    truth=data["dictionary"].to(decoder)
    cosines=F.normalize(decoder,dim=0).T @ F.normalize(truth,dim=0)
    li,ti=linear_sum_assignment(-cosines.cpu().numpy())
    aligned=torch.zeros_like(data["z"],device=device,dtype=code.dtype)
    aligned[:,ti]=code[:,li]
    target=data["z"].to(aligned)
    target_support=data["support"].to(device)
    support=code>0
    tp=(support[:,li]&target_support[:,ti]).sum()
    clean=data["clean"].to(recon)
    return dict(**extra,hard_mse=float((recon-x).square().mean()),
        clean_signal_mse=float((recon-clean).square().mean()),hard_l0=float(support.float().sum(1).mean()),
        true_l0=float(target_support.float().sum(1).mean()),
        signed_dictionary_cosine=float(cosines[li,ti].mean()),
        recovered_fraction_095=float((cosines[li,ti]>=.95).float().mean()),
        support_f1=float(2*tp/(support.sum()+target_support.sum()).clamp_min(1)),
        coefficient_nmse=float((aligned-target).square().sum()/target.square().sum().clamp_min(1e-12)))


@torch.no_grad()
def profile_batch_diagnostics(model: VariationalGarroteSAE, x: torch.Tensor,
                              batch_sizes: list[int]) -> list[dict]:
    """Evaluate the configured energy without changing parameters or precision."""
    output=model.free_energy(x)
    m,a,h=output["m"],output["a"],output["h"]
    energy=.5*(x-model.decode(h)).square().sum(1)
    if model.config.use_variance_term:
        energy+=.5*(m*(1-m)*a.square()*model.decoder_column_sqnorms()).sum(1)
    d=x.shape[1]
    eps=model.config.loss_eps
    full_risk=(2*energy.mean()/d).clamp_min(eps)
    full=.5*d*full_risk.log()
    rows=[]
    for size in batch_sizes:
        if len(x)%size:
            raise ValueError("Calibration sample count must be divisible by profile batch size.")
        risk=(2*energy.reshape(-1,size).mean(1)/d).clamp_min(eps)
        beta=risk.reciprocal()
        mean_log=.5*d*risk.log().mean()
        rows.append(dict(batch_size=size,full_profile_gaussian=float(full),
            mean_batch_profile_gaussian=float(mean_log),jensen_gap=float(full-mean_log),
            beta_mean=float(beta.mean()),beta_std=float(beta.std(unbiased=False)),
            beta_full=float(full_risk.reciprocal()),floor_active=bool((risk<=eps).any())))
    return rows


def build_joint_model(method: str, value: float, c: dict, seed: int, device: str):
    torch.manual_seed(seed+10000)
    if method.startswith("vg_"):
        return VariationalGarroteSAE(VGSAEConfig(input_dim=c["input_dim"],n_latents=c["width"],
            beta_mode="learned" if method=="vg_learned" else "profiled",beta=c["initial_beta"],
            lambda_sparsity=value,use_variance_term=method!="vg_no_variance",
            use_entropy_term=method!="vg_no_entropy")).to(device)
    common=dict(device=device,dtype="float32",normalize_activations="none",
                apply_b_dec_to_input=True)
    if method=="l1":
        return build_sae("standard",c["input_dim"],c["width"],l1_coefficient=value,**common)
    if method=="topk":
        return build_sae("topk",c["input_dim"],c["width"],k=int(value),**common)
    raise ValueError(method)


def joint_block(cfg: dict, seed: int, out: Path, device: str, deadline: float=math.inf) -> None:
    c=cfg["joint"]
    d=torch.as_tensor(make_unit_dictionary(c["input_dim"],c["width"],np.random.default_rng(seed)),dtype=torch.float64)
    data={role:sample_data(d,c[role+"_samples"],seed*100+offset,c["probability"],c["noise_std"],c["amplitude_scale"])
          for role,offset in [("train",21),("calibration",22),("test",23)]}
    grid=[]
    for method in c["methods"]:
        controls=c["l1_coefficients"] if method=="l1" else c["topk_values"] if method=="topk" else c["gammas"]
        grid.extend(dict(method=method,control=value,index=i) for i,value in enumerate(controls))
    write_json(out/"manifest.json",dict(seed=seed,seed_roles=dict(dictionary=seed,train=seed*100+21,
        calibration=seed*100+22,test=seed*100+23,initialization=seed+10000,batches=seed+20000),
        jobs=grid,config=c))
    progress=[]
    for job in grid:
        check_deadline(deadline)
        dest=out/f'{job["method"]}_{job["index"]}'
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/"result.json").exists():
            progress.append({**job,"status":"completed"})
            continue
        model=build_joint_model(job["method"],job["control"],c,seed,device)
        history=[]
        def snapshot(current,step):
            history.append(dict(step=step,**joint_metrics(current,data["calibration"],device,c["effective_code_threshold"])))
        write_json(out/"progress.json",progress+[{**job,"status":"running"}])
        timing=train_shared_batches(model,data["train"]["x"].to(device=device,dtype=torch.float32),
            steps=c["steps"],seed=seed+20000,lr=c["lr"],batch_size=c["batch_size"],
            checkpoints=tuple(c["checkpoints"]),snapshot=snapshot,
            before_step=lambda _model,_step: check_deadline(deadline))
        torch.save(model.state_dict(),dest/"final.pt")
        metrics=joint_metrics(model,data["test"],device,c["effective_code_threshold"])
        profiles=[]
        if isinstance(model,VariationalGarroteSAE):
            profiles=profile_batch_diagnostics(model,data["calibration"]["x"].to(device=device,dtype=torch.float32),c["profile_batch_sizes"])
        write_json(dest/"result.json",dict(block="joint",seed=seed,**job,metrics=metrics,
            history=history,profile_diagnostics=profiles,timing=timing,
            trainer=dict(optimizer="Adam",betas=[.9,.999],weight_decay=0.,gradient_clip=1.,
                lr_schedule="constant",dead_feature_window=1000,normalize_activations="none"),
            config=model.cfg.to_dict() if hasattr(model,"cfg") else {
                k:str(v) if isinstance(v,torch.dtype) else v for k,v in vars(model.config).items()}))
        progress.append({**job,"status":"completed"})
        write_json(out/"progress.json",progress)
        print(json.dumps(dict(block="joint",seed=seed,**job,**metrics)),flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def analyze(out: Path) -> None:
    records=[json.loads(path.read_text()) for path in out.glob("*/seed_*/*/result.json")]
    exact=[json.loads(path.read_text()) for path in out.glob("exact/seed_*/result.json")]
    joint=[dict(seed=r["seed"],method=r["method"],control=r["control"],index=r["index"],
                **r["metrics"],seconds=r["timing"]["training_and_snapshot_seconds"]) for r in records if r["block"]=="joint"]
    frozen=[dict(seed=r["seed"],coherence=r["coherence"],mean_refinement_drop=r["mean_refinement_drop"],
                 min_refinement_drop=r["min_refinement_drop"],mf_converged_fraction=r["mf_converged_fraction"],
                 encoder_mean_residual=r["encoder_mean_residual"],encoder_max_residual=r["encoder_max_residual"],
                 **{row["method"]+"_"+metric:row[metric] for row in r["rows"]
                    for metric in ["brier","marginal_nll","marginal_mse_to_exact"]},
                 **{row["method"]+"_kl":row["reverse_kl"] for row in r["rows"]}) for r in records if r["block"]=="frozen"]
    exact_rows=[row for r in exact for row in r["rows"]]
    profiles=[dict(seed=r["seed"],method=r["method"],control=r["control"],**p) for r in records if r["block"]=="joint" for p in r["profile_diagnostics"]]
    for name,rows in [("joint",joint),("frozen",frozen),("exact",exact_rows),("profile",profiles)]:
        write_csv(out/(name+"_summary.csv"),rows)
    pairs=[]
    indexed={(r["seed"],r["method"],r["control"]):r for r in joint}
    for row in joint:
        if row["method"] not in {"vg_no_entropy","vg_no_variance","vg_learned"}:
            continue
        baseline=indexed.get((row["seed"],"vg_profiled",row["control"]))
        if baseline:
            pairs.append(dict(seed=row["seed"],variant=row["method"],gamma=row["control"],
                **{k+"_delta":row[k]-baseline[k] for k in ["hard_mse","full_stochastic_mse","hard_l0","expected_mask_count","signed_dictionary_cosine","support_f1"]}))
    write_csv(out/"paired_differences.csv",pairs)
    seconds=sum(r["timing"]["training_and_snapshot_seconds"] for r in records)
    worker_records=[json.loads(p.read_text()) for p in out.glob("*/seed_*/worker_status.json")]
    summary=dict(exact_workers=len(exact),exact_cells=len(exact_rows),
        exact_controls_valid=bool(exact) and all(r["control_valid"] for r in exact),
        frozen_fits=len(frozen),joint_fits=len(joint),training_device_hours=seconds/3600,
        assigned_gpu_wall_hours=sum(r["elapsed_seconds"] for r in worker_records if r["device"].startswith("cuda"))/3600,
        failed_workers=[r for r in worker_records if r["status"]!="completed"],
        min_profile_jensen_gap=min((p["jensen_gap"] for p in profiles),default=None),
        limitations=["Initial three-world campaign; no SOTA or calibrated semantic-posterior claim.",
                     "Numerical exact identities are validation, not new theory.",
                     "Best-found mean field has no global certificate; optimization residuals are reported."])
    write_json(out/"summary.json",summary)
    print(json.dumps(summary),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/first_principles_20260924.json"))
    parser.add_argument("--block",choices=["exact","frozen","joint","analyze"],required=True)
    parser.add_argument("--seed",type=int,default=2401)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/first_principles_20260924"))
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("Experiment outputs must stay inside the project.")
    if args.smoke:
        cfg=copy.deepcopy(cfg)
        cfg["exact"].update(samples=16,coherences=[0.,.95],betas=[8.],gammas=[math.log(3)],max_sweeps=30)
        cfg["frozen"].update(train_samples=256,calibration_samples=64,test_samples=64,steps=30,coherences=[0.,.95],max_sweeps=30)
        cfg["joint"].update(train_samples=256,calibration_samples=256,test_samples=64,steps=30,checkpoints=[30],gammas=[2.],l1_coefficients=[.01],topk_values=[2])
    if args.block=="analyze":
        analyze(out)
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    activate_worker_device(args.device)
    dest=out/args.block/f"seed_{args.seed}"
    dest.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    fraction=cfg["gpu_budget_fractions"].get("smoke" if args.smoke else args.block,0.)
    time_limit=cfg["gpu_hour_limit"]*3600*fraction/len(cfg["seeds"])
    deadline=started+time_limit if args.device.startswith("cuda") and fraction else math.inf
    spec=dict(config=cfg,block=args.block,seed=args.seed,device=args.device,smoke=args.smoke)
    if (dest/"invocation.json").exists() and json.loads((dest/"invocation.json").read_text())!=spec:
        raise ValueError("Existing run has different configuration; choose a new output directory.")
    write_json(dest/"invocation.json",spec)
    write_json(dest/"environment.json",dict(python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,cuda=torch.version.cuda,
        sae_lens=importlib.metadata.version("sae-lens"),
        sae_lens_revision=json.loads(importlib.metadata.distribution("sae-lens").read_text("direct_url.json"))["vcs_info"]["commit_id"],
        git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
        worker_time_limit_seconds=time_limit,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None))
    try:
        if args.block=="exact":
            exact_block(cfg,args.seed,dest)
        elif args.block=="frozen":
            frozen_block(cfg,args.seed,dest,args.device,deadline)
        else:
            joint_block(cfg,args.seed,dest,args.device,deadline)
        check_deadline(deadline)
    except Exception as exc:
        write_json(dest/"worker_status.json",dict(status="failed",block=args.block,seed=args.seed,
            device=args.device,elapsed_seconds=time.perf_counter()-started,error=str(exc)))
        raise
    write_json(dest/"worker_status.json",dict(status="completed",block=args.block,seed=args.seed,
        device=args.device,elapsed_seconds=time.perf_counter()-started))
    write_json(dest/"completed.json",dict(elapsed_seconds=time.perf_counter()-started,
        peak_memory_mib=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith("cuda") else 0))


if __name__=="__main__":
    main()
