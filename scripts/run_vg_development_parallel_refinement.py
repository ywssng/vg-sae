"""P3b exploratory single parallel VG update and matched-product amplitude PG.

This is a post-P3 exploratory follow-up. No training or test-based selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_vg_development_holdout import (
    prepare_model, sample_fixed_dictionary, save_csv, save_json,
    sha256_file, tensors, utc_now,
)
from scripts.run_vg_development_refinement import (
    encode_state, exact_count_mask, fixed_objective, model_dictionary, support_nnls,
)
from src.sae_model import VariationalGarroteSAE
from src.sae_sweep import SweepConfig

P3 = ROOT / "outputs/vg_sae_development_20260921/refinement"
HOLDOUT = ROOT / "outputs/vg_sae_development_20260921/holdout"
OUT = ROOT / "outputs/vg_sae_development_20260921/parallel_refinement"
PROTOCOL = ROOT / "idea-stage/runs/vg-sae-development-20260921/pilots/parallel_followup_protocol.md"
ETAS = (0., .25, .5, 1.)
CAL_SEED, TEST_SEED = 2026092194, 2026092195
N_SAMPLES, BATCH_SIZE = 512, 128


@torch.no_grad()
def parallel_refinement(x, m, a, decoder, bias, beta, gamma, eta, *, scores=None):
    """One damped Jacobi step. It has no free-energy monotonicity guarantee."""
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0,1]")
    if scores is None:
        scores = torch.logit(m)
    if eta == 0:
        return m.clone(), scores.clone()
    norms = decoder.square().sum(0)
    residual = x-bias-(m*a)@decoder.T
    excluded_inner_product = residual@decoder + m*a*norms
    target_scores = beta*(a*excluded_inner_product-.5*a.square()*norms)-gamma
    target = target_scores.sigmoid()
    if eta == 1:
        return target, target_scores
    updated = (1-eta)*m+eta*target
    # Rank the mixture, with scores retaining information at sigmoid saturation.
    log_probability = torch.logaddexp(math.log1p(-eta)+F.logsigmoid(scores),
                                     math.log(eta)+F.logsigmoid(target_scores))
    log_complement = torch.logaddexp(math.log1p(-eta)+F.logsigmoid(-scores),
                                    math.log(eta)+F.logsigmoid(-target_scores))
    return updated, log_probability-log_complement


@torch.no_grad()
def support_projected_gradient(x, code, decoder, bias, support):
    """One nonnegative hard-SSE step with per-example support Gram trace bound."""
    trace = (support*decoder.square().sum(0)).sum(1)
    # Includes empty support and all-zero selected decoder columns.
    safe_trace = torch.where(trace > 0, trace, torch.ones_like(trace))
    step = (trace > 0).to(code.dtype)/safe_trace
    residual = x-bias-code@decoder.T
    return (code+step[:,None]*(residual@decoder)).clamp_min(0)*support


@torch.no_grad()
def infer_arm(model, x, eta, readout, beta):
    d,b = model_dictionary(model)
    m,a,scores = encode_state(model,x)
    source_support = (m>=.5) if m is not None else a>0
    if readout.startswith("count_"):
        if m is None:
            raise ValueError("Gate count arm requires VG")
        m,scores = parallel_refinement(x,m,a,d,b,beta,model.config.lambda_sparsity,eta,scores=scores)
        support = exact_count_mask(scores,source_support.sum(1))
    else:
        if eta:
            raise ValueError("Native support arms have eta0")
        support = source_support
    code = a*support
    if readout == "native_pg":
        code = support_projected_gradient(x,code,d,b,support)
    elif readout.endswith("nnls"):
        fitted = support_nnls(x.cpu().numpy(),d.cpu().numpy(),b.cpu().numpy(),support.cpu().numpy())
        code = torch.as_tensor(fitted,device=x.device,dtype=x.dtype)
    reconstruction = model.decode(code)
    return code,support,reconstruction,m,a


def arms(is_vg):
    result = [(0.,r) for r in ("native_a","native_pg","native_nnls")]
    if is_vg:
        result += [(eta,"count_a") for eta in ETAS]
        result += [(eta,"count_nnls") for eta in ETAS[1:]]
    return result


@torch.no_grad()
def evaluate_arm(model,data,matching,eta,readout,beta):
    sums = dict(sse_z=0.,target_z=0.,sse_x=0.,actual_count=0.,candidate_count=0.,tp=0.,fp=0.,fn=0.,
                free_energy=0.,mean_residual_half_sse=0.,bernoulli_variance_half_sse=0.,
                stochastic_half_sse=0.,expected_l0=0.)
    device = data["x"].device
    li = torch.as_tensor(matching["learned_idx"],device=device)
    ti = torch.as_tensor(matching["true_idx"],device=device)
    signs = torch.as_tensor(matching["signs"],device=device)
    d,b = model_dictionary(model)
    report_conditional = isinstance(model,VariationalGarroteSAE) and readout != "native_pg"
    for begin in range(0,len(data["x"]),BATCH_SIZE):
        x,z,truth = (data[k][begin:begin+BATCH_SIZE] for k in ("x","z","support"))
        code,support,recon,m,a = infer_arm(model,x,eta,readout,beta)
        actual = code>0
        aligned = code[:,li].double()*signs
        target,pred,truth = z[:,ti].double(),actual[:,li],truth[:,ti]
        sums["sse_z"] += float((aligned-target).square().sum())
        sums["target_z"] += float(target.square().sum())
        sums["sse_x"] += float((recon.double()-x.double()).square().sum())
        sums["actual_count"] += int(actual.sum())
        sums["candidate_count"] += int(support.sum())
        sums["tp"] += int((pred&truth).sum())
        sums["fp"] += int((pred&~truth).sum())
        sums["fn"] += int((~pred&truth).sum())
        if report_conditional:
            terms = fixed_objective(x.double(),m.double(),a.double(),d.double(),b.double(),beta,
                                    model.config.lambda_sparsity)
            for k,value in terms.items():
                sums[k] += float(value.sum())
            sums["expected_l0"] += float(m.double().sum())
    x = data["x"].double()
    n,energy = len(x),float((x-x.mean(0)).square().sum())
    result = {"n_samples":n,"hard_latent_relative_error":math.sqrt(sums["sse_z"]/sums["target_z"]),
        "hard_support_f1":2*sums["tp"]/max(2*sums["tp"]+sums["fp"]+sums["fn"],1.),
        "hard_explained_variance":1-sums["sse_x"]/energy,"hard_reconstruction_mse":sums["sse_x"]/x.numel(),
        "actual_nonzero_l0":sums["actual_count"]/n,"candidate_support_l0":sums["candidate_count"]/n,
        "decoder_recovery_cosine":matching["matched_cosine"],"truth_l0":float(data["support"].sum())/n,
        "tp":sums["tp"],"fp":sums["fp"],"fn":sums["fn"]}
    if report_conditional:
        result.update({"conditional_"+k:sums[k]/n for k in ("free_energy","mean_residual_half_sse",
            "bernoulli_variance_half_sse","stochastic_half_sse","expected_l0")})
        result["conditional_metrics_scope"] = "probability state with original amplitude; no refit/PG coefficients"
    return result


@torch.no_grad()
def benchmark_arm(model,x,eta,readout,beta):
    for _ in range(2):
        infer_arm(model,x,eta,readout,beta)
    torch.cuda.synchronize(x.device)
    samples = []
    for _ in range(11):
        torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        infer_arm(model,x,eta,readout,beta)
        torch.cuda.synchronize(x.device)
        samples.append(time.perf_counter()-start)
    return {"median_batch_seconds":float(np.median(samples)),"samples_seconds":samples,
            "batch_size":len(x),"warmups":2,"repetitions":11}


def select_eta(rows):
    candidates = [r for r in rows if r["method"]=="vgsae" and r["readout"]=="count_a"]
    base = next(r for r in candidates if r["eta"]==0)
    feasible = [r for r in candidates if r["hard_support_f1"]>=base["hard_support_f1"]-.01
                and r["conditional_free_energy"]<=base["conditional_free_energy"]+1e-10]
    return min(feasible,key=lambda r:(r["hard_latent_relative_error"],r["eta"]))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--device",default="cuda:2")
    parser.add_argument("--max-seconds",type=float,default=900.)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUT):
        raise ValueError("Output must be inside assigned P3b directory")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"protocol.json").exists():
        raise FileExistsError("Refusing to overwrite a preregistered run")
    start = time.monotonic()
    def deadline():
        if time.monotonic()-start>args.max_seconds:
            raise TimeoutError("P3b hard wall-time limit")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    metadata = json.loads((P3/"checkpoint_manifest.json").read_text())
    choices = metadata["checkpoints"]
    protocol = {"created_at":utc_now(),"exploratory_after_P3":True,"last_followup_in_this_run":True,
        "markdown":str(PROTOCOL.relative_to(ROOT)),"markdown_sha256":sha256_file(PROTOCOL),
        "script_sha256":sha256_file(Path(__file__)),"p3_source_manifest_sha256":sha256_file(P3/"checkpoint_manifest.json"),
        "p3_helper_sha256":sha256_file(ROOT/"scripts/run_vg_development_refinement.py"),
        "cal_seed":CAL_SEED,"test_seed":TEST_SEED,"n_cal":N_SAMPLES,"n_test":N_SAMPLES,
        "eta_choices":ETAS,"batch_size":BATCH_SIZE,"device":args.device,"max_seconds":args.max_seconds,
        "baseline_arms":arms(False),"vg_arms":arms(True),"checkpoint_manifest":metadata,
        "selection":"Cal min count_a latent error subject F1>=base-.01 and mean conditional F<=base+1e-10; tie smaller eta"}
    save_json(out/"protocol.json",protocol)
    save_json(out/"environment.json",{"python":platform.python_version(),"torch":torch.__version__,
        "numpy":np.__version__,"scipy":scipy.__version__,"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(args.device),"torch_threads":torch.get_num_threads()})
    source_file = HOLDOUT/"exponential_frozen_source.npz"
    if sha256_file(source_file)!=metadata["source_frozen_sha256"]:
        raise RuntimeError("Frozen source changed since P3")
    source = np.load(source_file)
    dictionary,probabilities = source["dictionary"],source["feature_probabilities"]
    cfg = SweepConfig.from_dict(metadata["source_config"])
    kwargs = {"n_samples":N_SAMPLES,"amplitude_mode":cfg.data.amplitude_mode,
        "amplitude_scale":cfg.data.amplitude_scale,"noise_std":cfg.data.noise_std,"source_offset":0}
    cal_array = sample_fixed_dictionary(dictionary,probabilities,base_seed=CAL_SEED,**kwargs)
    np.savez_compressed(out/"calibration.npz",**cal_array)
    cal = tensors(cal_array,args.device)
    models,matchings,cal_rows,latencies = {},{},[],[]
    for choice in choices:
        deadline()
        key = choice["run_id"]
        path = ROOT/choice["checkpoint"]
        if sha256_file(path)!=choice["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed since P3")
        model,_,matching = prepare_model(path,torch.as_tensor(dictionary,device=args.device),args.device)
        if isinstance(model,VariationalGarroteSAE) and (not model.config.use_variance_term or not model.config.use_entropy_term
                or model.config.entropy_weight!=1 or not model.config.nonnegative_amplitudes):
            raise ValueError("This pilot requires full VG with nonnegative amplitudes")
        models[key],matchings[key] = model,matching
        saved_match = np.load(P3/f"{key}_matching.npz")
        for name in ("learned_idx","true_idx","signs"):
            np.testing.assert_array_equal(saved_match[name],matching[name])
        for eta,readout in arms(isinstance(model,VariationalGarroteSAE)):
            deadline()
            prefix = {"run_id":key,"method":choice["method"],"eta":eta,"readout":readout}
            cal_rows.append({**prefix,**evaluate_arm(model,cal,matching,eta,readout,choice["beta"])})
            timing = benchmark_arm(model,cal["x"][:BATCH_SIZE],eta,readout,choice["beta"])
            latencies.append({**prefix,**timing,
                "extra_dense_decoder_products":2 if eta>0 or readout=="native_pg" else 0,
                "nnls_internal_products":"not exposed by scipy" if readout.endswith("nnls") else 0})
        print(f"calibration+timing {key} elapsed={time.monotonic()-start:.2f}s",flush=True)
    save_json(out/"calibration_metrics.json",cal_rows)
    save_csv(out/"calibration_metrics.csv",cal_rows)
    save_json(out/"latencies.json",latencies)
    selected = select_eta(cal_rows)
    save_json(out/"frozen_selection.json",{"frozen_at":utc_now(),"test_generated":False,
        "selected_eta":selected["eta"],"selected_readout":"count_a","selected_calibration_row":selected,
        "selection_rule":protocol["selection"],"calibration_metrics_sha256":sha256_file(out/"calibration_metrics.json")})
    frozen_hash = sha256_file(out/"frozen_selection.json")
    print(f"Selection frozen eta={selected['eta']}; test now generated",flush=True)
    test_array = sample_fixed_dictionary(dictionary,probabilities,base_seed=TEST_SEED,**kwargs)
    np.savez_compressed(out/"test.npz",**test_array)
    test = tensors(test_array,args.device)
    test_rows = []
    for choice in choices:
        key,model = choice["run_id"],models[choice["run_id"]]
        for eta,readout in arms(isinstance(model,VariationalGarroteSAE)):
            deadline()
            prefix = {"run_id":key,"method":choice["method"],"eta":eta,"readout":readout,
                "cal_selected_policy":choice["method"]=="vgsae" and eta==selected["eta"] and readout=="count_a"}
            test_rows.append({**prefix,**evaluate_arm(model,test,matchings[key],eta,readout,choice["beta"])})
    if sha256_file(out/"frozen_selection.json")!=frozen_hash:
        raise RuntimeError("Frozen selection changed")
    save_json(out/"test_metrics.json",test_rows)
    save_csv(out/"test_metrics.csv",test_rows)
    save_json(out/"completion.json",{"completed_at":utc_now(),"status":"complete","wall_seconds":time.monotonic()-start,
        "device_allocation_hours":(time.monotonic()-start)/3600,"unique_checkpoints":len(choices),"test_arms":len(test_rows),
        "frozen_selection_sha256":frozen_hash,"script_sha256":sha256_file(Path(__file__)),
        "calibration_npz_sha256":sha256_file(out/"calibration.npz"),"test_npz_sha256":sha256_file(out/"test.npz"),
        "test_selection_performed":False,"training_performed":False,"added_parameters":0})
    print(f"P3b complete {len(test_rows)} arms elapsed={time.monotonic()-start:.2f}s",flush=True)


if __name__=="__main__":
    main()
