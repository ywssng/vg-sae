"""Plot the complete registered campaign from its exported CSV tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, base: Path) -> None:
    for extension in ["png", "svg"]:
        path = base.with_suffix("." + extension)
        fig.savefig(path)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def plot(out: Path) -> None:
    exact=rows(out/"exact_summary.csv")
    frozen=rows(out/"frozen_summary.csv")
    joint=rows(out/"joint_summary.csv")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "savefig.dpi":180,"svg.fonttype":"none"})
    figures=out/"figures"
    figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for coherence,color in [(0.,"#426b9e"),(.7,"#b88932"),(.95,"#a8464e")]:
        subset=[r for r in exact if float(r["coherence"])==coherence and float(r["beta"])==8.]
        xs=sorted({float(r["gamma"]) for r in subset})
        means=[np.mean([float(r["exact_expected_l0"]) for r in subset if float(r["gamma"])==x]) for x in xs]
        axes[0].plot(xs,means,"o-",label=f"cosine={coherence:g}",color=color)
    axes[0].set(xlabel="Selection cost gamma",ylabel="Exact expected support count",title="Fixed-input response (beta=8)")
    axes[0].legend(fontsize=8)
    matched=[r for r in exact if r["model_matched"]=="True"]
    for seed in sorted({r["seed"] for r in matched}):
        subset=sorted([r for r in matched if r["seed"]==seed],key=lambda r:float(r["coherence"]))
        axes[1].plot([float(r["coherence"]) for r in subset],[float(r["mf_reverse_kl"]) for r in subset],"o-",alpha=.7,label=seed)
    axes[1].set(xlabel="Dictionary-pair cosine",ylabel="Best-found MF reverse KL (nats)",title="Factorization gap, matched model")
    axes[1].legend(title="Data seed",fontsize=8)
    for method,color,label in [("encoder_kl","#426b9e","Trained encoder"),("mean_field_kl","#a8464e","Best-found mean field")]:
        xs=sorted({float(r["coherence"]) for r in frozen})
        means=[np.mean([float(r[method]) for r in frozen if float(r["coherence"])==x]) for x in xs]
        sds=[np.std([float(r[method]) for r in frozen if float(r["coherence"])==x],ddof=1) for x in xs]
        axes[2].errorbar(xs,means,yerr=sds,marker="o",capsize=3,color=color,label=label)
    axes[2].set(xlabel="Dictionary-pair cosine",ylabel="Reverse KL to exact (nats)",title="Frozen encoder: mean +/- seed SD")
    axes[2].legend(fontsize=8)
    save_figure(fig, figures/"conditional_inference")
    plt.close(fig)

    methods=["vg_learned","vg_profiled","vg_no_entropy","vg_no_variance"]
    labels=["Learned beta","Profiled beta","No entropy","No variance"]
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    x=np.arange(4)
    for field,offset,color,label in [("mean_mse",-.17,"#426b9e","Posterior mean"),("full_stochastic_mse",.17,"#a8464e","Full stochastic risk")]:
        values=[[float(r[field]) for r in joint if r["method"]==m and float(r["control"])==2.] for m in methods]
        axes[0].plot(x+offset,[np.mean(v) for v in values],"_",markersize=15,color=color,label=label)
        for i,v in enumerate(values):
            axes[0].scatter(i+offset+np.linspace(-.025,.025,len(v)),v,s=22,color=color,alpha=.7)
    axes[0].set(xticks=x,xticklabels=labels,ylabel="Per-coordinate squared error",title="Joint models at gamma=2")
    axes[0].tick_params(axis="x",labelrotation=25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    for field,offset,color,label in [("expected_mask_count",-.17,"#426b9e","Expected mask count"),("hard_l0",.17,"#a8464e","Hard code L0")]:
        values=[[float(r[field]) for r in joint if r["method"]==m and float(r["control"])==2.] for m in methods]
        axes[1].bar(x+offset,[np.mean(v) for v in values],width=.32,yerr=[np.std(v,ddof=1) for v in values],capsize=2,color=color,label=label)
    axes[1].axhline(2,color="#555555",linestyle=":",label="Generating expected count")
    axes[1].set(xticks=x,xticklabels=labels,ylabel="Latents per sample",title="Different support readouts")
    axes[1].tick_params(axis="x",labelrotation=25)
    axes[1].legend(fontsize=8)
    colors=["#426b9e","#77a9cf","#9cbcc4","#a8464e","#b88932","#555555"]
    for method,color in zip(methods+["l1","topk"],colors):
        subset=[r for r in joint if r["method"]==method]
        axes[2].scatter([float(r["hard_l0"]) for r in subset],[float(r["hard_mse"]) for r in subset],s=25,color=color,label=method.replace("vg_",""),alpha=.85)
    axes[2].set(xlabel="Native hard-code L0",ylabel="Native reconstruction MSE",title="All final grid points (no selection)")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7)
    save_figure(fig, figures/"joint_objectives")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/first_principles_20260924"))
    plot(parser.parse_args().output_dir)
