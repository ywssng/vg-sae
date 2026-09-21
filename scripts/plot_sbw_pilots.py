"""Descriptive plots of frozen paper-anchored pilot results; no model selection."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/sbw_20260922/mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = ROOT / "idea-stage/runs/vg-sae-sparse-but-wrong-20260922/figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10})
    rows = []
    for world in [210, 211, 212]:
        rows.extend(json.loads((ROOT / f"outputs/sbw_20260922/prior/world{world}/test_results.json").read_text())["metrics"])
    fig, axes = plt.subplots(2, 3, figsize=(8, 6.5), sharex=True)
    fig.subplots_adjust(top=.81, bottom=.17, left=.085, right=.985, hspace=.36, wspace=.40)
    metrics = [("matched_positive_cosine", "Signed cosine", (0, 1.04)),
               ("support_f1", "Support F1", (0, 1.04)),
               ("hard_l0", "Native hard L0", (0, 5.25))]
    colors = {"fixed": "#276FBF", "learned": "#D46A1F"}
    for row_index, rho in enumerate([-.4, .4]):
        for column, (metric, label, ylim) in enumerate(metrics):
            ax = axes[row_index, column]
            for position, probability in enumerate([.2, .4, .6]):
                for world in [210, 211, 212]:
                    pair = []
                    for kind, offset in [("fixed", -.10), ("learned", .10)]:
                        record = next(x for x in rows if x["world"] == world and x["correlation"] == rho
                            and x["firing_probability"] == probability and x["prior_kind"] == kind
                            and x["gamma_initial"] == 2)
                        x = position + offset + (world-211)*.025
                        y = record[metric]
                        pair.append((x, y))
                        ax.scatter(x, y, s=22, color=colors[kind], alpha=.85,
                            label=kind if position == 0 and world == 210 else None, zorder=3)
                    ax.plot([p[0] for p in pair], [p[1] for p in pair], color=".68", lw=.7, zorder=1)
                if metric == "support_f1":
                    ax.plot([position-.2, position+.2], [2*probability/(1+probability)]*2, color=".4", ls=":", lw=1)
                if metric == "hard_l0":
                    ax.plot([position-.2, position+.2], [5*probability]*2, color=".4", ls=":", lw=1)
            ax.set(ylim=ylim, xticks=[0, 1, 2], xticklabels=[".2", ".4", ".6"], ylabel=label)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 1:
                ax.set_xlabel("True firing probability")
            if column == 0:
                ax.set_title(f"Copula rho = {rho:+.1f}", loc="left")
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["Fixed gamma 2", "Learned gamma (init 2)"], loc="upper center",
               bbox_to_anchor=(.5, .93), ncols=2, frameon=False)
    fig.suptitle("P2: scalar-prior adaptation", y=.985, fontsize=13)
    fig.text(.5, .945, "All 3 paired worlds; final checkpoints", ha="center", fontsize=9)
    fig.text(.5, .045, "Dotted: all-active F1 (middle); true mean L0 (right).\nConnections pair the same training world.", ha="center", fontsize=9)
    fig.savefig(out / "prior_adaptation.png", dpi=160)
    plt.close(fig)

    joint = json.loads((ROOT / "outputs/sbw_20260922/joint/test_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    fig.subplots_adjust(top=.70, bottom=.22, left=.08, right=.98, wspace=.28)
    worlds = [(rho, world) for rho in [-.4, .4] for world in [210, 211, 212]]
    styles = {"amortized": ("#276FBF", "o"), "optimized_mf": ("#D46A1F", "s"), "exact32": ("#23845A", "^")}
    for ax, key, title in zip(axes, ["matched_positive_cosine", "matched_abs_cosine"],
                              ["Signed feature identity", "Axis alignment, ignoring sign"]):
        for index, (arm, (color, marker)) in enumerate(styles.items()):
            values = [next(x for x in joint if x["rho"] == rho and x["world"] == world
                and x["arm"] == arm and x["gamma"] == 0)["test"][key] for rho, world in worlds]
            ax.scatter(np.arange(6)+(index-1)*.18, values, color=color, marker=marker, label=arm, s=35)
        ax.set(xticks=range(6), xticklabels=[f"{rho:+.1f}\n{world}" for rho, world in worlds],
               ylim=(0, 1.06), xlabel="Copula rho / world", ylabel="Matched cosine", title=title)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .88), ncols=3, frameon=False)
    fig.suptitle("P3: posterior routes at predeclared gamma 0", y=.99, fontsize=12)
    fig.text(.5, .905, "Same-gamma total effects; not L0-matched", ha="center", fontsize=9)
    fig.savefig(out / "joint_signed_vs_absolute.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
