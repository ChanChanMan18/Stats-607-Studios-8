import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

def harmonize_columns(df):
    rename_map = {"n_sim": "nsim", 
                  "MSE": "MSE_hat",
                  "se(MSE)": "se_MSE_hat"}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    return df

def produce_figure(df, output_dir):
    """
    Create the three-panel plot and save to figure.pdf and figure.png in outdir.
    """

    panels = [(1, "nsim = 1"), 
              (50, "nsim = 50"),
              (1000, "nsim = 1000")]
    x_min, x_max = 0.1, 10.0

    fig, axes = plt.subplots(3, 1, figsize = (7.0, 11.0), sharey=True)

    for ax, (nsim_val, title) in zip(axes, panels):
        d = df[df["nsim"] == nsim_val].copy().sort_values("gamma")

        # Theoretical curve  
        gL = np.logspace(np.log10(x_min), np.log10(0.9), 400)
        gR = np.logspace(np.log10(1.1), np.log10(x_max), 400)
        sigma2, r2 = 1.0, 5.0
        tL = sigma2 * gL / (1.0 - gL)
        tR = r2 * (1.0 - 1.0 / gR) + sigma2 / (gR - 1.0)
        ax.plot(gL, tL, linewidth=2, color="black", label="Theory")
        ax.plot(gR, tR, linewidth=2, color="black")

        # Scatter  (blue with some transparency to show overlap)
        ms = 18 if nsim_val > 1 else 14
        ax.scatter(d["gamma"], d["MSE_hat"], s=ms, color="tab:blue", alpha=0.5,
                   zorder=3, label="Estimated risk")
        
        # Error bars for nsim > 1  (match blue, slightly transparent)
        if nsim_val > 1 and "se_MSE_hat" in d.columns:
            ax.errorbar(
                d["gamma"], d["MSE_hat"], 
                yerr=2.0 * d["se_MSE_hat"],
                fmt="none", capsize=2, linewidth=1,
                color="tab:blue", ecolor="tab:blue", alpha=0.4,
                zorder=2, label="± 2 SE"
            )

        # Formatting
        ax.axvline(1.0, linestyle="--", linewidth=1, color="tab:gray")  # vertical asymptote at γ=1
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_title(title)

    axes[-1].set_xlabel(r"$\gamma = p/n$")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Ridgeless Least Squares: Estimated Risk vs. $\\gamma$", y=0.995, fontsize=12)
    fig.supylabel(r"Risk  $\widehat{\mathrm{MSE}}$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_png = output_dir / "figure.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved {out_png}")
    return out_png

def main():
    
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    pkl_path = (here / ".." / "data" / "simulation_results.pkl").resolve()

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    
    df = harmonize_columns(df)

    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    output_dir = (base / ".." / "figures").resolve()

    produce_figure(df, output_dir)

if __name__ == "__main__":
    main()