"""make_plots.py - reads results/*.csv and writes figures/*.png."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed; just save files
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)


def finish(ax, fig, name, xlabel, ylabel, title, ylim=None):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=200)
    plt.close(fig)
    print("saved figures/" + name + ".png")


# Plot 1: coverage vs N for each Rs
df = pd.read_csv("results/coverage_vs_N.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for Rs, g in df.groupby("Rs"):
    ax.errorbar(g["N"], g["coverage"], yerr=g["coverage_std"],
                marker="o", capsize=3, label=f"Rs = {Rs} m")
finish(ax, fig, "coverage_vs_N", "Number of nodes N",
       "Covered area fraction", "Coverage vs number of nodes", (0, 1.02))

# Plots 2-4: connectivity metrics vs Rc for each N
df = pd.read_csv("results/connectivity_vs_Rc.csv")
panels = [
    ("connected", "p_connected_vs_Rc", "Probability network is connected",
     "Connectivity probability vs communication range", (0, 1.02)),
    ("largest_frac", "largest_component_vs_Rc", "Largest component (fraction of nodes)",
     "Largest connected component vs communication range", (0, 1.02)),
    ("avg_degree", "avg_degree_vs_Rc", "Average node degree",
     "Average degree vs communication range", None),
]
for col, name, ylabel, title, ylim in panels:
    fig, ax = plt.subplots(figsize=(6, 4))
    for N, g in df.groupby("N"):
        ax.plot(g["Rc"], g[col], marker="o", markersize=3, label=f"N = {N}")
    finish(ax, fig, name, "Communication range Rc (m)", ylabel, title, ylim)
