"""compare_theory.py - overlay theory on simulation results."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

L = 100
TRIALS = 100
rng = np.random.default_rng(3)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------- theory ----------
def theory_coverage(N, Rs):
    """Expected covered fraction, uniform random deployment."""
    return 1 - np.exp(-(N / L**2) * np.pi * Rs**2)


def theory_connected(N, Rc):
    """Penrose / Gupta-Kumar: P(connected) ~ exp(-N * exp(-rho*pi*Rc^2))."""
    return np.exp(-N * np.exp(-(N / L**2) * np.pi * Rc**2))


# ---------- simulation with wrap-around edges (no boundary effect) ----------
def torus_coverage(N, Rs, res=2.0):
    pts = rng.uniform(0, L, (N, 2))
    xs = (np.arange(int(L / res)) + 0.5) * res
    gx, gy = np.meshgrid(xs, xs)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    d, _ = cKDTree(pts, boxsize=L).query(grid)
    return float(np.mean(d <= Rs))


def torus_connected(N, Rc):
    pts = rng.uniform(0, L, (N, 2))
    diff = np.abs(pts[:, None, :] - pts[None, :, :])
    diff = np.minimum(diff, L - diff)
    adj = (diff ** 2).sum(-1) <= Rc ** 2
    return float(connected_components(adj, directed=False)[0] == 1)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------- A: coverage ----------
df1 = pd.read_csv("results/coverage_vs_N.csv")
df1["theory"] = theory_coverage(df1["N"], df1["Rs"])
df1["torus"] = [np.mean([torus_coverage(int(n), r) for _ in range(TRIALS)])
                for n, r in zip(df1["N"], df1["Rs"])]
df1.to_csv("results/theory_coverage.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for c, (Rs, g) in zip(colors, df1.groupby("Rs")):
    ax.plot(g["N"], g["coverage"], "o-", color=c, label=f"Rs = {Rs} m")
    ax.plot(g["N"], g["torus"], "x", color=c)
    ax.plot(g["N"], g["theory"], "--", color=c)
ax.plot([], [], "kx", label="no-boundary simulation")
ax.plot([], [], "k--", label="theory")
ax.set(xlabel="Number of nodes N", ylabel="Covered area fraction",
       title="Coverage: simulation vs theory", ylim=(0, 1.02))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/theory_coverage.png", dpi=200)
plt.close(fig)

# ---------- B: connectivity ----------
df2 = pd.read_csv("results/connectivity_vs_Rc.csv")
df2["theory"] = theory_connected(df2["N"], df2["Rc"])
df2["torus"] = [np.mean([torus_connected(int(n), r) for _ in range(TRIALS)])
                for n, r in zip(df2["N"], df2["Rc"])]
df2.to_csv("results/theory_connectivity.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for c, (N, g) in zip(colors, df2.groupby("N")):
    ax.plot(g["Rc"], g["connected"], "o-", markersize=3, color=c, label=f"N = {N}")
    ax.plot(g["Rc"], g["torus"], "x", markersize=4, color=c)
    ax.plot(g["Rc"], g["theory"], "--", color=c)
ax.plot([], [], "kx", label="no-boundary simulation")
ax.plot([], [], "k--", label="theory")
ax.set(xlabel="Communication range Rc (m)", ylabel="Probability network is connected",
       title="Connectivity: simulation vs theory", ylim=(0, 1.02))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/theory_connectivity.png", dpi=200)
plt.close(fig)

# ---------- summary ----------
print("Coverage     RMSE vs theory:  square area = %.4f   no-boundary = %.4f"
      % (rmse(df1["coverage"], df1["theory"]), rmse(df1["torus"], df1["theory"])))
print("Connectivity RMSE vs theory:  square area = %.4f   no-boundary = %.4f"
      % (rmse(df2["connected"], df2["theory"]), rmse(df2["torus"], df2["theory"])))
print("Saved figures/theory_coverage.png and figures/theory_connectivity.png")
