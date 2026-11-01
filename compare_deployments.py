"""compare_deployments.py - random vs square vs triangular deployment,
the Rc/Rs rule, and robustness to random node failures."""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wsn_core import deploy_random, coverage, build_graph, connectivity, sink_index

L, Rs = 100, 10
TARGET = 0.99
rng = np.random.default_rng(4)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def lattice(kind, d):
    """Square or triangular lattice with nearest-neighbour spacing d, clipped to the area."""
    if kind == "square":
        xs = np.arange(d / 2, L, d)
        gx, gy = np.meshgrid(xs, xs)
        return np.column_stack([gx.ravel(), gy.ravel()])
    h = d * np.sqrt(3) / 2                      # row spacing of a triangular lattice
    rows = []
    for j, y in enumerate(np.arange(h / 2, L, h)):
        x0 = d / 4 + (d / 2 if j % 2 else 0)    # alternate rows are shifted by d/2
        xs = np.arange(x0, L, d)
        rows.append(np.column_stack([xs, np.full_like(xs, y)]))
    return np.vstack(rows)


# ================= A: nodes needed for 99% coverage =================
rows = []
for kind in ("square", "triangular"):
    for d in np.arange(6, 40.01, 0.25):
        pts = lattice(kind, d)
        rows.append({"deployment": kind, "nodes": len(pts), "spacing": round(float(d), 2),
                     "coverage": coverage(pts, L, Rs)})
for N in range(20, 341, 10):
    covs = [coverage(deploy_random(N, L, rng), L, Rs) for _ in range(50)]
    rows.append({"deployment": "random", "nodes": N, "spacing": np.nan,
                 "coverage": float(np.mean(covs))})
dfA = pd.DataFrame(rows)
dfA.to_csv("results/deployments.csv", index=False)

best = {}
for kind, g in dfA.groupby("deployment"):
    ok = g[g["coverage"] >= TARGET]
    best[kind] = ok.sort_values("nodes").iloc[0] if len(ok) else g.sort_values("coverage").iloc[-1]

print(f"Nodes needed for >= {TARGET:.0%} coverage (Rs = {Rs} m, {L}x{L} m area):")
for kind in ("triangular", "square", "random"):
    b = best[kind]
    extra = f", spacing {b['spacing']} m = {b['spacing'] / Rs:.2f} Rs" if kind != "random" else ""
    print(f"  {kind:11s} {int(b['nodes']):4d} nodes  (coverage {b['coverage']:.3f}{extra})")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for kind, g in dfA.groupby("deployment"):
    g = g.sort_values("nodes")
    if kind == "random":
        ax.plot(g["nodes"], g["coverage"], "-", label="random (mean of 50 runs)")
    else:
        ax.plot(g["nodes"], g["coverage"], "o", markersize=3, label=kind + " lattice")
ax.axhline(TARGET, color="gray", ls=":", lw=1)
ax.set(xlabel="Number of nodes", ylabel="Covered area fraction",
       title=f"Coverage vs nodes by deployment (Rs = {Rs} m)", ylim=(0, 1.02))
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/coverage_by_deployment.png", dpi=200)
plt.close(fig)

# layouts at the 99% point, drawn with their sensing disks
random_N = int(best["random"]["nodes"])
layouts = {
    "triangular": lattice("triangular", best["triangular"]["spacing"]),
    "square": lattice("square", best["square"]["spacing"]),
    "random": deploy_random(random_N, L, rng),
}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (name, pts) in zip(axes, layouts.items()):
    for x, y in pts:
        ax.add_patch(plt.Circle((x, y), Rs, alpha=0.12, lw=0))
    ax.plot(pts[:, 0], pts[:, 1], "k.", markersize=3)
    ax.set(xlim=(0, L), ylim=(0, L), aspect="equal",
           title=f"{name}: {len(pts)} nodes, coverage {coverage(pts, L, Rs):.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig("figures/deployments_layout.png", dpi=200)
plt.close(fig)


# ================= B: connectivity vs Rc/Rs =================
ratios = np.round(np.arange(0.5, 2.51, 0.05), 2)


def p_connected(make_pts, Rc, trials):
    ok = sum(nx.is_connected(build_graph(make_pts(), Rc)) for _ in range(trials))
    return ok / trials


rowsB = []
for name, make_pts, trials in (
    ("triangular", lambda: layouts["triangular"], 1),
    ("square", lambda: layouts["square"], 1),
    ("random", lambda: deploy_random(random_N, L, rng), 100),
):
    for r in ratios:
        rowsB.append({"deployment": name, "Rc_over_Rs": r,
                      "connected": p_connected(make_pts, r * Rs, trials)})
dfB = pd.DataFrame(rowsB)
dfB.to_csv("results/rc_rs_ratio.csv", index=False)

print("\nSmallest Rc/Rs giving P(connected) >= 0.99:")
for name, g in dfB.groupby("deployment"):
    ok = g[g["connected"] >= 0.99]
    print(f"  {name:11s} {ok['Rc_over_Rs'].min() if len(ok) else 'not reached'}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, g in dfB.groupby("deployment"):
    ax.plot(g["Rc_over_Rs"], g["connected"], "o-", markersize=3, label=name)
for r, lab in ((np.sqrt(2), "sqrt(2)"), (np.sqrt(3), "sqrt(3)"), (2.0, "2")):
    ax.axvline(r, color="gray", ls=":", lw=1)
    ax.text(r, 0.03, lab, ha="center", fontsize=8, backgroundcolor="white")
ax.set(xlabel="Rc / Rs", ylabel="Probability network is connected",
       title="Connectivity vs Rc/Rs (layouts with ~99% coverage)", ylim=(0, 1.02))
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig("figures/rc_rs_ratio.png", dpi=200)
plt.close(fig)


# ================= C: robustness to random node failures =================
fracs = np.round(np.arange(0, 0.51, 0.05), 2)
Rc_fail = 2 * Rs
TRIALS = 100
variants = {
    "triangular": lambda: layouts["triangular"],
    "square": lambda: layouts["square"],
    "random": lambda: deploy_random(random_N, L, rng),
    "random (same nodes as triangular)":
        lambda: deploy_random(len(layouts["triangular"]), L, rng),
}
rowsC = []
for name, make_pts in variants.items():
    for f in fracs:
        cov, lc, conn = [], [], []
        for _ in range(TRIALS):
            pts = make_pts()
            keep = rng.choice(len(pts), size=int(round(len(pts) * (1 - f))), replace=False)
            alive = pts[keep]
            cov.append(coverage(alive, L, Rs, res=2.0))
            m = connectivity(alive, Rc_fail, sink_index(alive, L))
            lc.append(m["largest_frac"])
            conn.append(m["connected"])
        rowsC.append({"deployment": name, "failed_fraction": f, "coverage": np.mean(cov),
                      "largest_frac": np.mean(lc), "connected": np.mean(conn)})
dfC = pd.DataFrame(rowsC)
dfC.to_csv("results/robustness.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panels = [("coverage", "Covered area fraction"),
          ("largest_frac", "Largest component (fraction of survivors)"),
          ("connected", "Probability network is connected")]
for ax, (col, ylabel) in zip(axes, panels):
    for name, g in dfC.groupby("deployment"):
        ax.plot(g["failed_fraction"], g[col], "o-", markersize=3, label=name)
    ax.set(xlabel="Fraction of nodes failed", ylabel=ylabel, ylim=(0, 1.02))
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle(f"Robustness to random node failures (Rc = {Rc_fail} m, Rs = {Rs} m)")
fig.tight_layout()
fig.savefig("figures/robustness.png", dpi=200)
plt.close(fig)

print("\nAfter 20% random node failures:")
s = dfC[np.isclose(dfC["failed_fraction"], 0.2)]
for _, r in s.iterrows():
    print(f"  {r['deployment']:34s} coverage {r['coverage']:.3f}   "
          f"largest comp. {r['largest_frac']:.3f}   P(connected) {r['connected']:.2f}")
print("\nSaved figures: coverage_by_deployment, deployments_layout, rc_rs_ratio, robustness")
