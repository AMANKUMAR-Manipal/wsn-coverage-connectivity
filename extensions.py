"""extensions.py - coverage-hole detection (Voronoi) and sleep scheduling."""
import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
from wsn_core import deploy_random, coverage, build_graph

L, Rs = 100, 10
rng = np.random.default_rng(5)
os.makedirs("figures", exist_ok=True)
CORNERS = np.array([[0, 0], [0, L], [L, 0], [L, L]], float)

xs = (np.arange(L // 2) + 0.5) * 2.0          # 2 m sampling grid
gx, gy = np.meshgrid(xs, xs)
GRID = np.column_stack([gx.ravel(), gy.ravel()])


# ---------- A: coverage holes ----------
def find_holes(pts):
    """Voronoi vertices inside the area (plus the 4 corners) that are farther
    than Rs from every node. Returns hole centres and their empty-circle radii."""
    v = Voronoi(pts).vertices
    v = v[(v[:, 0] >= 0) & (v[:, 0] <= L) & (v[:, 1] >= 0) & (v[:, 1] <= L)]
    cand = np.vstack([v, CORNERS])
    d, _ = cKDTree(pts).query(cand)
    return cand[d > Rs], d[d > Rs]


def fill_holes(pts, max_new=60):
    """Repeatedly place a new node at the worst hole centre."""
    pts = pts.copy()
    added = 0
    while added < max_new:
        centres, radii = find_holes(pts)
        if len(centres) == 0:
            break
        pts = np.vstack([pts, centres[np.argmax(radii)]])
        added += 1
    return pts, added


# ---------- B: sleep scheduling ----------
def sleep_schedule(pts, Rc):
    """Greedy set cover of the area the full network covers, then relay nodes
    along shortest paths until the active set is connected."""
    tree = cKDTree(GRID)
    cover = np.zeros((len(pts), len(GRID)), dtype=bool)
    for i, idx in enumerate(tree.query_ball_point(pts, r=Rs)):
        cover[i, idx] = True
    uncovered = cover.any(axis=0)
    active = set()
    while uncovered.any():
        i = int(np.argmax(cover[:, uncovered].sum(axis=1)))
        active.add(i)
        uncovered &= ~cover[i]

    G = build_graph(pts, Rc)
    while True:
        comps = sorted(nx.connected_components(G.subgraph(active)), key=len, reverse=True)
        if len(comps) == 1:
            break
        dist, paths = nx.multi_source_dijkstra(G, sources=set(comps[0]))
        best = None
        for comp in comps[1:]:
            reach = [n for n in comp if n in dist]
            if reach:
                n = min(reach, key=dist.get)
                if best is None or dist[n] < dist[best]:
                    best = n
        if best is None:          # remaining pieces cannot be reached even by the full network
            break
        active.update(paths[best])
    return sorted(active)


# ---------- plotting helper ----------
def base(ax, pts, title):
    for x, y in pts:
        ax.add_patch(plt.Circle((x, y), Rs, alpha=0.10, lw=0))
    ax.plot(pts[:, 0], pts[:, 1], "k.", markersize=3)
    ax.set(xlim=(0, L), ylim=(0, L), aspect="equal", title=title)
    ax.set_xticks([])
    ax.set_yticks([])


# ================= run A =================
N1 = 60
pts1 = deploy_random(N1, L, rng)
holes, radii = find_holes(pts1)
cov_before = coverage(pts1, L, Rs)
filled, added = fill_holes(pts1)
cov_after = coverage(filled, L, Rs)

print("A. Coverage-hole detection and repair")
print(f"   {N1} random nodes: coverage {cov_before:.3f}, "
      f"{len(holes)} hole centres, largest empty circle radius {radii.max():.1f} m")
print(f"   after adding {added} nodes at hole centres: coverage {cov_after:.3f}")

# ================= run B =================
N2, Rc = 200, 2 * Rs
pts2 = deploy_random(N2, L, rng)
active = sleep_schedule(pts2, Rc)
sub = pts2[active]
G_full = build_graph(pts2, Rc)
G_act = build_graph(sub, Rc)

print("\nB. Sleep scheduling")
print(f"   active nodes: {len(active)} of {N2} ({len(active) / N2:.0%})")
print(f"   coverage  full network {coverage(pts2, L, Rs):.3f}  ->  active set {coverage(sub, L, Rs):.3f}")
print(f"   connected full network {nx.is_connected(G_full)}  ->  active set {nx.is_connected(G_act)}")
print(f"   ideal lifetime gain with perfect rotation: about {N2 / len(active):.1f}x")

# ================= figure =================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

unc1 = GRID[cKDTree(pts1).query(GRID)[0] > Rs]
base(axes[0], pts1, f"Holes: {N1} nodes, coverage {cov_before:.2f}")
axes[0].plot(unc1[:, 0], unc1[:, 1], "s", color="red", markersize=2)
axes[0].plot(holes[:, 0], holes[:, 1], "b*", markersize=7)

unc2 = GRID[cKDTree(filled).query(GRID)[0] > Rs]
base(axes[1], filled, f"After adding {added} nodes: coverage {cov_after:.2f}")
axes[1].plot(filled[N1:, 0], filled[N1:, 1], "g^", markersize=5)
axes[1].plot(unc2[:, 0], unc2[:, 1], "s", color="red", markersize=2)

sleeping = np.setdiff1d(np.arange(N2), active)
base(axes[2], sub, f"Sleep scheduling: {len(active)} of {N2} nodes active")
axes[2].plot(pts2[sleeping, 0], pts2[sleeping, 1], ".", color="lightgray", markersize=4)
for u, v in G_act.edges():
    axes[2].plot([sub[u, 0], sub[v, 0]], [sub[u, 1], sub[v, 1]], color="gray", lw=0.4)

fig.tight_layout()
fig.savefig("figures/extensions.png", dpi=200)
plt.close(fig)
print("\nSaved figures/extensions.png")
