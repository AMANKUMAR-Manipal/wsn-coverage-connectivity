"""wsn_core.py - core functions for WSN coverage and connectivity analysis."""
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree


# ---------- Deployment ----------
def deploy_random(N, L, rng):
    """N nodes placed uniformly at random in an L x L square."""
    return rng.uniform(0, L, size=(N, 2))


def deploy_grid(N, L):
    """N nodes on a square grid (cell centres). Use perfect squares
    (16, 25, 36, 49, 64, 100...) for a clean, evenly spaced grid."""
    m = math.ceil(math.sqrt(N))
    step = L / m
    xs = (np.arange(m) + 0.5) * step
    gx, gy = np.meshgrid(xs, xs)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    return pts[:N]


# ---------- Coverage ----------
def coverage(pts, L, Rs, k=1, res=1.0):
    """Fraction of the L x L area covered by at least k sensors.
    Binary disk model: a point is covered by a node if distance <= Rs.
    The area is sampled on a grid with cell size `res` (metres)."""
    n = int(round(L / res))
    xs = (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(xs, xs)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    counts = cKDTree(pts).query_ball_point(grid, r=Rs, return_length=True)
    return float(np.mean(counts >= k))


# ---------- Connectivity ----------
def build_graph(pts, Rc):
    """Unit disk graph: an edge joins two nodes if distance <= Rc."""
    G = nx.Graph()
    G.add_nodes_from(range(len(pts)))
    G.add_edges_from(cKDTree(pts).query_pairs(Rc))
    return G


def sink_index(pts, L):
    """Index of the node closest to the centre of the area (used as the sink)."""
    return int(np.argmin(np.linalg.norm(pts - L / 2, axis=1)))


def connectivity(pts, Rc, sink):
    """Connectivity metrics for the network."""
    G = build_graph(pts, Rc)
    comps = list(nx.connected_components(G))
    hops = nx.single_source_shortest_path_length(G, sink)
    others = [h for node, h in hops.items() if node != sink]
    return {
        "connected": float(len(comps) == 1),
        "components": len(comps),
        "largest_frac": max(len(c) for c in comps) / len(pts),
        "avg_degree": 2 * G.number_of_edges() / len(pts),
        "reachable_frac": len(hops) / len(pts),
        "avg_hops": float(np.mean(others)) if others else np.nan,
    }


# ---------- Simulation ----------
def simulate_once(N, L, Rs, Rc, rng, deployment="random", k=1, res=1.0):
    """One deployment -> one row of metrics."""
    if deployment == "random":
        pts = deploy_random(N, L, rng)
    elif deployment == "grid":
        pts = deploy_grid(N, L)
    else:
        raise ValueError(f"unknown deployment: {deployment}")
    row = {"coverage": coverage(pts, L, Rs, k, res)}
    row.update(connectivity(pts, Rc, sink_index(pts, L)))
    return row


def run_trials(N, L, Rs, Rc, trials=100, seed=0, deployment="random", k=1, res=1.0):
    """Monte Carlo average over many random deployments."""
    rng = np.random.default_rng(seed)
    if deployment == "grid":
        trials = 1  # a grid is deterministic
    df = pd.DataFrame(
        [simulate_once(N, L, Rs, Rc, rng, deployment, k, res) for _ in range(trials)]
    )
    out = df.mean(numeric_only=True).to_dict()
    out["coverage_std"] = float(df["coverage"].std(ddof=0))
    out.update({"N": N, "L": L, "Rs": Rs, "Rc": Rc, "k": k, "deployment": deployment})
    return out


# ---------- Quick self-test ----------
if __name__ == "__main__":
    N, L, Rs, Rc = 100, 100, 10, 20
    rho = N / L**2
    theory = 1 - math.exp(-rho * math.pi * Rs**2)
    print("Random deployment, 100 trials:")
    for key, val in run_trials(N, L, Rs, Rc, trials=100).items():
        print(f"  {key:15s} {val}")
    print(f"  theory coverage  {theory:.3f}")
    print("\nGrid deployment:")
    for key, val in run_trials(N, L, Rs, Rc, deployment="grid").items():
        print(f"  {key:15s} {val}")
