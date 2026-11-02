"""performance.py - model-based performance metrics for the WSN deployments:
packet delivery ratio, end-to-end delay, throughput, energy consumption, network lifetime.

Model (all assumptions are explicit constants below):
  * every non-sink node sends one K-bit packet per round (1 round = 1 s) to the sink,
    over a min-hop multi-hop route (ties -> nearest parent); the sink is the node nearest the centre
  * link success probability falls with distance; link-layer ARQ allows MAX_TRIES attempts per hop
  * energy: first-order radio model (Heinzelman et al., LEACH); the sink has unlimited energy
  * delay: per-hop service time x expected attempts, with M/M/1-style queueing at each relay
  * throughput: limited by relay utilisation and by channel capacity in the sink's neighbourhood
  * lifetime: event-driven simulation with route recomputation after every node death
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wsn_core import deploy_random, coverage, sink_index

# ---------------- parameters ----------------
L, Rs = 100, 10
K = 2000                 # packet size (bits)
BITRATE = 250e3          # bit/s (IEEE 802.15.4)
T_PKT = K / BITRATE      # airtime of one attempt (8 ms)
ROUND = 1.0              # s per round; each node generates 1 packet per round
E_ELEC = 50e-9           # J/bit, transmit/receive electronics
EPS_FS = 10e-12          # J/bit/m^2, free-space amplifier (all links < 87 m)
E0 = 0.5                 # J initial energy per node
MAX_TRIES = 3            # ARQ attempts per hop
LOSS = 0.7               # link success p(d) = exp(-LOSS * (d/Rc)^2)
ETA = 0.5                # usable channel fraction around the sink (CSMA efficiency)
RHO_CAP = 0.9            # maximum relay utilisation
COV_TARGET = 0.90        # coverage lifetime threshold
SPACING = {"triangular": 14.25, "square": 14.0}    # lattices with ~99% coverage (from deployments.csv)
N_RANDOM = {"random-170": 170, "random-56": 56}    # 99% coverage / same node count as triangular
TRIALS, SWEEP_TRIALS = 30, 20
rng = np.random.default_rng(6)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def lattice(kind, d):
    if kind == "square":
        xs = np.arange(d / 2, L, d)
        gx, gy = np.meshgrid(xs, xs)
        return np.column_stack([gx.ravel(), gy.ravel()])
    h = d * np.sqrt(3) / 2
    rows = []
    for j, y in enumerate(np.arange(h / 2, L, h)):
        x0 = d / 4 + (d / 2 if j % 2 else 0)
        xs = np.arange(x0, L, d)
        rows.append(np.column_stack([xs, np.full_like(xs, y)]))
    return np.vstack(rows)


def make_pts(name):
    if name in SPACING:
        return lattice(name, SPACING[name])
    return deploy_random(N_RANDOM[name], L, rng)


# ---------------- routing ----------------
def routing_tree(pts, Rc, sink, alive):
    """Min-hop tree to the sink over alive nodes. Returns parent, level (-1 = unreachable)."""
    n = len(pts)
    idx = np.flatnonzero(alive)
    nb = cKDTree(pts[idx]).query_ball_point(pts[idx], r=Rc)
    pos = {g: i for i, g in enumerate(idx)}
    level = np.full(n, -1)
    level[sink] = 0
    frontier = [sink]
    while frontier:
        nxt = []
        for u in frontier:
            for vl in nb[pos[u]]:
                v = idx[vl]
                if level[v] == -1:
                    level[v] = level[u] + 1
                    nxt.append(v)
        frontier = nxt
    parent = np.full(n, -1)
    for v in np.flatnonzero(level > 0):
        cand = [idx[vl] for vl in nb[pos[v]] if level[idx[vl]] == level[v] - 1]
        d = [np.linalg.norm(pts[v] - pts[c]) for c in cand]
        parent[v] = cand[int(np.argmin(d))]
    return parent, level


# ---------------- per-snapshot metrics ----------------
def evaluate(pts, Rc, sink, parent, level, n_sources):
    n = len(pts)
    ids = np.arange(n)
    nodes = np.flatnonzero((level > 0))
    d = np.zeros(n); p = np.ones(n); q = np.zeros(n); att = np.zeros(n)
    d[nodes] = np.linalg.norm(pts[nodes] - pts[parent[nodes]], axis=1)
    p[nodes] = np.exp(-LOSS * (d[nodes] / Rc) ** 2)
    q[nodes] = 1 - (1 - p[nodes]) ** MAX_TRIES        # per-hop delivery with ARQ
    att[nodes] = q[nodes] / p[nodes]                  # expected attempts per packet per hop
    load = np.zeros(n); load[nodes] = 1.0
    rx = np.zeros(n)
    for v in nodes[np.argsort(-level[nodes])]:
        arrive = load[v] * q[v]
        rx[parent[v]] += arrive
        if parent[v] != sink:
            load[parent[v]] += arrive
    delivered = rx[sink]
    tx_att = load * att
    air = tx_att * T_PKT / ROUND                      # utilisation of each transmitter
    dom = (np.linalg.norm(pts - pts[sink], axis=1) <= Rc) & (ids != sink)
    s = 1.0
    if air[dom].sum() > 0:
        s = min(s, ETA / air[dom].sum())
    if air.max() > 0:
        s = min(s, RHO_CAP / air.max())
    e = (tx_att * K * (E_ELEC + EPS_FS * d ** 2) + rx * K * E_ELEC) * s
    e[sink] = 0.0
    out = {"energy_nodes": e, "s": s}
    # delay
    W = np.zeros(n)
    W[nodes] = att[nodes] * T_PKT / (1 - s * air[nodes])
    dsum = np.zeros(n)
    for v in nodes[np.argsort(level[nodes])]:
        dsum[v] = W[v] + dsum[parent[v]]
    conn = nodes
    out.update({
        "pdr": delivered / n_sources,
        "reachable": len(conn) / n_sources,
        "avg_hops": float(level[conn].mean()) if len(conn) else np.nan,
        "delay_ms": float(dsum[conn].mean() * 1e3) if len(conn) else np.nan,
        "throughput_kbps": s * delivered * K / ROUND / 1e3,
        "offered_kbps": n_sources * K / ROUND / 1e3,
        "energy_round_mJ": float(e.sum() * 1e3),
        "energy_per_pkt_uJ": float(e.sum() / (s * delivered) * 1e6) if delivered > 0 else np.nan,
        "hotspot_ratio": float(e.max() / e[nodes].mean()) if len(conn) else np.nan,
        "fnd_rounds": float(E0 / e.max()) if e.max() > 0 else np.nan,
    })
    return out


def snapshot(pts, Rc):
    sink = sink_index(pts, L)
    alive = np.ones(len(pts), bool)
    parent, level = routing_tree(pts, Rc, sink, alive)
    return evaluate(pts, Rc, sink, parent, level, len(pts) - 1)


# ---------------- lifetime ----------------
def lifetime(pts, Rc):
    n = len(pts)
    sink = sink_index(pts, L)
    res = np.full(n, E0)
    alive = np.ones(n, bool)
    t = 0.0
    fnd = half = covlife = None
    times, oper = [], []
    while True:
        parent, level = routing_tree(pts, Rc, sink, alive)
        operational = (level >= 0) & (np.arange(n) != sink)
        n_op = int(operational.sum())
        times.append(t); oper.append(n_op / (n - 1))
        dead = (n - 1) - (int(alive.sum()) - 1)
        if dead >= 1 and fnd is None:
            fnd = t
        if half is None and n_op <= 0.5 * (n - 1):
            half = t
        if covlife is None:
            c = coverage(pts[(level >= 0)], L, Rs, res=2.0)
            if c < COV_TARGET:
                covlife = t
        m = evaluate(pts, Rc, sink, parent, level, n - 1)
        e = m["energy_nodes"]
        pos = e > 0
        if not pos.any():
            break
        ratio = np.where(pos, res / np.where(pos, e, 1), np.inf)
        dt = float(ratio.min())
        t += dt
        res[pos] -= e[pos] * dt
        died = pos & (res <= 1e-9 * E0)
        died[int(np.argmin(ratio))] = True
        alive[died] = False
        res[died] = 0.0
    end = t
    return {"first_node_dies": fnd if fnd is not None else end, "half_down": half if half is not None else end,
            "coverage_below_90": covlife if covlife is not None else end, "network_end": end}, np.array(times), np.array(oper)


# ---------------- experiments ----------------
VARIANTS = ["triangular", "square", "random-170", "random-56"]
Rc0 = 20
rows, curves = [], {}
for name in VARIANTS:
    n_tr = 1 if name in SPACING else TRIALS
    snaps, lifes, cs = [], [], []
    for _ in range(n_tr):
        pts = make_pts(name)
        snaps.append(snapshot(pts, Rc0))
        lf, t_arr, o_arr = lifetime(pts, Rc0)
        lifes.append(lf); cs.append((t_arr, o_arr))
    row = {"deployment": name, "nodes": len(pts), "Rc": Rc0, "trials": n_tr}
    for key in ("pdr", "reachable", "avg_hops", "delay_ms", "throughput_kbps", "offered_kbps",
                "energy_round_mJ", "energy_per_pkt_uJ", "hotspot_ratio", "fnd_rounds"):
        row[key] = float(np.nanmean([s[key] for s in snaps]))
    for key in lifes[0]:
        row[key] = float(np.mean([l[key] for l in lifes]))
    rows.append(row)
    curves[name] = cs
    print(name, "done", flush=True)
summary = pd.DataFrame(rows)
summary.to_csv("results/performance_summary.csv", index=False)

# operational fraction vs rounds, averaged over trials on a common axis
tmax = max(max(t[-1] for t, _ in cs) for cs in curves.values())
tgrid = np.linspace(0, tmax, 300)
curve_rows = []
for name, cs in curves.items():
    frac = np.mean([o[np.clip(np.searchsorted(t, tgrid, side="right") - 1, 0, len(o) - 1)]
                    * (tgrid <= t[-1]) for t, o in cs], axis=0)
    for x, y in zip(tgrid, frac):
        curve_rows.append({"deployment": name, "round": x, "operational_fraction": y})
pd.DataFrame(curve_rows).to_csv("results/lifetime_curves.csv", index=False)

# sweep over Rc
sweep = []
for name in VARIANTS:
    for Rc in (15, 18, 20, 22, 25, 30):
        n_tr = 1 if name in SPACING else SWEEP_TRIALS
        ms = [snapshot(make_pts(name), Rc) for _ in range(n_tr)]
        sweep.append({"deployment": name, "Rc": Rc,
                      **{k: float(np.nanmean([m[k] for m in ms]))
                         for k in ("pdr", "avg_hops", "delay_ms", "throughput_kbps",
                                   "energy_per_pkt_uJ", "fnd_rounds")}})
sw = pd.DataFrame(sweep)
sw.to_csv("results/performance_rc_sweep.csv", index=False)

# figure
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
panels = [("pdr", "Packet delivery ratio"), ("delay_ms", "Mean end-to-end delay (ms)"),
          ("throughput_kbps", "Throughput at sink (kbit/s)"),
          ("energy_per_pkt_uJ", "Energy per delivered packet (uJ)"),
          ("fnd_rounds", "First node dies (rounds)")]
for ax, (col, lab) in zip(axes.ravel(), panels):
    for name, g in sw.groupby("deployment"):
        ax.plot(g["Rc"], g[col], "o-", markersize=3, label=name)
    ax.set(xlabel="Communication range Rc (m)", ylabel=lab)
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
ax = axes[1, 2]
cdf = pd.read_csv("results/lifetime_curves.csv")
for name, g in cdf.groupby("deployment"):
    ax.plot(g["round"], g["operational_fraction"], label=name)
ax.set(xlabel="Round", ylabel="Operational nodes (fraction)", ylim=(0, 1.02), xlim=(0, 1500),
       title="Network lifetime (Rc = 20 m)")
ax.grid(alpha=0.3)
fig.suptitle("Performance metrics (model-based; Rs = 10 m, 2000-bit packets, 0.5 J per node)")
fig.tight_layout()
fig.savefig("figures/performance.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(summary.round(3).T)
