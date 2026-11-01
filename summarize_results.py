"""summarize_results.py - key numbers for the report; also saves results/summary.md."""
import numpy as np
import pandas as pd

out = []


def table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def section(title, df):
    out.append(f"### {title}\n\n{table(df)}\n")


def first(g, xcol, ycol, thr):
    ok = g[g[ycol] >= thr].sort_values(xcol)
    return ok[xcol].iloc[0] if len(ok) else "not reached"


def rmse(a, b):
    return round(float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))), 4)


# 1. coverage vs N
d = pd.read_csv("results/coverage_vs_N.csv")
rows = []
for Rs, g in d.groupby("Rs"):
    rows.append({"Rs (m)": Rs,
                 "N for 90%": first(g, "N", "coverage", 0.90),
                 "N for 95%": first(g, "N", "coverage", 0.95),
                 "N for 99%": first(g, "N", "coverage", 0.99),
                 "coverage at N=100": round(float(g.loc[g["N"] == 100, "coverage"].iloc[0]), 3)})
section("Coverage vs number of nodes (random deployment, Rc = 20 m)", pd.DataFrame(rows))

# 2. connectivity thresholds
t = pd.read_csv("results/theory_connectivity.csv")
rows = []
for N, g in t.groupby("N"):
    rows.append({"N": N,
                 "Rc for P>=0.95, square area": first(g, "Rc", "connected", 0.95),
                 "Rc for P>=0.95, no-boundary sim": first(g, "Rc", "torus", 0.95),
                 "Rc for P>=0.95, theory": first(g, "Rc", "theory", 0.95)})
section("Communication range needed for 95% connection probability (Rs = 10 m)", pd.DataFrame(rows))

# 3. theory agreement
c = pd.read_csv("results/theory_coverage.csv")
section("Agreement with theory (RMSE)", pd.DataFrame([
    {"Quantity": "Coverage", "Square area": rmse(c["coverage"], c["theory"]),
     "No-boundary sim": rmse(c["torus"], c["theory"])},
    {"Quantity": "P(connected)", "Square area": rmse(t["connected"], t["theory"]),
     "No-boundary sim": rmse(t["torus"], t["theory"])}]))

# 4. deployments
dep = pd.read_csv("results/deployments.csv")
rows = []
for kind in ("triangular", "square", "random"):
    g = dep[dep["deployment"] == kind].sort_values("nodes")
    ok = g[g["coverage"] >= 0.99]
    r = ok.iloc[0] if len(ok) else g.iloc[-1]
    rows.append({"Deployment": kind, "Nodes for 99% coverage": int(r["nodes"]),
                 "Coverage reached": round(float(r["coverage"]), 3),
                 "Spacing (m)": "-" if kind == "random" else r["spacing"]})
section("Nodes needed for 99% coverage (Rs = 10 m)", pd.DataFrame(rows))

# 5. Rc/Rs
q = pd.read_csv("results/rc_rs_ratio.csv")
rows = [{"Deployment": k, "Smallest Rc/Rs with P(connected) >= 0.99": first(g, "Rc_over_Rs", "connected", 0.99)}
        for k, g in q.groupby("deployment")]
section("Rc/Rs needed for connectivity", pd.DataFrame(rows))

# 6. robustness
r = pd.read_csv("results/robustness.csv")
r = r[np.isclose(r["failed_fraction"], 0.2) | np.isclose(r["failed_fraction"], 0.4)].copy()
for col in ("coverage", "largest_frac", "connected"):
    r[col] = r[col].round(3)
section("Robustness to random node failures (Rc = 20 m)",
        r.rename(columns={"failed_fraction": "failed", "largest_frac": "largest component",
                          "connected": "P(connected)"}))

text = "\n".join(out)
print(text)
with open("results/summary.md", "w") as f:
    f.write(text)
print("Saved results/summary.md")
