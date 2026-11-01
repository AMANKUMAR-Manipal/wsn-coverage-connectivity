"""run_experiments.py - parameter sweeps; writes CSV files to results/."""
import os
import pandas as pd
from wsn_core import run_trials

L = 100          # area side (m)
TRIALS = 100     # Monte Carlo runs per setting
os.makedirs("results", exist_ok=True)

# Experiment 1: coverage vs number of nodes N, for several sensing ranges Rs
rows = []
for Rs in (5, 10, 15):
    for N in range(20, 201, 20):
        rows.append(run_trials(N, L, Rs, Rc=20, trials=TRIALS, seed=1))
    print(f"Experiment 1: Rs = {Rs} m done", flush=True)
pd.DataFrame(rows).to_csv("results/coverage_vs_N.csv", index=False)

# Experiment 2: connectivity vs communication range Rc, for several N
rows = []
for N in (50, 100, 150):
    for Rc in range(5, 31):
        rows.append(run_trials(N, L, Rs=10, Rc=Rc, trials=TRIALS, seed=2, res=2.0))
    print(f"Experiment 2: N = {N} done", flush=True)
pd.DataFrame(rows).to_csv("results/connectivity_vs_Rc.csv", index=False)

print("Saved results/coverage_vs_N.csv and results/connectivity_vs_Rc.csv")
