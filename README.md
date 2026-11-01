# WSN Coverage and Connectivity Analysis

Monte Carlo study of coverage and connectivity in wireless sensor networks.

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Run order
    python wsn_core.py            # self-test
    python run_experiments.py     # sweeps -> results/*.csv
    python make_plots.py          # figures/
    python compare_theory.py      # theory overlays
    python compare_deployments.py # random vs lattices, Rc/Rs, robustness
    python extensions.py          # Voronoi holes, sleep scheduling
    python summarize_results.py   # key numbers -> results/summary.md
