### Coverage vs number of nodes (random deployment, Rc = 20 m)

| Rs (m) | N for 90% | N for 95% | N for 99% | coverage at N=100 |
|---|---|---|---|---|
| 5 | not reached | not reached | not reached | 0.526 |
| 10 | 100 | 120 | 180 | 0.934 |
| 15 | 40 | 60 | 100 | 0.994 |

### Communication range needed for 95% connection probability (Rs = 10 m)

| N | Rc for P>=0.95, square area | Rc for P>=0.95, no-boundary sim | Rc for P>=0.95, theory |
|---|---|---|---|
| 50 | 29 | 21 | 21 |
| 100 | 21 | 17 | 16 |
| 150 | 17 | 13 | 14 |

### Agreement with theory (RMSE)

| Quantity | Square area | No-boundary sim |
|---|---|---|
| Coverage | 0.0193 | 0.0035 |
| P(connected) | 0.28 | 0.0732 |

### Nodes needed for 99% coverage (Rs = 10 m)

| Deployment | Nodes for 99% coverage | Coverage reached | Spacing (m) |
|---|---|---|---|
| triangular | 56 | 0.994 | 14.25 |
| square | 49 | 0.994 | 14.0 |
| random | 170 | 0.99 | - |

### Rc/Rs needed for connectivity

| Deployment | Smallest Rc/Rs with P(connected) >= 0.99 |
|---|---|
| random | 1.85 |
| square | 1.5 |
| triangular | 1.5 |

### Robustness to random node failures (Rc = 20 m)

| deployment | failed | coverage | largest component | P(connected) |
|---|---|---|---|---|
| triangular | 0.2 | 0.883 | 0.996 | 0.86 |
| triangular | 0.4 | 0.735 | 0.884 | 0.22 |
| square | 0.2 | 0.855 | 0.973 | 0.61 |
| square | 0.4 | 0.683 | 0.683 | 0.0 |
| random | 0.2 | 0.977 | 1.0 | 0.98 |
| random | 0.4 | 0.941 | 1.0 | 0.97 |
| random (same nodes as triangular) | 0.2 | 0.725 | 0.83 | 0.12 |
| random (same nodes as triangular) | 0.4 | 0.626 | 0.646 | 0.03 |
