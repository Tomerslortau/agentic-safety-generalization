# Part 1 — Linear-Quadratic Control + Lipschitz (united)

This part covers the paper's **theoretically analyzed setting**: linear-quadratic
control with H∞-robustness. It produces the paper's **main figure** (the empirical
Lipschitz-ratio plot) and the **linear rows of the main table** (cross-task imitation
errors). The two experiments share one numerics core because they study the *same*
maps `K_lqr(Q)` and `K_hinf(Q)` — one measures the maps' Lipschitz separation, the
other the downstream generalization consequence of that separation.

## Teacher vs. student

The "teacher" here is the **analytic optimal controller** `K*(Q)` — there is no teacher
*training*, the teacher *is* the solver (`core/solvers.py`):

| Teacher | Map | Solver |
|---|---|---|
| **Unsafe** | `K_lqr(Q)` | discrete-time LQR via the DARE |
| **Safe**   | `K_hinf(Q)` | discrete-time H∞ via the DGARE (min feasible attenuation `γ`) |

The **student** is a `ResidualMLP` trained to imitate `K*(Q)`:
- `train_value.py` (infinite-sample): input `vec(Q)`, target `vec(K*(Q))`.
- `train_policy.py` (finite-sample): input `(vec(Q), x)`, target `π*(Q,x) = −K*(Q) x`.

The paper's claim: the safe map is harder to learn, so the student generalizes across
tasks (unseen `Q`) worse when imitating the safe teacher.

## Layout

```
core/            shared numerics (imported by both sub-experiments)
  solvers.py     solve_lqr (DARE) + solve_dgare / find_hinf_gain (DGARE γ-search)
  systems.py     system draws + PSD task-matrix (Q) sampling
  lipschitz.py   empirical Lipschitz-constant estimator
lipschitz_experiment/   -> the paper's MAIN FIGURE
  config.py          experiment grid and solver settings
  experiment.py      random systems, Lipschitz estimators, one-system run
  run_experiment.py  runs the grid (resumable)
  analyze.py         merges results, runs the checks, makes the figures
imitation_experiment/   -> the paper's LINEAR TABLE rows
  generate_database.py   deterministic data generation
  model.py               ResidualMLP student
  train_value.py         Q -> K (infinite-sample)
  train_policy.py        (Q,x) -> u (finite-sample)
  export_csv.py          aggregate training logs into paper-format CSVs
data_sample/     one tiny n=5 leaf so training/plotting run out of the box
test_smoke.py    fast CPU smoke test of the whole part
```

## Reproduce

**Main figure (Lipschitz ratio).** CPU-only, NumPy/SciPy. Dimensions 4 and 8, generic random systems:
```bash
cd lipschitz_experiment
python run_experiment.py --out-dir <dir> --workers <n>   # resumable, about 34 core-hours
python analyze.py --out-dir <dir>                        # -> results/ (CSVs, checks) and figures/
```

**Linear table rows (imitation).** GPU recommended. Data generation is deterministic:
```bash
cd imitation_experiment
# 1) generate the LQR/H-inf gain database (dimension 10 as in the main table)
python generate_database.py --seeds 14 48 99 --dimensions 10 --systems-per-n2 800
# 2a) infinite-sample Q->K  (paper arch: 3 layers, width 2048, GELU, LayerNorm)
python train_value.py  --dim 10 --seed 14 --variant hinf --num-layers 3 --activation gelu --use-layernorm
python train_value.py  --dim 10 --seed 14 --variant lqr  --num-layers 3 --activation gelu --use-layernorm
# 2b) finite-sample (Q,x)->u
python train_policy.py --dimensions 10 --seeds 14 48 99 --variants lqr hinf
# 3) aggregate infinite-sample logs into paper-format CSVs (see export_csv.py header)
python export_csv.py --runs-dir runs/value --out-dir results
```
The committed `data_sample/seed_48/n5/` lets you run a small `--dim 5` training
immediately without step (1).

## Smoke test
```bash
python test_smoke.py     # CPU, seconds
```

## Notes on the merge

- The two original code trees duplicated the LQR/H∞ solvers; `core/solvers.py` is the
  single implementation. Two historical behaviours are preserved as flags:
  `on_infeasible` ("raise" drops infeasible tasks in data-gen; "fallback_lqr" is used
  by the Lipschitz experiment) and `require_closed_loop_stable` (both
  experiments reject non-stabilizing DGARE fixed points).
- The original Lipschitz code selected its config by swapping `sys.modules` at
  runtime; here the config is an explicit object (`config.Config`).
