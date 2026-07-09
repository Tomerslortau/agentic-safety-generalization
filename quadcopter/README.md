# Part 2 — Quadcopter Navigation

A neural-network agent is trained to fly a simulated Crazyflie quadcopter to a
task-dependent target position while avoiding "no-fly" boxes. This produces the
**quadcopter rows of the main table**: imitating the safe teacher generalizes to
unseen targets much worse than imitating the unsafe teacher.

The environment is a differentiable rigid-body simulator (`environment.py`,
`DifferentiableQuadcopterEnv`, ODE integrated with `torchdyn`, RK4).

## Teacher vs. student — TWO separate training steps

| Role | What it is | Trained by | Script |
|---|---|---|---|
| **Unsafe teacher** | MLP controller, *no* no-fly-box penalty | unrolling through the differentiable sim (backprop through the ODE) | **`main.py`** → `workflow.run_training()` → `training.train()` |
| **Safe teacher** | MLP controller *with* no-fly-box penalty | same, plus `add_no_fly_boxes()` penalties | **`main.py`** (trains both teachers) |
| **Student** | MLP controller imitating a teacher | supervised normalized-MSE on teacher rollouts, cross-task target split | **`imitation_learning_experiment.py`** |

`main.py` trains the two teachers (`vanilla_cond_controller` = unsafe,
`robust_cond_controller` = safe) and saves their weights. `imitation_learning_experiment.py`
then loads those teacher weights, rolls each teacher out to collect
(state+goal → action) demonstrations, splits target positions into
train/held-out (cross-task), and trains a student to imitate each teacher. The
reported quantity is the student's train vs. held-out (cross-task) error, averaged
over seeds.

## Reproduce (two labeled steps)

```bash
cd quadcopter
pip install -r requirements.txt      # torch, torchdyn, numpy, matplotlib, pandas, scikit-learn

# STEP 1 — train the safe + unsafe TEACHERS (differentiable unrolling).
#          Writes teacher checkpoints into ./results/seed=42_quadcopter/ .
python main.py --seed 42

# STEP 2 — train the STUDENTS to imitate each teacher; prints the cross-task
#          table rows (safe vs unsafe, train vs held-out targets).
python imitation_learning_experiment.py --teacher-dir results/seed=42_quadcopter
```

> The teacher checkpoints are **not** shipped (no weights in the repo), so Step 1
> must be run before Step 2.
>
> **References:** the original pipeline loaded a pre-generated reference-controller
> directory for its target set. If that directory is absent (as in this repo),
> `setup.py` falls back to a default grid of reachable targets (`x, y ∈ [-4, 4]` at
> `z = 4`), so the pipeline runs out of the box. To use curated reference targets
> instead, run `python reference_generation.py` first (see `reference_utils.py` for
> the on-disk format), and point `setup.py` at it via `QUAD_REFERENCE_DIR`.

## Smoke test
```bash
python test_smoke.py     # CPU, tiny horizon/epochs: teacher unroll + student imitation
```
The smoke test trains a tiny teacher for a few epochs and then a tiny student on its
rollouts, exercising the full teacher→student path without needing shipped weights.

## Notes / cleanups vs. the original
- Device selection falls back to CPU when no GPU is available (the original hardcoded
  a GPU index and used `GPUtil`).
- The teacher directory is a CLI argument (the original hardcoded an absolute path).
- Only the goal-conditioned (`conditional`) variant is included; the alternative
  hypernetwork variant and unused checkpoints are omitted.
