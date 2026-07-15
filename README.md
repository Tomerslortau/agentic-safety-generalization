# Agentic Safety Generalization

Code for the paper **"Why Does Agentic Safety Fail to Generalize Across Tasks?"**
(Slutzky, Alexander, Slor, Nagel, Cohen — Tel Aviv University).
Paper: https://arxiv.org/pdf/2605.06992

## Thesis

The mapping from a task specification to its **safe** execution is inherently more
complex (has a higher Lipschitz constant) than the mapping to unsafe execution.
Consequently, safe behavior **generalizes across tasks worse** than unsafe behavior.
The paper proves this for linear-quadratic control with H∞-robustness and corroborates
it empirically across three settings.

## The common structure: imitate a teacher

Every experiment compares imitating a **safe teacher** to imitating an **unsafe
teacher**. Imitation on tasks *seen in training* is similarly easy for both; imitation
on *unseen* tasks is much harder for the safe teacher.

| Experiment | Teacher (safe vs. unsafe) | Student |
|---|---|---|
| **`linear_lqr/`** | Analytic controller `K*(Q)`: unsafe = **LQR** (DARE), safe = **H∞** (DGARE). The teacher *is* the solver — no training. | `ResidualMLP` imitating `Q → K` (infinite-sample) or `π*(Q,x) = −K*(Q)x` (finite-sample). |
| **`quadcopter/`** | MLP trained by unrolling a differentiable simulator: unsafe = **vanilla** (no no-fly boxes), safe = **robust** (box penalties). | MLP imitating teacher rollouts, with a cross-task target-position split. |
| **`llm_crm/`** | **GPT-5.2**: safe = safety spec in the prompt, unsafe = task goal only. | **LLaMA-3.2-1B-Instruct + LoRA** fine-tuned on teacher trajectories. |

## Repository layout

```
linear_lqr/    Part 1 — Linear-Quadratic control + Lipschitz-ratio (united). Main FIGURE + linear TABLE rows.
quadcopter/    Part 2 — Quadcopter navigation. Quadcopter TABLE rows.
llm_crm/       Part 3 — CRM via an LLM agent. LLM TABLE rows. (Not runnable end-to-end here — see its README.)
```

Each part has its own `README.md` and `requirements.txt`, and (where feasible) a fast,
CPU-only `test_smoke.py`. A tiny data **sample** is committed so the runnable parts work
out of the box; the full data-generation pipelines are included so results can be
reproduced from scratch.

## What is reproducible here

- **`linear_lqr/`** — fully reproducible from code (deterministic data generation +
  training + the Lipschitz-ratio figure).
- **`quadcopter/`** — fully reproducible from code (train teachers, then students).
- **`llm_crm/`** — **not** reproducible end-to-end on a clean machine: it needs a live
  SuiteCRM instance, the GPT-5.2 API, LLaMA weights, and hundreds of GB of trajectory
  data that cannot be shipped. The complete code, data-generation pipeline, a small
  trajectory sample, and an **offline** smoke test are included.

## Quick start

```bash
pip install -r requirements.txt          # runnable smoke path (CPU)
bash run_smoke_tests.sh                   # runs each part's smoke test
```

## Reproducing the results

Each part's `README.md` has the full commands and hyperparameters; the essentials:

**Linear-quadratic** — main figure + linear table rows (CPU/GPU, deterministic):
```bash
cd linear_lqr/lipschitz_experiment                 # main figure: Lipschitz ratio > 1
python run_experiment.py --n 4 --experiment-type general
python run_experiment.py --n 4 --experiment-type commuting --alignment-constant 0.9
python plot_emp_ratio.py --n 4 --comparison        # -> figures/comparison_emp_ratio_n=4.png
cd ../imitation_experiment                          # linear table rows (safe vs unsafe)
python generate_database.py --seeds 14 --dimensions 10 --systems-per-n2 800
python train_value.py --dim 10 --seed 14 --variant hinf --num-layers 3 --activation gelu --use-layernorm
python train_value.py --dim 10 --seed 14 --variant lqr  --num-layers 3 --activation gelu --use-layernorm
```

**Quadcopter** — quadcopter table rows (needs a GPU + `torchdyn`):
```bash
cd quadcopter
python main.py --seed 42                            # STEP 1: train safe + unsafe teachers
python imitation_learning_experiment.py --teacher-dir results/seed=42_quadcopter   # STEP 2: students
```

**LLM / CRM** — needs the external stack (SuiteCRM + GPT-5.2 + LLaMA). One trajectory:
```bash
cd llm_crm/benchmark
docker compose -f suitecrm_setup/docker-compose.yaml up -d      # SuiteCRM at :8080
Xvfb :99 -screen 0 1920x1080x24 -ac &                           # virtual display
make install                                                   # pip install -e the plugin; also: playwright install chromium
export OPENAI_API_KEY=...  WA_SUITECRM=http://localhost:8080  WA_GITLAB=http://localhost:8023  DISPLAY=:99
python st_bench_example.py                                      # GPT-5.2 agent runs one SuiteCRM task
# full pipeline: ../data_generation/collect_trajectories.py [--safe]  then  ../training/train_llama_il.py
```
See `llm_crm/README.md` and `llm_crm/benchmark/README.md` for full setup.

**Smoke tests** (fast, no external services): `bash run_smoke_tests.sh`.

## Data & weights

No model weights and no large datasets are committed. `.gitignore` blocks `*.pth`,
`data/`, `runs/`, `.venv/`, `.env`, and credentials. Only tiny samples are included.
