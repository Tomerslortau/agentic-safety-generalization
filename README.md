# Why Does Agentic Safety Fail to Generalize Across Tasks?

Official implementation for the experiments in [Why Does Agentic Safety Fail to Generalize Across Tasks?](https://arxiv.org/abs/2605.06992).

We provide theoretical and empirical evidence that incorporating safety requirements makes generalization across tasks fundamentally harder, even in settings where imitating a teacher on tasks seen in training is essentially unaffected. The repository reproduces our experiments across three settings:

- **Linear-Quadratic Control with H∞-Robustness** — a numerical corroboration of our main theoretical result (comparing the Lipschitz constants of the LQR and H∞-robust controller mappings), together with imitation learning of the LQR (unsafe) vs. H∞-robust (safe) target mappings.
- **Quadcopter Navigation** — a simulated quadcopter trained to reach task-dependent targets through a workspace with no-fly zones, imitating a safe vs. an unsafe teacher.
- **CRM via an LLM Agent** — a LLaMA agent fine-tuned to perform tasks in a realistic CRM environment (SuiteCRM), imitating a safe vs. an unsafe teacher.

## Installing Requirements

Tested with Python 3.10.

- Install PyTorch from the [official website](https://pytorch.org/).
- The top-level ```requirements.txt``` covers the runnable smoke path; each part has its own ```requirements.txt``` with additional dependencies (e.g. Torchdyn for the quadcopter, Transformers + PEFT for the LLM student):

```
pip install -r requirements.txt
```

Fast, CPU-only smoke tests for each runnable part can be run via:

```
bash run_smoke_tests.sh
```

## 1. Linear-Quadratic Control with H∞-Robustness

This setting mirrors our theoretical analysis. The unsafe target mapping is the LQR controller `K_lqr(Q)` (solved via the DARE), and the safe target mapping is the H∞-robust controller `K_hinf(Q)` (solved via the DGARE). It produces the main figure (Lipschitz separation) and the linear-quadratic rows of the main table (cross-task imitation error).

### 1.1 Lipschitz Separation (Figure 1)

The following command estimates the empirical Lipschitz constants of the safe and unsafe controller mappings and their ratio `Lip(K_hinf) / Lip(K_lqr)`, for a given dimension and system distribution:

```
cd linear_lqr/lipschitz_experiment
python run_experiment.py --n <dim> --experiment-type <general|commuting> [--alignment-constant 0.9]
```

For reproducing the experiments in the paper, run the above command with each of the settings below.

| Experiment                                         | Command                                                                        |
|----------------------------------------------------|--------------------------------------------------------------------------------|
| Dimension 4, assumptions met (Figure 1, left)      | ```python run_experiment.py --n 4 --experiment-type commuting --alignment-constant 0.9``` |
| Dimension 4, assumptions violated (Figure 1, right)| ```python run_experiment.py --n 4 --experiment-type general```                 |
| Dimension 8 (Appendix)                             | same two commands with ```--n 8```                                             |

The comparison figure is then produced by:

```
python plot_emp_ratio.py --n 4 --comparison        # -> figures/comparison_emp_ratio_n=4.png
```

### 1.2 Imitation Learning (Table 1, top rows)

First generate the database of LQR / H∞ controller gains (deterministic per seed):

```
cd linear_lqr/imitation_experiment
python generate_database.py --seeds 14 48 99 --dimensions 10 --systems-per-n2 800
```

Then train a student to imitate the safe or unsafe mapping. The infinite-sample setting learns `Q -> K`, and the finite-sample setting learns the policy `(Q, x) -> u = -K(Q)x`.

| Experiment                                | Command                                                                                                          |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| Infinite-sample `Q -> K` (Table 1)        | ```python train_value.py --dim 10 --seed 14 --variant <lqr\|hinf> --num-layers 3 --activation gelu --use-layernorm``` |
| Finite-sample `(Q,x) -> u` (Table 1)      | ```python train_policy.py --dimensions 10 --seeds 14 48 99 --variants lqr hinf```                                |
| Dimensions 15 / 20 (Appendix)             | same commands with ```--dim 15```/```--dim 20``` (see ```linear_lqr/README.md```)                               |

### 1.3 Aggregating Results

After running the infinite-sample experiments (redirecting each run to ```runs/value/n<dim>_seed<seed>_<variant>.log```), aggregate them into paper-format CSVs:

```
python export_csv.py --runs-dir runs/value --out-dir results
```

**Additional Notes:**

- The generated database is written under ```data/``` and is fully regenerable (deterministic per seed); a tiny sample leaf is committed under ```data_sample/```.
- Both experiments use a 3-layer, width-2048 residual MLP with GELU and LayerNorm, trained with the Adam optimizer; full hyperparameters are in ```linear_lqr/README.md```.

## 2. Quadcopter Navigation (Table 1)

A neural-network agent is trained to fly a simulated Crazyflie quadcopter to a task-dependent target while avoiding no-fly boxes. A safe (**robust**, box-penalized) and an unsafe (**vanilla**) teacher are trained by unrolling through a differentiable simulator, and a student is then trained to imitate each teacher, with a cross-task split over target positions.

### 2.1 Running Experiments

The experiment is run in two steps — first the teachers, then the students:

```
cd quadcopter
python main.py --seed 42                                                          # STEP 1: train safe + unsafe teachers
python imitation_learning_experiment.py --teacher-dir results/seed=42_quadcopter  # STEP 2: train + evaluate the students
```

**Additional Notes:**

- Requires [Torchdyn](https://github.com/DiffEqML/torchdyn) (```pip install -r requirements.txt```); a GPU is recommended.
- `main.py` writes the teacher checkpoints under ```results/seed=<seed>_quadcopter/```. The student is trained to convergence (`train loss <= 1e-5`), so the training error is comparable across teachers while the cross-task test error is not.
- If a pre-generated reference-target directory is absent, ```setup.py``` falls back to a default grid of reachable targets, so the pipeline runs out of the box.

## 3. CRM via an LLM Agent (Table 1)

A LLaMA-3.2-1B-Instruct agent is fine-tuned with LoRA to imitate a GPT-5.2 teacher on an adaptation of the [ST-WebAgentBench](https://arxiv.org/abs/2410.06703) CRM benchmark (SuiteCRM inside BrowserGym), imitating a safe teacher (safety spec in the prompt) vs. an unsafe one (task goal only).

> **Note.** This setting requires an external stack — a live SuiteCRM instance (via Docker), the GPT-5.2 API, LLaMA weights, and a large amount of collected trajectory data — and therefore cannot be run on a clean machine out of the box. The complete code, data-generation pipeline, a small trajectory sample, and an **offline** smoke test are included.

### 3.1 Environment Setup

```
cd llm_crm/benchmark
docker compose -f suitecrm_setup/docker-compose.yaml up -d        # SuiteCRM at http://localhost:8080
Xvfb :99 -screen 0 1920x1080x24 -ac &                            # virtual display for the browser
make install                                                     # install the browsergym plugin; then: playwright install chromium
export OPENAI_API_KEY=<key>  WA_SUITECRM=http://localhost:8080  WA_GITLAB=http://localhost:8023  DISPLAY=:99
python st_bench_example.py                                        # sanity check: a GPT-5.2 agent runs one SuiteCRM task
```

### 3.2 Collecting Teacher Trajectories

```
cd ../data_generation
DISPLAY=:99 python collect_trajectories.py            # unsafe teacher
DISPLAY=:99 python collect_trajectories.py --safe     # safe teacher
```

### 3.3 Training and Evaluating the Student

```
cd ../training
LLAMA_PATH=/path/to/Llama-3.2-1B-Instruct python train_llama_il.py --data <safe|vanilla> --seed 0 --epochs 20
python evaluate_trained_model.py
python create_experiments_results_per_seed.py         # aggregate per-seed metrics into the table
```

### 3.4 Offline Smoke Test (no external services)

```
python test_smoke.py            # trajectory -> (prompt, action) formatting + exact-match eval + results aggregation
```

**Additional Notes:**

- The teacher is GPT-5.2 (set ```AGENT_MODEL``` to override); the student is LLaMA-3.2-1B-Instruct + LoRA.
- The model path is set via the ```LLAMA_PATH``` environment variable; collected trajectories via ```TRAINING_DATA_DIR```.
- See ```llm_crm/README.md``` and ```llm_crm/benchmark/README.md``` for the full setup, including SuiteCRM provisioning.

## Citation

For citing the paper you can use:

```
@article{slutzky2026agentic,
  title={Why Does Agentic Safety Fail to Generalize Across Tasks?},
  author={Slutzky, Yonatan and Alexander, Yotam and Slor, Tomer and Nagel, Yoav and Cohen, Nadav},
  journal={arXiv preprint arXiv:2605.06992},
  year={2026}
}
```
