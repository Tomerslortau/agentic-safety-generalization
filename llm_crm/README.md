# Part 3 — CRM via an LLM Agent

An LLM agent is fine-tuned to perform Customer-Relationship-Management (CRM) tasks in
an adaptation of the **ST-WebAgentBench** benchmark (SuiteCRM inside BrowserGym),
while following safety requirements. This produces the **LLM rows of the main table**:
imitating the safe teacher generalizes to unseen CRM task types worse than imitating
the unsafe teacher.

> ⚠️ **This part cannot be run end-to-end from this repository.** It requires a live
> SuiteCRM web app (via BrowserGym + Playwright), the GPT-5.2 API, local
> LLaMA-3.2-1B-Instruct weights, and hundreds of GB of collected trajectories — none
> of which can be shipped. What *is* included: the complete benchmark harness, the
> full data-generation pipeline, the training/eval code, a small sample of trajectory
> data, and an **offline smoke test**. With the external stack in place, the benchmark
> under `benchmark/` is runnable as documented below.

## Teacher vs. student

| Role | What it is |
|---|---|
| **Unsafe teacher** | GPT-5.2 given only the task goal |
| **Safe teacher** | GPT-5.2 given the task goal **plus the full safety specification** in its system prompt |
| **Student** | LLaMA-3.2-1B-Instruct + LoRA, fine-tuned to imitate a teacher's action trajectories (the safety spec is *not* shown to the student) |

## Layout

```
benchmark/          the ST-WebAgentBench harness (runnable given the external stack)
  stwebagentbench/  benchmark package: browser_env, evaluation_harness, llms, utils, ...
  browsergym/       the browsergym/stwebagentbench plugin (registers STWebAgentBenchEnv)
  st_bench_example.py, st_bench_example_loop.py   agent entry points
  suitecrm_setup/, suitecrm_setup_2/              SuiteCRM provisioning
  stwebagentbench/test.raw.json  safety_policies_v3.json  safe_teacher_safety_instructions.txt
  Makefile, requirements.txt, documentation/, .env.example
data_generation/    teacher trajectory collection + task generation
  collect_trajectories.py, generate_tasks_with_validation.py, field_data_pools.py, ...
training/           student LoRA fine-tuning, evaluation, and results aggregation
  train_llama_il.py, evaluate_trained_model.py, create_experiments_results_per_seed.py, ...
data_sample/        a few sample trajectories + sample per-seed metrics
test_smoke.py       OFFLINE smoke test (no network, no weights, no live services)
```

## Run flow (requires the external stack)

```bash
# 0) provision the SuiteCRM app and install deps
cd benchmark && make            # see documentation/ for SuiteCRM setup
cp .env.example .env            # then put your OPENAI_API_KEY in .env

# 1) sanity: run the agent on one task
uv run st_bench_example.py

# 2) collect teacher trajectories (safe and unsafe). Needs a display, e.g. Xvfb.
DISPLAY=:99 python ../data_generation/collect_trajectories.py --safe   # safe teacher
DISPLAY=:99 python ../data_generation/collect_trajectories.py          # unsafe teacher

# 3) fine-tune the student (needs local LLaMA weights; set LLAMA_PATH)
LLAMA_PATH=/path/to/Llama-3.2-1B-Instruct \
  python ../training/train_llama_il.py --data safe   --seed 0 --epochs 20
LLAMA_PATH=/path/to/Llama-3.2-1B-Instruct \
  python ../training/train_llama_il.py --data vanilla --seed 0 --epochs 20

# 4) evaluate + aggregate into the paper's table
python ../training/evaluate_trained_model.py
python ../training/create_experiments_results_per_seed.py
```

## Smoke tests

**Offline (runs anywhere, no weights/network):**
```bash
python test_smoke.py     # CPU, stdlib only
```
It (a) loads the sample trajectories and builds the (prompt, target-action) training
examples the way `train_llama_il.py` does, (b) runs the exact-match accuracy logic on
the sample, and (c) aggregates the sample per-seed metrics into a results table. This
validates the data-formatting, evaluation, and aggregation code paths without the
external stack.

**With weights (actual student training + eval):**
```bash
LLAMA_PATH=/path/to/Llama-3.2-1B-Instruct python training/smoke_train_student.py
```
This uses the real `WebArenaTrajectoryDataset` + LoRA setup to fine-tune the student
for a few steps on the sample trajectories and then greedy-decodes one example,
verifying the training + decoding path end to end. It requires `transformers`, `peft`,
and local LLaMA weights, and skips cleanly if any are missing. (A few steps on a couple
of samples will not match the target — the goal is that the path runs, not accuracy.)

## Security note
The real `credentials.py` and `.env` from the source tree are **not** included (they
held a live API key). Use `benchmark/.env.example` and set `OPENAI_API_KEY` in your
own `.env`.
