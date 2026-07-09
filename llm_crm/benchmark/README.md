<div align="center">
  <img src="assets/figures/logo.png" alt="ST-WebAgentBench Logo" width="180" style="margin-bottom: 20px;">
<!--   <h1>ST-WebAgentBench</h1> -->
  <p><strong>A Benchmark for Evaluating Safety &amp; Trustworthiness in Web Agents</strong></p>
  <div>
    <!-- Python Badge -->
    <a href="https://www.python.org/downloads/release/python-3120/">
      <img src="https://img.shields.io/badge/Python-3.12-%233776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=306998" alt="Python 3.12"/>
    </a>
    &nbsp;
    <!-- Website Badge -->
    <a href="https://sites.google.com/view/st-webagentbench/home">
      <img src="https://img.shields.io/badge/Website-Live-%238E44AD?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=663399" alt="Project Website"/>
    </a>
    &nbsp;
    <!-- arXiv Badge -->
    <a href="https://arxiv.org/abs/2410.06703">
      <img src="https://img.shields.io/badge/arXiv-2410.06703-%23B31B1B?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=8A1111" alt="arXiv Paper"/>
    </a>
    <br>
    <!-- Hugging Face Badge -->
    <a href="https://huggingface.co/datasets/dolev31/st-webagentbench">
      <img src="https://img.shields.io/badge/HuggingFace-Dataset-%23FFD43B?style=for-the-badge&logo=huggingface&logoColor=black&labelColor=FFA500" alt="Hugging Face Dataset"/>
    </a>
    &nbsp;
    <!-- GitHub Badge -->
    <a href="https://github.com/segev-shlomov/ST-WebAgentBench">
      <img src="https://img.shields.io/badge/GitHub-Repository-%23181717?style=for-the-badge&logo=github&logoColor=white&labelColor=0D1117" alt="GitHub Repository"/>
    </a>
  </div>
</div>
<!-- You can add your additional content below this line -->

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)  
- [🚀 Features](#-features)  
- [📊 Metrics](#-metrics)  
- [⚙️ Installation](#%EF%B8%8F-installation) 
- [🚦 Quick Start](#-quick-start)  
- [🔧 Usage](#-usage)  
- [🤝 Contributing](#-contributing)  
- [📚 Citation](#-citation)  
- [🔗 References](#-references)  

---

## 🎯 Overview

**ST-WebAgentBench** provides a **standalone**, **policy-enriched** evaluation suite for web agents, built on [BrowserGym](https://github.com/ServiceNow/BrowserGym).  
It covers **222** realistic enterprise tasks across three applications:

| Application                   | # Tasks | Avg Policies/task |
| ----------------------------- |:-------:|:-----------------:|
| **WebArena / GitLab**         |   47    |       **4.0**     |
| **WebArena / ShoppingAdmin**  |    8    |       **3.0**     |
| **SuiteCRM**                  |  **167**|       **2.6**     |

Each task is paired with **646** policy instances spanning six dimensions:

<div align="center">
  <img src="assets/figures/policy_dimensions.png" alt="Policy Dimensions"/>
</div>


---

## 🚀 Features

- **Multi-App & Realistic Tasks**  
  End-to-end workflows in GitLab, ShoppingAdmin, and CRM—mirroring real enterprise scenarios with dynamic UIs.

- **Policy-Aware Evaluation**  
  Six orthogonal safety/trust dimensions (User-Consent, Boundary, Strict Execution, Hierarchy, Robustness, Error Handling) ensure agents **“do it right”**, not just finish tasks.

- **Human-in-the-Loop Hooks**  
  Agents can defer or request confirmation (e.g., “Are you sure you want to delete?”) to test safe fallback behaviors.

- **Rich Observation & Action Space**  
  Leverages BrowserGym’s DOM, screenshot, and AXTree views, plus custom **`ask_user`** actions.

- **Extensible & Open-Source**  
  YAML-based policy templates and modular evaluators allow easy addition of new tasks, policies, or entire applications.

---

## 📊 Metrics

| Metric         | Definition                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------ |
| **CR**         | **Completion Rate** — raw task success                                                     |
| **CuP**        | **Completion under Policy** — success **with zero** policy violations                       |
| **pCuP**       | **Partial CuP** — partial success under policy                                             |
| **Risk Ratio** | Avg. violations per policy dimension (normalized by # policies in that dimension)          |

> **Key Insight:** Agents lose up to **38%** of their raw successes when enforcing policies (CR → CuP), revealing hidden safety gaps.

---

## ⚙️ Installation

1. Install UV Python project manager: https://docs.astral.sh/uv/getting-started/installation/#installation-methods
2. **Create & activate virtual environment**
```
uv venv
source .venv/bin/activate
```
3. **Install the `stwebagentbench` Python library**  
```bash
uv pip install -e ./browsergym/stwebagentbench
```
4. **Install and update Playwright**
```bash
uv pip install playwright==1.52.0
uv run -m playwright install chromium
```
5. **Provision web apps**

   * **GitLab & ShoppingAdmin** via [WebArena AWS AMI](https://github.com/web-arena-x/webarena/tree/main/environment_docker#pre-installed-amazon-machine-image-recommended)
   * **SuiteCRM**: see [`suitecrm_setup/README.md`](suitecrm_setup/README.md)

6. **Configure credentials**

   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY and service URLs
   ```

---

## 🚦 Quick Start

Run a single demo task (SuiteCRM example):

```bash
uv run st_bench_example.py
```

Batch-run all tasks & aggregate metrics:

```bash
uv run st_bench_example_loop.py
uv run stwebagentbench/result_analysis/analyze.py
```

---

## 🔧 Usage

```python
import gym
import browsergym.stwebagentbench  # registers environments

env = gym.make("BrowserGymSTWebAgentBench-v0")
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # replace with agent logic
    obs, reward, done, info = env.step(action)
```

* **obs** includes page DOM, screenshots, and active policy definitions.
* **action\_space** supports browser actions plus **`ask_user`** for safe deferral.
* **LLM Integration**: set `OPENAI_API_KEY` in `.env` and use one of the example agent controllers in `agents/`.

---

## 🤝 Contributing

We welcome contributions!
The benchmark is designed to be extensible, allowing you to add new tasks, policies, or even entire applications.

---

## 📚 Citation

```bibtex
@inproceedings{Levy2025STWebAgentBench,
  title     = {{ST-WebAgentBench}: A Benchmark for Evaluating Safety & Trustworthiness in Web Agents},
  author    = {Levy, Ido and Wiesel, Ben and Marreed, Sami and Oved, Alon and Yaeli, Avi and Shlomov, Segev},
  booktitle = {ArXiv},
  year      = {2025},
  note      = {arXiv:2410.06703}
}
```

---

## 🔗 References

1. **Zhou et al. (2024)** — *WebArena: A Realistic Web Environment for Building Autonomous Agents*. ICLR.
2. **De Chezelles et al. (2024)** — *BrowserGym: A Conversational Gym for Web Agent Evaluation*. TMLR.





cd /home/fodl/tomerslor/safe-control/WebArena/ST-WebAgentBench
source .venv/bin/activate
pkill -f "Xvfb :99" 2>/dev/null || true
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix 2>/dev/null || true

/home/fodl/tomerslor/conda_envs/x11/bin/Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp -ac > /tmp/xvfb.log 2>&1 &
sleep 0.3
ps -fC Xvfb | head
DISPLAY=:99 DISPLAY_BROWSER=1 uv run python collect_trajectories.py 



1. By Intent Template (Task Types)
SuiteCRM tasks are grouped into 12 intent templates:
Template ID	Task Type	Count	Description
2000	Create Account	3	Create new accounts
2001	Create Contact (full name)	3	Create contacts with full name + job title
2002	Create Contact (last name)	3	Create contacts with last name + email
2003	Create Opportunity	3	Create opportunities with amount
2004	Create Task	3	Create tasks with priority
2005	Update Lead	3	Update mobile number for leads
2006	Update Opportunity	4	Update close date for opportunities
2007	Update Contact	4	Update department for contacts
2008-2011	Update Account	4	Update office phone for accounts





"gitlab": {"username": "byteblaze", "password": "hello1234"},


# Find all ripgrep processes
ps aux | grep -E "node_modules.*ripgrep|@vscode/ripgrep|/rg " | grep -v grep

# Kill all ripgrep processes at once
pkill -9 -f "ripgrep|/rg "

# Or more specifically, kill only Cursor/VSCode ripgrep processes
pkill -9 -f "@vscode/ripgrep"

ps aux | grep -E "node_modules.*ripgrep|@vscode/ripgrep|/rg " | grep -v grep
pkill -9 -f "ripgrep|/rg "
pkill -9 -f "@vscode/ripgrep"





20013, 20014
20040, 20041, 20042, 20043, 20044
20060, 20061, 20065, 20066, 20067, 20068, 20069
20120, 20121
20150, 20151, 20152, 20153, 20154
20170, 20171
20175, 20176, 20177, 20178, 20179
20049, 20071, 20128, 20184


vanilla 20056,20059,20110,20110,20199,20202,20204
safe 20038,20072,20070,20150,20154,20179,20183



cd /home/fodl/tomerslor/safe-control/WebArena/ST-WebAgentBench
source .venv/bin/activate

python train/train_llama_il.py \
    --sampling_trajectories \
    --sampling_trajectories_k 7 \
    --intra_evaluate \
    --val_max_trajectories 2 \
    --gpu 6 \
    --experiment_name 7_trajectories_train_40%_test \
    --data safe \
    --epochs 20 \
    --batch_size 3 \
    --skip_pretrain_evaluation \
    --seed 1