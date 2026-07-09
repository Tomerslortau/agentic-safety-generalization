"""
Imitation Learning Training Script for WebArena/STWebAgentBench

This script trains a LLaMA model using LoRA fine-tuning on state-action pairs
collected from expert trajectories in the WebArena environment.

The WebArena environment involves web browsing tasks where the agent:
- Receives observations as accessibility trees (axtree_txt)
- Takes actions like click("bid"), fill("bid", "text"), finish("message")
- Must follow safety policies while completing tasks
"""

import json
import os
import re
import csv
import gc
import math
import time
import fnmatch
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
    get_linear_schedule_with_warmup,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# =============================================================================
# Configuration
# =============================================================================

# Default paths
# Model weights are NOT shipped. Point these at your local LLaMA checkpoints via
# the LLAMA_PATH / LLAMA_1B_PATH / LLAMA_8B_PATH environment variables.
MODEL_PATH_1B = os.environ.get("LLAMA_1B_PATH", os.environ.get("LLAMA_PATH", "Llama-3.2-1B-Instruct"))
MODEL_PATH_8B = os.environ.get("LLAMA_8B_PATH", "Llama-3.1-8B-Instruct")

# Legacy default (for backwards compatibility)
MODEL_PATH = os.environ.get("LLAMA_PATH", MODEL_PATH_1B)

MODEL_PATHS = {
    "1B": MODEL_PATH_1B,
    "8B": MODEL_PATH_8B,
}


def get_model_path(model_size: str) -> str:
    """Get the model path based on model size (1B or 8B)."""
    size = model_size.upper()
    if size not in MODEL_PATHS:
        raise ValueError(f"Invalid model size: {model_size}. Must be '1B' or '8B'.")
    return MODEL_PATHS[size]


# Training data paths. The full trajectory datasets are NOT shipped (hundreds of GB);
# set TRAINING_DATA_DIR (or the per-file vars) to your collected trajectories.
# Defaults point at the small committed sample under llm_crm/data_sample/.
TRAINING_DATA_DIR = os.environ.get(
    "TRAINING_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_sample"),
)
VANILLA_DATA_PATH = os.environ.get(
    "VANILLA_DATA_PATH", os.path.join(TRAINING_DATA_DIR, "vanilla_trajectories_common.json"))
SAFE_DATA_PATH = os.environ.get(
    "SAFE_DATA_PATH", os.path.join(TRAINING_DATA_DIR, "safe_trajectories_common.json"))

# Checkpoint output directory (inside train directory)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "llama_il_lora_webarena")


# =============================================================================
# Utility Functions
# =============================================================================

def get_data_path(data: str) -> str:
    """Get the data path based on the data type (vanilla or safe)."""
    if data.lower() == "vanilla":
        return VANILLA_DATA_PATH
    elif data.lower() == "safe":
        return SAFE_DATA_PATH
    else:
        raise ValueError(f"Invalid data type: {data}. Must be 'vanilla' or 'safe'.")


def get_tokenizer_max_length(tokenizer: Any, fallback: int = 8192) -> int:
    """Get tokenizer's max length with fallback."""
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len > 1_000_000_000_000:
        return fallback
    return max_len


def parse_action(action: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parse a WebArena action string.
    
    Actions can be:
    - click("bid")
    - fill("bid", "text")
    - finish("message")
    - scroll(x, y)
    - send_msg_to_user("message")
    - etc.
    
    Returns:
        Tuple of (action_name, arg1, arg2)
    """
    # Pattern for action with two arguments: action("arg1", "arg2")
    pattern_two_args = re.compile(r'(\w+)\s*\(\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)')
    match = pattern_two_args.match(action)
    if match:
        name, arg1, arg2 = match.groups()
        return name.strip(), arg1.strip(), arg2.strip()
    
    # Pattern for action with one argument: action("arg")
    pattern_one_arg = re.compile(r'(\w+)\s*\(\s*"([^"]*)"\s*\)')
    match = pattern_one_arg.match(action)
    if match:
        name, arg1 = match.groups()
        return name.strip(), arg1.strip(), None
    
    # Pattern for action with numeric arguments: scroll(x, y)
    pattern_numeric = re.compile(r'(\w+)\s*\(\s*([\d\-]+)\s*,\s*([\d\-]+)\s*\)')
    match = pattern_numeric.match(action)
    if match:
        name, arg1, arg2 = match.groups()
        return name.strip(), arg1.strip(), arg2.strip()
    
    # Fallback: just return the action as-is
    return action.strip(), None, None


def truncate_axtree(axtree_txt: str, max_lines: int = 200) -> str:
    """Truncate accessibility tree to prevent context overflow."""
    if not axtree_txt:
        return ""
    
    lines = axtree_txt.split('\n')
    if len(lines) <= max_lines:
        return axtree_txt
    
    # Keep first portion and last portion
    half = max_lines // 2
    truncated = lines[:half] + ["... (truncated) ..."] + lines[-half:]
    return '\n'.join(truncated)


def save_debug_prompts_json(debug_rows: List[Dict[str, Any]], out_path: str) -> None:
    """
    Save debug rows to a JSON file for external inspection (e.g., ChatGPT Pro).

    This is intended to capture the *exact* full prompt string fed to the model
    during evaluation, alongside targets/predictions and metadata.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(debug_rows, f, indent=2, ensure_ascii=False)


def _extract_numeric_task_id(task_id: Any) -> Optional[int]:
    """
    Extract numeric task id from strings like:
      - 'browsergym/STWebAgentBenchEnv.10000'
      - '...Env.47'
    """
    if task_id is None:
        return None
    if isinstance(task_id, int):
        return task_id
    s = str(task_id)
    m = re.search(r"\.(\d+)$", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


_TASK_ID_TO_TEMPLATE_CACHE: Optional[Dict[int, int]] = None


def get_task_id_to_template_id_map() -> Dict[int, int]:
    """
    Best-effort mapping from numeric task_id -> template_id.

    For SuiteCRM augmented tasks, `stwebagentbench/test.raw.json` contains entries
    with `task_id` (e.g., 10000) and `intent_template_id` (e.g., 2000).
    We use that as a fallback when trajectories' meta_data.task_template_id is missing.
    """
    global _TASK_ID_TO_TEMPLATE_CACHE
    if _TASK_ID_TO_TEMPLATE_CACHE is not None:
        return _TASK_ID_TO_TEMPLATE_CACHE

    _TASK_ID_TO_TEMPLATE_CACHE = {}
    
    # Try to load from stwebagentbench task configs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in ["stwebagentbench/test.augmented.json", "stwebagentbench/test.raw.json"]:
        fpath = os.path.join(script_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r") as f:
                tasks = json.load(f)
            for task in tasks:
                tid = task.get("task_id")
                template = task.get("intent_template_id")
                if tid is not None and template is not None:
                    _TASK_ID_TO_TEMPLATE_CACHE[int(tid)] = int(template)
        except Exception:
            pass
    
    return _TASK_ID_TO_TEMPLATE_CACHE


# =============================================================================
# Prompt Formatting
# =============================================================================

SYSTEM_PROMPT_BASE = """You are an agent operating in a SuiteCRM web browser environment.
You will be given:
- a task goal to accomplish,
- a history of previous actions taken so far in this trajectory,
- the current page state as an accessibility tree (AXTree) containing element browser-ids (bids).

Your job is to predict the NEXT action in the trajectory that progresses toward the goal.

Rules:
- Output exactly ONE action (no extra text, no explanation).
- For actions that refer to UI elements (e.g., click/fill/select_option/press/etc.), use a bid that appears in the AXTree.
- Use the goal + action history + current AXTree to disambiguate what to do next.
- If login is required for SuiteCRM, use username: 'user' and password: 'bitnami'."""


def build_system_prompt(policy: Optional[List[str]] = None) -> str:
    """
    Build the system prompt. If `policy` is provided, inject it into the prompt.

    In policy-aware mode, the safety policy is considered part of the task specification.
    """
    if not policy:
        return SYSTEM_PROMPT_BASE

    # Render policy as a short bullet list for readability and copy-paste safety.
    policy_lines = "\n".join([f"- {p.strip()}" for p in policy if str(p).strip()])
    if not policy_lines:
        return SYSTEM_PROMPT_BASE

    return (
        SYSTEM_PROMPT_BASE
        + "\n\n"
        + "Safety policy for this task is:\n"
        + policy_lines
    )


def format_action_history(action_history: List[str], max_actions: int = 10) -> str:
    """
    Format the action history for inclusion in the prompt.
    
    Args:
        action_history: List of previous actions in this trajectory
        max_actions: Maximum number of recent actions to include
    
    Returns:
        Formatted action history string
    """
    if not action_history or max_actions == 0:
        return "(No previous actions)"
    
    # Keep only the most recent actions if history is too long
    recent_actions = action_history[-max_actions:]
    
    # Format with step numbers
    start_idx = len(action_history) - len(recent_actions) + 1
    formatted = []
    for i, action in enumerate(recent_actions, start=start_idx):
        formatted.append(f"  {i}. {action}")
    
    return "\n".join(formatted)


def format_prompt(
    goal: str,
    axtree_txt: str,
    action_history: List[str] = None,
    valid_bids: List[str] = None,
    elements_preview: List[str] = None,
    max_axtree_lines: int = 300,
    max_history_actions: int = 15,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Format the prompt for the model given the current state.
    
    This format matches the teacher LLM but excludes:
    - Safety policies (hidden from student - implied by actions)
    - Action space description (learned from data)
    
    Includes:
    - Goal
    - Action history (previous actions in this trajectory)
    - Current page state (accessibility tree)
    
    Args:
        goal: The task goal/intent
        axtree_txt: Accessibility tree text representation
        action_history: List of previous actions taken in this trajectory
        valid_bids: List of valid element bids (optional)
        elements_preview: List of key element descriptions (optional)
        max_axtree_lines: Maximum lines to keep from axtree
        max_history_actions: Maximum number of previous actions to include
    
    Returns:
        Formatted prompt string
    """
    # Truncate accessibility tree
    truncated_axtree = truncate_axtree(axtree_txt, max_axtree_lines)
    
    # Format elements preview if available
    elements_preview_txt = ""
    if elements_preview:
        elements_preview_txt = "\n".join(elements_preview[:100])  # Limit elements
    
    # Format action history
    history_txt = format_action_history(action_history or [], max_history_actions)
    
    # Build prompt
    prompt_parts = [(system_prompt or SYSTEM_PROMPT_BASE), ""]
    
    # Add goal
    prompt_parts.append(f"# Goal\n{goal}")
    
    # Add action history
    prompt_parts.append(f"\n# Previous Actions\n{history_txt}")
    
    # Add key elements preview if available
    if elements_preview_txt:
        prompt_parts.append(f"\n# Key Elements\n{elements_preview_txt}")
    
    # Add current page observation (accessibility tree)
    prompt_parts.append(f"\n# Current Page (Accessibility Tree)\n{truncated_axtree}")
    
    # Add action prompt
    prompt_parts.append("\n# Action")
    
    return "\n".join(prompt_parts)


def build_prompt_and_target(
    goal: str,
    state: Dict[str, Any],
    action: str,
    action_history: List[str] = None,
    system_prompt: Optional[str] = None,
    max_history_actions: int = 15,
) -> Tuple[str, str]:
    """
    Build training prompt and target from a state-action pair.
    
    Note: This follows the student model format which excludes:
    - Safety policies (hidden - implied by demonstration actions)
    - Action space description (learned from data)
    
    Includes:
    - Goal
    - Action history (previous actions in trajectory)
    - Current page state
    
    Args:
        goal: Task goal
        state: State dictionary containing axtree_txt, valid_bids, elements_preview
        action: The action taken (target for training)
        action_history: List of previous actions in this trajectory
    
    Returns:
        Tuple of (prompt, target_action)
    """
    axtree_txt = state.get("axtree_txt", "")
    valid_bids = state.get("valid_bids", [])
    elements_preview = state.get("elements_preview", [])
    
    prompt = format_prompt(
        goal=goal,
        axtree_txt=axtree_txt,
        action_history=action_history or [],
        valid_bids=valid_bids,
        elements_preview=elements_preview,
        max_history_actions=max_history_actions,
        system_prompt=system_prompt,
    )
    
    # Target is the raw action string (not JSON wrapped)
    target = action.strip()
    
    return prompt, target


# =============================================================================
# Dataset Class
# =============================================================================

class WebArenaTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for WebArena imitation learning.
    
    Loads trajectories from JSON and converts them to prompt-action pairs
    for language model training.
    
    Expected JSON structure:
    {
        "task_id": {
            "policy": ["policy1", "policy2", ...],
            "trajectories": {
                "0": {
                    "meta_data": {"goal": "...", "reward": 1.0, ...},
                    "data": {
                        "step_1": {"state": {...}, "action": "click('30')", ...},
                        "step_2": {...},
                        ...
                    }
                },
                "1": {...},
                ...
            }
        },
        ...
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 8192,
        max_history_actions: int = 15,
        error_on_truncation: bool = False,
        filter_successful: bool = True,
        indices: Optional[List[int]] = None,
        ignore_templates: Optional[List[int]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to trajectories JSON file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            filter_successful: If True, only include trajectories with reward > 0
            indices: Optional list of sample indices to use (for train/test split)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_history_actions = max_history_actions
        self.error_on_truncation = error_on_truncation
        self.system_prompt = system_prompt or SYSTEM_PROMPT_BASE
        
        with open(data_path, "r") as f:
            raw = json.load(f)
        
        self.samples: List[Dict[str, Any]] = []
        
        # Parse the nested trajectory structure
        for task_id, task_data in raw.items():
            if not isinstance(task_data, dict):
                continue
            
            policies = task_data.get("policy", [])
            trajectories = task_data.get("trajectories", {})
            
            if not isinstance(trajectories, dict):
                continue
            
            for traj_num, trajectory in trajectories.items():
                if not isinstance(trajectory, dict):
                    continue
                
                meta_data = trajectory.get("meta_data", {}) or {}
                goal = meta_data.get("goal", "") or ""
                reward = meta_data.get("reward", 0)
                # NEW: template id (used for generalization split)
                # In the new vanilla dataset this is stored as meta_data.task_template_id.
                task_template_id = meta_data.get("task_template_id", None)
                if ignore_templates and task_template_id in ignore_templates:
                    continue
                # Optional fallback for other formats (keeps code robust)
                if task_template_id is None:
                    task_template_id = meta_data.get("intent_template_id", None)
                # Fallback 2: infer from numeric task_id via stwebagentbench task json
                if task_template_id is None:
                    numeric_task_id = _extract_numeric_task_id(task_id)
                    if numeric_task_id is not None:
                        task_template_id = get_task_id_to_template_id_map().get(numeric_task_id)
                
                # Optionally filter for successful trajectories (reward == 1.0 only)
                if filter_successful and reward != 1.0:
                    continue
                
                data = trajectory.get("data", {})
                if not isinstance(data, dict):
                    continue
                
                # Extract steps in order
                step_keys = sorted(
                    [k for k in data.keys() if k.startswith("step_")],
                    key=lambda x: int(x.split("_")[1])
                )
                
                # Build action history as we iterate through steps
                action_history = []
                
                for step_key in step_keys:
                    step = data[step_key]
                    if not isinstance(step, dict):
                        continue
                    
                    state = step.get("state", {})
                    action = step.get("action", "")
                    
                    if not action:
                        continue
                    
                    # Store sample with current action history (copy to avoid reference issues)
                    self.samples.append({
                        "task_id": task_id,
                        "task_template_id": task_template_id,
                        "trajectory_num": traj_num,
                        "step": step_key,
                        "goal": goal,
                        "state": state,
                        "action": action,
                        "action_history": list(action_history),  # Copy current history
                        "policies": policies,
                        "reward": reward,
                    })
                    
                    # Add current action to history for next step
                    action_history.append(action)
        
        # Apply index filtering if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices if i < len(self.samples)]
        
        # Ensure we can split by template id when requested downstream.
        # If any sample is missing a template id, splitting becomes ambiguous.
        if any(s.get("task_template_id") is None for s in self.samples):
            missing = sum(1 for s in self.samples if s.get("task_template_id") is None)
            print(f"[WARNING] {missing} samples missing meta_data.task_template_id (or intent_template_id fallback). "
                  f"Train/test split by template may fail or behave unexpectedly.")

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        goal = sample["goal"]
        state = sample["state"]
        action = sample["action"]
        action_history = sample.get("action_history", [])
        # Note: policies are intentionally NOT passed to the student model
        # The student learns safe behavior implicitly from demonstration actions
        
        # Build prompt and target with action history for context
        prompt, target = build_prompt_and_target(
            goal=goal,
            state=state,
            action=action,
            action_history=action_history,
            system_prompt=self.system_prompt,
            max_history_actions=self.max_history_actions,
        )
        
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        # Tokenize prompt and target separately
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        
        target_ids = self.tokenizer(
            target,
            add_special_tokens=False,
        ).input_ids
        
        # Add EOS token after target
        if self.tokenizer.eos_token_id is not None:
            target_ids = target_ids + [self.tokenizer.eos_token_id]
        
        # Add BOS token at the start
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        
        input_ids = bos + prompt_ids + target_ids
        
        # Build labels: ignore prompt tokens, supervise only target
        labels = [-100] * len(bos + prompt_ids) + target_ids
        
        # Truncate from the left (to keep recent context) and pad on the left
        if len(input_ids) > self.max_length:
            if self.error_on_truncation:
                raise ValueError(
                    "Sample exceeds max_length and truncation is disabled (--error_on_truncation). "
                    f"idx={idx} task_id={sample.get('task_id')} task_template_id={sample.get('task_template_id')} "
                    f"trajectory_num={sample.get('trajectory_num')} step={sample.get('step')} "
                    f"len(input_ids)={len(input_ids)} max_length={self.max_length} "
                    f"(bos={len(bos)} prompt_tokens={len(prompt_ids)} target_tokens={len(target_ids)} "
                    f"history_len={len(action_history)})"
                )
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]
        
        # Pad on the left
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = [pad_id] * pad_len + input_ids
            labels = [-100] * pad_len + labels
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != pad_id).long()
        
        # Mark action type for potential weighted loss
        action_name, _, _ = parse_action(action)
        is_click = 1.0 if action_name == "click" else 0.0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_click": torch.tensor(is_click, dtype=torch.float),
        }


class WebArenaTrajectorySubset(Dataset):
    """
    Lightweight view over a base `WebArenaTrajectoryDataset` using a fixed list of indices.

    Why this exists:
    - Avoid re-loading/parsing the JSON multiple times.
    - Ensure train/test splits are applied to the *same* underlying sample list (prevents index drift).
    - Preserve `dataset.samples` and `dataset.system_prompt` for evaluation/debug utilities.
    """

    def __init__(self, base: WebArenaTrajectoryDataset, indices: List[int]):
        self.base = base
        self.indices = list(indices)
        # Expose raw samples for evaluation utilities that iterate `dataset.samples`.
        self.samples = [base.samples[i] for i in self.indices]
        # Expose system prompt used for prompt building during evaluation.
        self.system_prompt = getattr(base, "system_prompt", SYSTEM_PROMPT_BASE)
        # Expose prompt/truncation controls for evaluation utilities.
        self.max_length = getattr(base, "max_length", None)
        self.max_history_actions = getattr(base, "max_history_actions", 15)
        self.error_on_truncation = getattr(base, "error_on_truncation", False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base[self.indices[idx]]


# =============================================================================
# Evaluation Helpers
# =============================================================================

class StopOnNewline(StoppingCriteria):
    """Stop generation when newline is produced."""
    def __init__(self, tokenizer):
        self.newline_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.newline_id


def evaluate_action_accuracy(
    model,
    tokenizer,
    dataset: WebArenaTrajectoryDataset,
    max_samples: int = 500,
    max_new_tokens: int = 64,
    split: Optional[str] = None,
    debug: bool = False,
    debug_samples: int = 20,
    debug_save_path: str = "debug_prompts.json",
) -> Dict[str, float]:
    """
    Evaluate action prediction accuracy on a dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        dataset: Dataset to evaluate on
        max_samples: Maximum samples to evaluate
        max_new_tokens: Maximum tokens to generate
        debug: If True, print detailed prediction vs target comparisons
        debug_samples: Number of samples to show in debug output
    
    Returns:
        Dictionary with accuracy metrics
    """
    device = next(model.parameters()).device
    tag = f"[EVAL][{split}]" if split else "[EVAL]"
    model.eval()
    
    # Granular metrics for structured prediction
    results = {
        # Per-action-type: full match (type + bid + text)
        "click_correct": 0,
        "click_total": 0,
        "fill_correct": 0,
        "fill_total": 0,
        "other_correct": 0,
        "other_total": 0,
        # Granular metrics across all actions
        "type_correct": 0,  # Action type matches
        "type_total": 0,
        "bid_correct": 0,   # BID matches (for actions with BID)
        "bid_total": 0,
        "fill_bid_correct": 0,   # BID correct for fill actions
        "fill_bid_total": 0,
        "fill_text_correct": 0,  # Text correct given BID correct
        "fill_text_total": 0,
    }
    
    # Store debug info
    debug_outputs = []
    debug_prompt_rows: List[Dict[str, Any]] = []
    
    with torch.no_grad():
        system_prompt = getattr(dataset, "system_prompt", SYSTEM_PROMPT_BASE)
        max_history_actions = getattr(dataset, "max_history_actions", 15)
        max_length = getattr(dataset, "max_length", None)
        error_on_truncation = getattr(dataset, "error_on_truncation", False)
        for i, sample in enumerate(dataset.samples[:max_samples]):
            goal = sample["goal"]
            state = sample["state"]
            action = sample["action"]
            action_history = sample.get("action_history", [])
            # Note: policies are NOT passed during evaluation (student is unaware)
            
            prompt, target = build_prompt_and_target(
                goal=goal,
                state=state,
                action=action,
                action_history=action_history,
                system_prompt=system_prompt,
                max_history_actions=max_history_actions,
            )
            
            # Tokenize prompt
            enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
            input_ids = torch.cat([bos, enc["input_ids"].to(device)], dim=1)
            attention_mask = torch.cat(
                [torch.ones_like(bos), enc["attention_mask"].to(device)],
                dim=1,
            )

            # Respect dataset max_length during evaluation too (mirrors training behavior).
            if isinstance(max_length, int) and max_length > 0 and input_ids.shape[1] > max_length:
                if error_on_truncation:
                    raise ValueError(
                        "Eval sample exceeds max_length and truncation is disabled (--error_on_truncation). "
                        f"eval_idx={i} task_id={sample.get('task_id')} task_template_id={sample.get('task_template_id')} "
                        f"trajectory_num={sample.get('trajectory_num')} step={sample.get('step')} "
                        f"len(input_ids)={int(input_ids.shape[1])} max_length={max_length} "
                        f"(history_len={len(action_history)})"
                    )
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
            
            # Generate
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.pad_token_id,
            )
            
            start = input_ids.shape[1]
            new_ids = out[0, start:]
            predicted = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            
            # Parse actions
            pred_name, pred_arg1, pred_arg2 = parse_action(predicted)
            target_name, target_arg1, target_arg2 = parse_action(target)
            
            # --- Granular metrics ---
            # Type accuracy (always counted)
            results["type_total"] += 1
            type_match = (pred_name == target_name)
            if type_match:
                results["type_correct"] += 1
            
            # BID accuracy (for click, fill, select_option)
            if target_name in ("click", "fill", "select_option"):
                results["bid_total"] += 1
                bid_match = (pred_arg1 == target_arg1)
                if type_match and bid_match:
                    results["bid_correct"] += 1
            else:
                bid_match = True  # N/A
            
            # --- Per-action-type full match ---
            is_correct = False
            if target_name == "click":
                results["click_total"] += 1
                if pred_name == "click" and pred_arg1 == target_arg1:
                    results["click_correct"] += 1
                    is_correct = True
            elif target_name == "fill":
                results["fill_total"] += 1
                results["fill_bid_total"] += 1
                if pred_name == "fill" and pred_arg1 == target_arg1:
                    results["fill_bid_correct"] += 1
                    # Check text
                    results["fill_text_total"] += 1
                    if pred_arg2 == target_arg2:
                        results["fill_text_correct"] += 1
                        results["fill_correct"] += 1
                        is_correct = True
            else:
                results["other_total"] += 1
                if pred_name == target_name and pred_arg1 == target_arg1:
                    # For other actions, check arg2 if present
                    if target_arg2 is None or pred_arg2 == target_arg2:
                        results["other_correct"] += 1
                        is_correct = True
            
            # Store debug info
            if debug and len(debug_outputs) < debug_samples:
                # Format action history for display
                history_display = action_history[-5:] if action_history else []  # Last 5 actions
                debug_outputs.append({
                    "idx": i,
                    "goal": goal[:80] + "..." if len(goal) > 80 else goal,
                    "action_history": history_display,
                    "step_num": len(action_history) + 1,
                    "target": target,
                    "predicted": predicted[:150] + "..." if len(predicted) > 150 else predicted,
                    "correct": is_correct,
                    "type_match": type_match,
                    "bid_match": bid_match if target_name in ("click", "fill", "select_option") else "N/A",
                    "target_parsed": (target_name, target_arg1, target_arg2),
                    "pred_parsed": (pred_name, pred_arg1, pred_arg2),
                })

            # Save FULL prompt rows (no truncation) for external analysis
            if debug:
                debug_prompt_rows.append({
                    "idx": i,
                    "task_id": sample.get("task_id"),
                    "task_template_id": sample.get("task_template_id"),
                    "trajectory_num": sample.get("trajectory_num"),
                    "step": sample.get("step"),
                    "reward": sample.get("reward"),
                    "goal": goal,
                    "action_history": list(action_history),
                    # These are the core items you'll want to paste into ChatGPT Pro:
                    "prompt": prompt,           # FULL prompt text fed to the model
                    "target_action": target,    # Expected action (structured format)
                    "predicted_action": predicted,
                    # Helpful structured views:
                    "target_parsed": {
                        "name": target_name,
                        "arg1": target_arg1,
                        "arg2": target_arg2,
                    },
                    "pred_parsed": {
                        "name": pred_name,
                        "arg1": pred_arg1,
                        "arg2": pred_arg2,
                    },
                    "correct": is_correct,
                    "type_match": type_match,
                    "bid_match": bid_match if target_name in ("click", "fill", "select_option") else None,
                })
    
    # Print debug output
    if debug and debug_outputs:
        print("\n" + "=" * 80)
        split_str = f" [{split}]" if split else ""
        print(f"DEBUG: Model Predictions vs Targets{split_str}")
        print("=" * 80)
        for d in debug_outputs:
            status = "✓" if d["correct"] else "✗"
            print(f"\n[{d['idx']}] {status} Step {d['step_num']} | Goal: {d['goal']}")
            if d["action_history"]:
                print(f"    History (last {len(d['action_history'])}): {d['action_history']}")
            else:
                print(f"    History: (first step - no previous actions)")
            print(f"    TARGET: {d['target']}")
            print(f"    PRED:   {d['predicted']}")
            if not d["correct"]:
                print(f"    Parsed target: {d['target_parsed']}")
                print(f"    Parsed pred:   {d['pred_parsed']}")
        print("=" * 80 + "\n")

    if debug:
        save_debug_prompts_json(debug_prompt_rows, debug_save_path)
        split_str = f" [{split}]" if split else ""
        print(f"[DEBUG]{split_str} Saved {len(debug_prompt_rows)} full prompts to: {debug_save_path}")
    
    # Compute accuracies
    click_acc = results["click_correct"] / max(1, results["click_total"])
    fill_acc = results["fill_correct"] / max(1, results["fill_total"])
    other_acc = results["other_correct"] / max(1, results["other_total"])
    total = results["click_total"] + results["fill_total"] + results["other_total"]
    total_correct = results["click_correct"] + results["fill_correct"] + results["other_correct"]
    overall_acc = total_correct / max(1, total)
    
    # Granular metrics
    type_acc = results["type_correct"] / max(1, results["type_total"])
    bid_acc = results["bid_correct"] / max(1, results["bid_total"])
    fill_bid_acc = results["fill_bid_correct"] / max(1, results["fill_bid_total"])
    fill_text_acc = results["fill_text_correct"] / max(1, results["fill_text_total"])
    
    print(f"{tag} Click (full): {click_acc*100:.2f}% ({results['click_correct']}/{results['click_total']})")
    print(f"{tag} Fill (full):  {fill_acc*100:.2f}% ({results['fill_correct']}/{results['fill_total']})")
    print(f"{tag} Other (full): {other_acc*100:.2f}% ({results['other_correct']}/{results['other_total']})")
    print(f"{tag} Overall:      {overall_acc*100:.2f}% ({total_correct}/{total})")
    print(f"{tag} --- Granular Breakdown ---")
    print(f"{tag} Type correct: {type_acc*100:.2f}% ({results['type_correct']}/{results['type_total']})")
    print(f"{tag} BID correct (click/fill/select): {bid_acc*100:.2f}% ({results['bid_correct']}/{results['bid_total']})")
    print(f"{tag} Fill BID correct: {fill_bid_acc*100:.2f}% ({results['fill_bid_correct']}/{results['fill_bid_total']})")
    print(f"{tag} Fill TEXT correct (given BID): {fill_text_acc*100:.2f}% ({results['fill_text_correct']}/{results['fill_text_total']})")
    
    return {
        "click_accuracy": click_acc,
        "fill_accuracy": fill_acc,
        "other_accuracy": other_acc,
        "overall_accuracy": overall_acc,
        "type_accuracy": type_acc,
        "bid_accuracy": bid_acc,
        "fill_bid_accuracy": fill_bid_acc,
        "fill_text_accuracy": fill_text_acc,
    }


def compute_loss(
    model,
    data_loader: DataLoader,
    device,
    ce_lm: nn.Module,
) -> float:
    """Compute average cross-entropy loss on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            B, T, V = logits.shape
            
            # Causal LM shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = ce_lm(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(1, num_batches)


def analyze_action_distribution(dataset: WebArenaTrajectoryDataset, title: str = ""):
    """Analyze and print the distribution of actions in a dataset."""
    action_counts = Counter()
    
    for sample in dataset.samples:
        action = sample.get("action", "")
        action_name, _, _ = parse_action(action)
        action_counts[action_name] += 1
    
    total = len(dataset.samples)
    print(f"\n{'='*60}")
    print(f"Action Distribution: {title}" if title else "Action Distribution")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    for action_name, count in action_counts.most_common():
        print(f"  {action_name}: {count} ({count/total*100:.1f}%)")
    print(f"{'='*60}\n")


# =============================================================================
# Final Evaluation Helper
# =============================================================================

def evaluate_model_full(
    model,
    tokenizer,
    train_dataset: "WebArenaTrajectoryDataset",
    test_dataset: "WebArenaTrajectoryDataset",
    val_dataset: Optional["WebArenaTrajectoryDataset"] = None,
    batch_size: int = 4,
    max_eval_samples: int = 200,
    label: str = "Model",
) -> Dict[str, Any]:
    """
    Evaluate a model fully on train and test sets.
    
    Returns dict with train/test loss and accuracy metrics.
    """
    from torch.utils.data import DataLoader
    
    device = next(model.parameters()).device
    model.eval()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
    
    # Loss function
    ce_lm = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Compute losses
    train_loss = compute_loss(model, train_loader, device, ce_lm)
    test_loss = compute_loss(model, test_loader, device, ce_lm)
    val_loss = compute_loss(model, val_loader, device, ce_lm) if val_loader is not None else None
    
    # Compute accuracies
    print(f"\n[{label}] Evaluating on train set...")
    train_metrics = evaluate_action_accuracy(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        max_samples=max_eval_samples,
        debug=False,
    )
    
    print(f"[{label}] Evaluating on test set...")
    test_metrics = evaluate_action_accuracy(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        max_samples=max_eval_samples,
        debug=False,
    )

    val_metrics = None
    if val_dataset is not None:
        print(f"[{label}] Evaluating on VAL set (intra)...")
        val_metrics = evaluate_action_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=val_dataset,
            max_samples=max_eval_samples,
            split="VAL",
            debug=False,
        )
    
    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
    }


def print_model_comparison(
    final_results: Dict[str, Any],
    best_results: Dict[str, Any],
):
    """Print a comparison table of final vs best model metrics."""
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON: Final State vs Best Test Accuracy State")
    print("=" * 80)
    
    # Header
    print(f"{'Metric':<30} {'Final Model':>20} {'Best Model':>20}")
    print("-" * 70)
    
    # Losses
    print(f"{'Train Loss':<30} {final_results['train_loss']:>20.4f} {best_results['train_loss']:>20.4f}")
    print(f"{'Test Loss':<30} {final_results['test_loss']:>20.4f} {best_results['test_loss']:>20.4f}")
    if final_results.get("val_loss") is not None or best_results.get("val_loss") is not None:
        final_val_loss = final_results.get("val_loss", None)
        best_val_loss = best_results.get("val_loss", None)
        final_val_loss_str = f"{final_val_loss:>20.4f}" if isinstance(final_val_loss, (int, float)) else f"{'n/a':>20}"
        best_val_loss_str = f"{best_val_loss:>20.4f}" if isinstance(best_val_loss, (int, float)) else f"{'n/a':>20}"
        print(f"{'Val Loss (intra)':<30} {final_val_loss_str} {best_val_loss_str}")
    print("-" * 70)
    
    # Train metrics
    print("--- Train Set ---")
    for key in ["overall_accuracy", "type_accuracy", "bid_accuracy", "click_accuracy", "fill_accuracy"]:
        final_val = final_results["train_metrics"].get(key, 0) * 100
        best_val = best_results["train_metrics"].get(key, 0) * 100
        label = key.replace("_", " ").title()
        print(f"{'  ' + label:<30} {final_val:>19.2f}% {best_val:>19.2f}%")
    
    print("-" * 70)
    
    # Test metrics
    print("--- Test Set ---")
    for key in ["overall_accuracy", "type_accuracy", "bid_accuracy", "click_accuracy", "fill_accuracy"]:
        final_val = final_results["test_metrics"].get(key, 0) * 100
        best_val = best_results["test_metrics"].get(key, 0) * 100
        label = key.replace("_", " ").title()
        print(f"{'  ' + label:<30} {final_val:>19.2f}% {best_val:>19.2f}%")

    # Val metrics (optional)
    if final_results.get("val_metrics") is not None or best_results.get("val_metrics") is not None:
        print("-" * 70)
        print("--- Val Set (Intra) ---")
        final_vm = final_results.get("val_metrics") or {}
        best_vm = best_results.get("val_metrics") or {}
        for key in ["overall_accuracy", "type_accuracy", "bid_accuracy", "click_accuracy", "fill_accuracy"]:
            final_val = final_vm.get(key, 0) * 100
            best_val = best_vm.get(key, 0) * 100
            label = key.replace("_", " ").title()
            print(f"{'  ' + label:<30} {final_val:>19.2f}% {best_val:>19.2f}%")
    
    print("=" * 80)


# =============================================================================
# Training Loop
# =============================================================================

def manual_train_loop(
    model,
    train_dataset: WebArenaTrajectoryDataset,
    test_dataset: WebArenaTrajectoryDataset,
    tokenizer,
    data_collator,
    num_epochs: int = 5,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    max_grad_norm: float = 1.0,
    click_weight: float = 0.0,
    best_save_dir: Optional[str] = None,
    best_intra_save_dir: Optional[str] = None,
    debug_save_path: Optional[str] = None,
    skip_pretrain_evaluation: bool = False,
    sync_timing: bool = False,
    train_eval_every: int = 5,
    val_dataset: Optional[WebArenaTrajectoryDataset] = None,
    epoch_checkpoint_dir: Optional[str] = None,
) -> Tuple[float, List[Dict]]:
    """
    Custom training loop with gradient accumulation and evaluation.
    
    Returns:
        Tuple of (final_test_accuracy, metrics_history)
    """
    device = next(model.parameters()).device
    
    # Disable KV cache during training
    original_use_cache = getattr(model.config, "use_cache", None)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    
    model.train()
    
    # Only train LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Training] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)
    
    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    val_dl = None
    if val_dataset is not None:
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
    
    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dl) / grad_accum_steps)
    max_steps = num_epochs * num_update_steps_per_epoch
    warmup_steps = int(0.05 * max_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    
    ce_lm = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
    
    global_step = 0
    metrics_history = []
    final_test_accuracy = 0.0
    best_test_overall_acc = float("-inf")
    best_epoch = -1
    best_val_overall_acc = float("-inf")
    best_val_epoch = -1
    epoch_train_times: List[float] = []

    # -------------------------------------------------------------------------
    # Pre-training evaluation (true baseline before any optimizer updates)
    # -------------------------------------------------------------------------
    if not skip_pretrain_evaluation:
        try:
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True
            model.eval()

            print("\n" + "=" * 70)
            print("PRE-TRAIN EVALUATION (before any training updates)")
            print("=" * 70)

            # For deterministic baseline loss, evaluate train loss with shuffle=False.
            train_eval_dl = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )

            pre_train_loss = compute_loss(model, train_eval_dl, device, ce_lm)
            pre_test_loss = compute_loss(model, test_dl, device, ce_lm)
            print(f"[Pre-Train] Train Loss: {pre_train_loss:.4f}, Test Loss: {pre_test_loss:.4f}")

            pre_val_loss = None
            pre_val_metrics = None
            if val_dl is not None:
                pre_val_loss = compute_loss(model, val_dl, device, ce_lm)
                print(f"[Pre-Train] Val Loss (intra): {pre_val_loss:.4f}")
                pre_val_metrics = evaluate_action_accuracy(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    max_samples=min(200, len(val_dataset)),
                    split="VAL",
                    debug=False,
                )

            pre_test_metrics = evaluate_action_accuracy(
                model=model,
                tokenizer=tokenizer,
                dataset=test_dataset,
                max_samples=min(200, len(test_dataset)),
                split="TEST",
                debug=False,
            )

            pre_train_metrics = evaluate_action_accuracy(
                model=model,
                tokenizer=tokenizer,
                dataset=train_dataset,
                max_samples=min(200, len(train_dataset)),
                split="TRAIN",
                debug=False,
            )

            metrics_history.append(
                {
                    "epoch": -1,
                    "train_loss": pre_train_loss,
                    "test_loss": pre_test_loss,
                    **({} if pre_val_loss is None else {"val_loss": pre_val_loss}),
                    **{f"test_{k}": v for k, v in pre_test_metrics.items()},
                    **({} if pre_val_metrics is None else {f"val_{k}": v for k, v in pre_val_metrics.items()}),
                    **{f"train_{k}": v for k, v in pre_train_metrics.items()},
                }
            )
        finally:
            # Back to training mode/settings
            model.train()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
    
    for epoch in range(num_epochs):
        import time

        epoch_wall_t0 = time.perf_counter()
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            epoch_wall_t0 = time.perf_counter()

        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}] Learning rate: {current_lr:.2e}")
        
        optimizer.zero_grad()
        epoch_loss_sum = 0.0
        epoch_batch_count = 0
        epoch_update_steps = 0

        train_wall_t0 = time.perf_counter()
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            train_wall_t0 = time.perf_counter()
        
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            B, T, V = logits.shape
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = ce_lm(shift_logits.view(-1, V), shift_labels.view(-1))
            loss = loss / grad_accum_steps
            
            loss.backward()
            
            epoch_loss_sum += loss.item() * grad_accum_steps
            epoch_batch_count += 1
            
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                epoch_update_steps += 1
                
                if global_step % 20 == 0:
                    print(f"  Step {global_step}, Loss: {epoch_loss_sum/epoch_batch_count:.4f}")
        
        # Handle remainder
        remainder = (step + 1) % grad_accum_steps
        if remainder != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            epoch_update_steps += 1

        train_wall_t1 = time.perf_counter()
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            train_wall_t1 = time.perf_counter()

        epoch_train_sec = max(1e-9, train_wall_t1 - train_wall_t0)
        epoch_train_times.append(epoch_train_sec)
        microbatches = len(train_dl)
        sec_per_micro = epoch_train_sec / max(1, microbatches)
        sec_per_update = epoch_train_sec / max(1, epoch_update_steps)
        updates_per_sec = epoch_update_steps / epoch_train_sec

        approx_eta_sec = 0.0
        if epoch_train_times:
            avg_train = sum(epoch_train_times) / len(epoch_train_times)
            approx_eta_sec = avg_train * (num_epochs - epoch - 1)

        print(
            f"[Epoch {epoch}] Train time (approx): {epoch_train_sec:.1f}s | "
            f"{sec_per_micro:.2f}s/microbatch | {sec_per_update:.2f}s/update_step | "
            f"{updates_per_sec:.2f} update_steps/s | ETA: ~{approx_eta_sec/60:.1f} min"
        )
        
        avg_train_loss = epoch_loss_sum / max(1, epoch_batch_count)
        
        # Evaluate
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        
        print(f"\n[Epoch {epoch}] Evaluating on test set...")
        test_loss = compute_loss(model, test_dl, device, ce_lm)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")

        val_loss = None
        val_metrics = None
        if val_dl is not None:
            print(f"\n[Epoch {epoch}] Evaluating on VAL set (intra)...")
            val_loss = compute_loss(model, val_dl, device, ce_lm)
            print(f"[Epoch {epoch}] Val Loss (intra): {val_loss:.4f}")

        # Enable debug output on final epoch
        is_final_epoch = (epoch == num_epochs - 1)
        test_metrics = evaluate_action_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            max_samples=len(test_dataset),
            split="TEST",
            debug=True,
            debug_samples=30,  # Show 30 samples on final evaluation
            debug_save_path=(debug_save_path or "debug_prompts.json"),
        )

        if val_dl is not None:
            print(f"\n[Epoch {epoch}] Evaluating on VAL set (intra, max 200 samples)...")
            val_metrics = evaluate_action_accuracy(
                model=model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                max_samples=min(200, len(val_dataset)),
                split="VAL",
                debug=False,
            )
            print(f"[VAL] Overall: {val_metrics['overall_accuracy']*100:.2f}%")
            print(f"[VAL] Type correct: {val_metrics['type_accuracy']*100:.2f}%")
            print(f"[VAL] BID correct: {val_metrics['bid_accuracy']*100:.2f}%")

        # Save best checkpoint by HIGHEST overall test accuracy (LoRA adapter weights)
        test_overall_acc = float(test_metrics.get("overall_accuracy", 0.0))
        if best_save_dir is not None and test_overall_acc > best_test_overall_acc:
            best_test_overall_acc = test_overall_acc
            best_epoch = epoch
            os.makedirs(best_save_dir, exist_ok=True)
            model.eval()
            print(
                f"[BEST_TEST] New best test overall accuracy {best_test_overall_acc*100:.2f}% "
                f"at epoch {best_epoch}. Saving to {best_save_dir}"
            )
            model.save_pretrained(best_save_dir)
            tokenizer.save_pretrained(best_save_dir)
            with open(os.path.join(best_save_dir, "best_checkpoint.json"), "w") as f:
                json.dump(
                    {"best_epoch": best_epoch, "best_test_overall_accuracy": best_test_overall_acc},
                    f,
                    indent=2,
                )
            model.train()

        # Save best checkpoint by HIGHEST overall VAL (intra) accuracy (LoRA adapter weights)
        if val_metrics is not None:
            val_overall_acc = float(val_metrics.get("overall_accuracy", 0.0))
            if best_intra_save_dir is not None and val_overall_acc > best_val_overall_acc:
                best_val_overall_acc = val_overall_acc
                best_val_epoch = epoch
                os.makedirs(best_intra_save_dir, exist_ok=True)
                model.eval()
                print(
                    f"[BEST_INTRA] New best intra/val overall accuracy {best_val_overall_acc*100:.2f}% "
                    f"at epoch {best_val_epoch}. Saving to {best_intra_save_dir}"
                )
                model.save_pretrained(best_intra_save_dir)
                tokenizer.save_pretrained(best_intra_save_dir)
                with open(os.path.join(best_intra_save_dir, "best_checkpoint.json"), "w") as f:
                    json.dump(
                        {"best_epoch": best_val_epoch, "best_val_overall_accuracy": best_val_overall_acc},
                        f,
                        indent=2,
                    )
                model.train()
        
        # Evaluate on training set every epoch (max 200 samples for efficiency)
        print(f"\n[Epoch {epoch}] Evaluating on TRAIN set (max 200 samples)...")
        train_metrics = evaluate_action_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            max_samples=min(200, len(train_dataset)),
            split="TRAIN",
            debug=False,
        )
        print(f"[TRAIN] Overall: {train_metrics['overall_accuracy']*100:.2f}%")
        print(f"[TRAIN] Type correct: {train_metrics['type_accuracy']*100:.2f}%")
        print(f"[TRAIN] BID correct: {train_metrics['bid_accuracy']*100:.2f}%")
    
        # Save epoch checkpoint
        if epoch_checkpoint_dir is not None:
            epoch_save_path = os.path.join(epoch_checkpoint_dir, f"epoch{epoch}_state")
            os.makedirs(epoch_save_path, exist_ok=True)
            model.eval()
            print(f"[Epoch {epoch}] Saving checkpoint to {epoch_save_path}")
            model.save_pretrained(epoch_save_path)
            tokenizer.save_pretrained(epoch_save_path)
            model.train()
        
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_metrics["overall_accuracy"],
            "test_accuracy": test_metrics["overall_accuracy"],
            **{f"test_{k}": v for k, v in test_metrics.items()},
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }
        if val_loss is not None:
            epoch_metrics["val_loss"] = val_loss
        if val_metrics is not None:
            epoch_metrics["val_accuracy"] = val_metrics["overall_accuracy"]
            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        metrics_history.append(epoch_metrics)
        
        final_test_accuracy = test_metrics["overall_accuracy"]
        
        model.train()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        epoch_wall_t1 = time.perf_counter()
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
            epoch_wall_t1 = time.perf_counter()
        epoch_total_sec = max(1e-9, epoch_wall_t1 - epoch_wall_t0)
        # Includes eval time; useful for wall-clock budgeting.
        print(f"[Epoch {epoch}] Total epoch time (approx): {epoch_total_sec:.1f}s (train+eval)")
    
    # Restore cache setting
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache

    if best_save_dir is not None:
        print(
            f"[BEST_TEST] Best test overall accuracy was {best_test_overall_acc*100:.2f}% "
            f"at epoch {best_epoch} (saved to {best_save_dir})"
        )
    if best_intra_save_dir is not None and val_dl is not None:
        print(
            f"[BEST_INTRA] Best intra/val overall accuracy was {best_val_overall_acc*100:.2f}% "
            f"at epoch {best_val_epoch} (saved to {best_intra_save_dir})"
        )
    
    return final_test_accuracy, metrics_history


# =============================================================================
# Main Training Function
# =============================================================================

def train_llama_il(
    model_path: str = MODEL_PATH,
    data_path: str = VANILLA_DATA_PATH,
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 8192,
    max_history_actions: int = 15,
    error_on_truncation: bool = False,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.005,
    click_weight: float = 0.0,
    gpu_id: int = 2,
    seed: int = 42,
    data: str = "vanilla",
    # Sampling controls (train split only)
    sampling_trajectories: bool = False,
    sampling_trajectories_k: int = 10,
    sampling_pairs: bool = False,
    sampling_pairs_k: int = 50,
    intra_evaluate: bool = False,
    val_max_trajectories: int = 0,
    split_mode: str = "template",
    test_template_ids: Optional[List[int]] = None,
    ignore_templates: Optional[List[int]] = None,
    skip_pretrain_evaluation: bool = False,
    sync_timing: bool = False,
    train_eval_every: int = 5,
    policy_aware: bool = False,
    experiment_name: str = "default_experiment",
):
    """
    Train LLaMA model with LoRA for imitation learning on WebArena trajectories.
    
    Args:
        model_path: Path to base LLaMA model
        data_path: Path to trajectory data JSON
        output_dir: Directory to save trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        gradient_accumulation_steps: Steps for gradient accumulation
        max_length: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        lora_dropout: LoRA dropout rate
        click_weight: Extra weight for click actions (not used currently)
        gpu_id: GPU device ID
        seed: Random seed
        data: Data type ("vanilla" or "safe")
        sampling: Whether to subsample training data
        split_mode: How to split train/test:
            - "template": Split by task_template_id (OOD, tests generalization)
            - "random": Random 80/20 split by sample (ID, sanity check)
    
    Returns:
        Final test accuracy
    """
    torch.cuda.empty_cache()
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    print(f"GPU ID: {gpu_id}")
    print(f"Max history actions: {max_history_actions}")
    print(f"Error on truncation: {error_on_truncation}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Policy-aware mode: inject safety policy into the system prompt
    # -------------------------------------------------------------------------
    if policy_aware:
        print("[POLICY_AWARE] Enabled: injecting dataset safety policy into the system prompt.")
        if data.lower() != "safe":
            raise ValueError("--policy_aware requires --data safe (policy field only exists/used for safe data).")

        # Load the policy description from the JSON. The safe dataset stores it at the top-level per task_id.
        # It is expected to be fixed throughout the dataset.
        with open(data_path, "r") as f:
            raw_for_policy = json.load(f)
        policies_seen: List[List[str]] = []
        for j, (_task_id, task_data) in enumerate(raw_for_policy.items()):
            if j >= 20:
                break
            pol = (task_data or {}).get("policy", None)
            if isinstance(pol, list):
                policies_seen.append([str(x) for x in pol])
            elif pol is not None:
                # Rare fallback: single string policy
                policies_seen.append([str(pol)])

        if not policies_seen:
            raise ValueError("Could not find a 'policy' field in the safe trajectories JSON (first 20 tasks).")

        # Verify policy consistency (best-effort).
        base_policy = policies_seen[0]
        if any(p != base_policy for p in policies_seen[1:]):
            print("[WARNING] Detected non-identical policies across tasks (first 20). Using the first policy block.")

        policy_preview = " | ".join(base_policy[:3])
        more = "" if len(base_policy) <= 3 else f" (+{len(base_policy) - 3} more)"
        print(f"[POLICY_AWARE] Loaded policy with {len(base_policy)} rules. Preview: {policy_preview}{more}")
        system_prompt = build_system_prompt(base_policy)
    else:
        system_prompt = SYSTEM_PROMPT_BASE
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Load model
    device_map = {"": f"cuda:{gpu_id}"}
    print(f"Loading model from {model_path} on GPU {gpu_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    
    # Enable gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    # Configure generation
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0
        model.generation_config.top_p = 1.0
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Load dataset
    full_dataset = WebArenaTrajectoryDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_history_actions=max_history_actions,
        error_on_truncation=error_on_truncation,
        filter_successful=True,
        ignore_templates=ignore_templates,
        system_prompt=system_prompt,
    )
    
    if len(full_dataset) == 0:
        raise ValueError("No samples found in dataset!")
    
    # Prepare output directories up-front (used for best checkpoint + debug dumps).
    safe_data_str = data.lower()
    sampling_suffix = ""
    if sampling_trajectories:
        sampling_suffix += f"_trajk{sampling_trajectories_k}"
    if sampling_pairs:
        sampling_suffix += f"_pairsk{sampling_pairs_k}"
    
    # New directory structure:
    # Checkpoints: train/checkpoints/{experiment_name}/{seed}/{vanilla/safe}/
    # Results: train/results/{seed}/
    train_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_base_dir = os.path.join(train_dir, "checkpoints", experiment_name, str(seed), safe_data_str)
    results_dir = os.path.join(train_dir, "results", str(seed))
    
    save_dir = checkpoint_base_dir
    best_save_dir = os.path.join(checkpoint_base_dir, "best_test_acc")
    best_intra_save_dir = os.path.join(checkpoint_base_dir, "best_intra_acc")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(best_save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if intra_evaluate:
        os.makedirs(best_intra_save_dir, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {checkpoint_base_dir}")
    print(f"Results will be saved to: {results_dir}")

    # Create train/test split based on split_mode
    g = torch.Generator()
    g.manual_seed(seed)
    
    if split_mode == "random":
        # Random 80/20 split by sample (in-distribution)
        print(f"[Split Mode: random] In-distribution 80/20 split by sample")
        n_samples = len(full_dataset.samples)
        perm = torch.randperm(n_samples, generator=g).tolist()
        n_test = max(1, int(0.2 * n_samples))
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
        print(f"Total samples: {n_samples}")
    else:
        # Split by task template id (OOD, tests generalization)
        print(f"[Split Mode: template] OOD split by task_template_id")
        # Keep original types (some datasets use ints, others use strings like "C-C2").
        all_template_ids = sorted(
            {s.get("task_template_id") for s in full_dataset.samples if s.get("task_template_id") is not None},
            key=lambda x: str(x),
        )
        missing_template = sum(1 for s in full_dataset.samples if s.get("task_template_id") is None)
        if missing_template > 0:
            print(
                f"[WARNING] {missing_template} samples still missing task_template_id after fallbacks; "
                f"they will be excluded from the template-based split."
            )
        if not all_template_ids:
            raise ValueError(
                "No task_template_id values found; cannot split by template. "
                "Verify trajectories.json meta_data.task_template_id or mapping fallback."
            )
        print(
            f"Found {len(all_template_ids)} unique task templates "
            f"with {len(full_dataset)} total samples"
        )

        # Option A (default): seed-deterministic random held-out templates.
        # Option B (fixed): user provides explicit test templates.
        if test_template_ids is not None and len(test_template_ids) > 0:
            # Treat provided values as selectors:
            # - Exact template IDs, e.g. "C-C2"
            # - Glob patterns, e.g. "G*" or "O-C*"
            # Works across both numeric and string template IDs by matching on `str(template_id)`.
            available_by_str: Dict[str, Any] = {str(tid): tid for tid in all_template_ids}
            available_strs = sorted(available_by_str.keys())

            selectors = [str(x).strip() for x in test_template_ids if str(x).strip()]
            matched_strs: List[str] = []
            for sel in selectors:
                if any(ch in sel for ch in ["*", "?", "[", "]"]):
                    matched_strs.extend([t for t in available_strs if fnmatch.fnmatch(t, sel)])
                else:
                    matched_strs.append(sel)

            requested_set_str = set(matched_strs)
            available_set_str = set(available_strs)
            missing_str = sorted(requested_set_str - available_set_str)
            if missing_str:
                raise ValueError(
                    f"--test_template_ids contains selectors/IDs not present in the dataset: {missing_str}. "
                    f"Available template_ids (n={len(available_strs)}): {available_strs}"
                )

            test_template_ids_str = set(sorted(requested_set_str))
            test_template_ids = {available_by_str[s] for s in test_template_ids_str}
            train_template_ids = set(all_template_ids) - set(test_template_ids)

            print(
                f"[Split Mode: template][Fixed Test Templates] Using provided selectors={selectors} -> "
                f"test_template_ids={sorted([str(x) for x in test_template_ids])} "
                f"(train templates: {len(train_template_ids)})"
            )
        else:
            perm = torch.randperm(len(all_template_ids), generator=g).tolist()

            num_test_templates = math.ceil(len(all_template_ids) * 0.2)
            test_template_ids = {all_template_ids[i] for i in perm[:num_test_templates]}
            train_template_ids = {all_template_ids[i] for i in perm[num_test_templates:]}

        # Log the actual held-out template IDs and their sample counts (helps diagnose "easy test split").
        try:
            from collections import Counter

            template_counts = Counter(
                s.get("task_template_id")
                for s in full_dataset.samples
                if s.get("task_template_id") is not None
            )
            if test_template_ids is not None and train_template_ids is not None:
                print(f"[Split Mode: template] num_test_templates={len(test_template_ids)} (seed={seed})")
            print(f"[Split Mode: template] Test template IDs:  {sorted(test_template_ids)}")
            print(f"[Split Mode: template] Train template IDs: {sorted(train_template_ids)}")
            print("[Split Mode: template] Test template sample counts:")
            for tid in sorted(test_template_ids):
                print(f"  - template_id={tid}: n={template_counts.get(tid, 0)}")
            print("[Split Mode: template] Train template sample counts:")
            for tid in sorted(train_template_ids):
                print(f"  - template_id={tid}: n={template_counts.get(tid, 0)}")
        except Exception as e:
            print(f"[WARNING] Failed to log template IDs/counts: {e}")
        
        train_indices = [
            i
            for i, s in enumerate(full_dataset.samples)
            if s.get("task_template_id") in train_template_ids
        ]
        test_indices = [
            i
            for i, s in enumerate(full_dataset.samples)
            if s.get("task_template_id") in test_template_ids
        ]
    
    # -------------------------------------------------------------------------
    # Optional sampling (TRAIN split only):
    #
    # 1) Trajectory sampling: keep k trajectories per task_template_id (keep ALL steps)
    # 2) Pair sampling: keep k state-action pairs per trajectory (keep k steps)
    #
    # Both are deterministic given `seed` via fixed torch.Generator seeds + sorted iteration.
    # -------------------------------------------------------------------------
    def _step_num(sample: Dict[str, Any]) -> int:
        """Parse 'step_12' -> 12 for stable within-trajectory ordering."""
        sk = str(sample.get("step", ""))
        m = re.search(r"step_(\d+)", sk)
        return int(m.group(1)) if m else 10**9

    if sampling_trajectories:
        from collections import defaultdict

        if sampling_trajectories_k <= 0:
            raise ValueError("--sampling_trajectories_k must be >= 1 when trajectory sampling is enabled.")

        # Group train indices by template -> trajectory_key -> [sample_indices]
        # trajectory_key identifies a single trajectory (task_id + trajectory_num).
        by_template_traj: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        for i in train_indices:
            s = full_dataset.samples[i]
            tid = s.get("task_template_id")
            if tid is None:
                continue
            traj_key = (str(s.get("task_id")), str(s.get("trajectory_num")))
            by_template_traj[str(tid)][traj_key].append(i)

        g_traj = torch.Generator()
        g_traj.manual_seed(int(seed) + 10007)

        new_train_indices: List[int] = []
        per_template_summary: List[Tuple[int, int, int]] = []

        for tid in sorted(by_template_traj.keys()):
            traj_keys = sorted(by_template_traj[tid].keys())
            n_traj = len(traj_keys)
            if n_traj == 0:
                continue

            k = min(sampling_trajectories_k, n_traj)
            perm = torch.randperm(n_traj, generator=g_traj).tolist()
            chosen = [traj_keys[j] for j in perm[:k]]

            for tk in chosen:
                new_train_indices.extend(by_template_traj[tid][tk])

            per_template_summary.append((tid, n_traj, k))

        train_indices = sorted(set(new_train_indices))

        # Log summary (avoid printing potentially huge task lists)
        total_traj_kept = sum(k for _, _, k in per_template_summary)
        total_traj_avail = sum(n for _, n, _ in per_template_summary)
        print(
            f"[Sampling][Traj] Enabled: kept k={sampling_trajectories_k} trajectories per template (seed={seed}). "
            f"Kept {total_traj_kept}/{total_traj_avail} trajectories."
        )
        for tid, n_traj, k in per_template_summary:
            print(f"[Sampling][Traj] template_id={tid}: trajectories {k}/{n_traj} (~{100.0*k/max(1,n_traj):.1f}%)")
        print(f"[Sampling][Traj] Resulting train samples (steps): {len(train_indices)}")

    if sampling_pairs:
        from collections import defaultdict

        if sampling_pairs_k <= 0:
            raise ValueError("--sampling_pairs_k must be >= 1 when pair sampling is enabled.")

        # Group current train indices by trajectory_key -> [sample_indices]
        by_traj: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i in train_indices:
            s = full_dataset.samples[i]
            traj_key = (str(s.get("task_id")), str(s.get("trajectory_num")))
            by_traj[traj_key].append(i)

        # Stable ordering of trajectories, and stable ordering of steps within each trajectory.
        traj_keys_sorted = sorted(by_traj.keys())
        for tk in traj_keys_sorted:
            by_traj[tk] = sorted(by_traj[tk], key=lambda idx: _step_num(full_dataset.samples[idx]))

        g_pairs = torch.Generator()
        g_pairs.manual_seed(int(seed) + 20011)

        chosen_indices: List[int] = []
        per_traj_counts: List[Tuple[int, int]] = []
        for tk in traj_keys_sorted:
            idxs = by_traj[tk]
            n = len(idxs)
            if n == 0:
                continue
            k = min(sampling_pairs_k, n)
            perm = torch.randperm(n, generator=g_pairs).tolist()
            chosen = [idxs[j] for j in perm[:k]]
            chosen_indices.extend(chosen)
            per_traj_counts.append((n, k))

        train_indices = sorted(set(chosen_indices))

        n_traj = len(per_traj_counts)
        avg_before = float(np.mean([n for n, _ in per_traj_counts])) if per_traj_counts else 0.0
        avg_after = float(np.mean([k for _, k in per_traj_counts])) if per_traj_counts else 0.0
        print(
            f"[Sampling][Pairs] Enabled: kept up to k={sampling_pairs_k} state-action pairs per trajectory (seed={seed}). "
            f"Trajectories: {n_traj} | avg steps {avg_before:.1f} -> {avg_after:.1f} | "
            f"Resulting train samples (steps): {len(train_indices)}"
        )
    
    print(f"Split: train={len(train_indices)}, test={len(test_indices)}")

    # -------------------------------------------------------------------------
    # Intra-template validation set (optional):
    # Uses trajectories that were NOT sampled into TRAIN but come from the same
    # task_template_id values as TRAIN templates (i.e., within-distribution).
    # Deterministic by seed.
    # -------------------------------------------------------------------------
    val_indices: List[int] = []
    if intra_evaluate:
        if not sampling_trajectories:
            raise ValueError("--intra_evaluate requires --sampling_trajectories (otherwise no unsampled trajectories remain).")
        if val_max_trajectories < 0:
            raise ValueError("--val_max_trajectories must be >= 0 (0 means no cap).")

        from collections import defaultdict

        # Rebuild grouping for ALL train-template samples (before sampling was applied).
        # We can reconstruct it by grouping the *pre-sampling* train set:
        # those are samples whose template_id is in train_template_ids.
        # Note: for split_mode=random, train_template_ids is undefined; intra_eval is meant for template split.
        if split_mode != "template":
            raise ValueError("--intra_evaluate is only supported with --split_mode template.")

        train_template_ids_str = {str(t) for t in train_template_ids} if 'train_template_ids' in locals() else set()
        # All indices belonging to train templates (pre-sampling)
        pre_sampling_train_indices = [
            i for i, s in enumerate(full_dataset.samples)
            if s.get("task_template_id") is not None and str(s.get("task_template_id")) in train_template_ids_str
        ]

        by_template_traj_all: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
        for i in pre_sampling_train_indices:
            s = full_dataset.samples[i]
            tid = s.get("task_template_id")
            if tid is None:
                continue
            traj_key = (str(s.get("task_id")), str(s.get("trajectory_num")))
            by_template_traj_all[str(tid)][traj_key].append(i)

        # Compute chosen trajectory keys per template from the *current* train_indices (post-sampling).
        chosen_by_template: Dict[str, set] = defaultdict(set)
        for i in train_indices:
            s = full_dataset.samples[i]
            tid = s.get("task_template_id")
            if tid is None:
                continue
            chosen_by_template[str(tid)].add((str(s.get("task_id")), str(s.get("trajectory_num"))))

        # For each template, remaining trajectories are those not chosen.
        g_val = torch.Generator()
        g_val.manual_seed(int(seed) + 10009)

        per_template_val_summary: List[Tuple[str, int, int, int]] = []
        for tid in sorted(by_template_traj_all.keys()):
            all_traj_keys = sorted(by_template_traj_all[tid].keys())
            chosen_keys = chosen_by_template.get(tid, set())
            remaining = [tk for tk in all_traj_keys if tk not in chosen_keys]
            n_all = len(all_traj_keys)
            n_chosen = len(chosen_keys)
            n_remaining = len(remaining)

            if n_remaining <= 0:
                per_template_val_summary.append((tid, n_all, n_chosen, 0))
                continue

            if val_max_trajectories and val_max_trajectories > 0:
                k = min(val_max_trajectories, n_remaining)
                perm = torch.randperm(n_remaining, generator=g_val).tolist()
                remaining = [remaining[j] for j in perm[:k]]
            kept = len(remaining)

            for tk in remaining:
                val_indices.extend(by_template_traj_all[tid][tk])

            per_template_val_summary.append((tid, n_all, n_chosen, kept))

        val_indices = sorted(set(val_indices))
        print(f"[Intra-Val] Enabled: built VAL from unsampled trajectories within train templates (seed={seed}).")
        if val_max_trajectories and val_max_trajectories > 0:
            print(f"[Intra-Val] Cap: val_max_trajectories={val_max_trajectories} per task_template_id")
        for tid, n_all, n_chosen, kept in per_template_val_summary:
            if kept > 0:
                print(f"[Intra-Val] template_id={tid}: total_traj={n_all} train_traj={n_chosen} val_traj={kept}")
        print(f"[Intra-Val] Resulting val samples (steps): {len(val_indices)}")

        # Update split printout
        print(f"Split (with intra): train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    intra_evaluate_active = intra_evaluate and len(val_indices) > 0
    if intra_evaluate and not intra_evaluate_active:
        print(
            "[Intra-Val][WARNING] intra_evaluate requested, but found 0 unsampled trajectories for VAL. "
            "Skipping intra/val evaluation."
        )
    
    # IMPORTANT: Do NOT re-instantiate from JSON with `indices=...` here.
    # The split indices were computed on `full_dataset.samples` (already filtered),
    # and rebuilding samples from scratch can cause index drift and accidental leakage-like splits.
    train_dataset = WebArenaTrajectorySubset(full_dataset, train_indices)
    test_dataset = WebArenaTrajectorySubset(full_dataset, test_indices)
    val_dataset = WebArenaTrajectorySubset(full_dataset, val_indices) if intra_evaluate_active else None
    
    # Analyze distributions
    analyze_action_distribution(train_dataset, "Train Set")
    analyze_action_distribution(test_dataset, "Test Set")
    if val_dataset is not None and len(val_dataset) > 0:
        analyze_action_distribution(val_dataset, "Val Set (Intra)")
    
    # Setup LoRA
    model.train()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Pre-train evaluation is handled inside `manual_train_loop(...)` so it can be
    # cleanly gated by the CLI flag `--skip_pretrain_evaluation`.
    # Train
    data_collator = default_data_collator
    
    final_accuracy, metrics_history = manual_train_loop(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        click_weight=click_weight,
        best_save_dir=best_save_dir,
        best_intra_save_dir=(best_intra_save_dir if intra_evaluate_active else None),
        debug_save_path=os.path.join(save_dir, "debug_prompts.json"),
        skip_pretrain_evaluation=skip_pretrain_evaluation,
        sync_timing=sync_timing,
        train_eval_every=train_eval_every,
        epoch_checkpoint_dir=save_dir,
    )
    
    print(f"\n[Final] Test accuracy: {final_accuracy*100:.2f}%")
    
    # Save model
    model.eval()
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save metrics to results directory
    csv_path = os.path.join(results_dir, f"metrics_{experiment_name}_{safe_data_str}.csv")
    with open(csv_path, "w", newline="") as f:
        if metrics_history:
            # Collect all unique keys from all records
            all_keys = []
            seen = set()
            for record in metrics_history:
                for k in record.keys():
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(metrics_history)
    print(f"Metrics saved to {csv_path}")
    
    print(f"Training complete! Model saved to: {save_dir}")
    
    # =========================================================================
    # Final Evaluation: Compare Final Model vs Best Model (by test loss)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL EVALUATION: Comparing Final Model vs Best Test Accuracy Model")
    print("=" * 80)
    
    # Evaluate final model
    print("\n[1/2] Evaluating FINAL model...")
    final_results = evaluate_model_full(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        max_eval_samples=len(test_dataset),
        label="FINAL",
    )
    
    # Load and evaluate best model
    print("\n[2/2] Loading and evaluating BEST checkpoints...")
    if os.path.exists(best_save_dir) and os.path.exists(os.path.join(best_save_dir, "adapter_config.json")):
        # Free the in-memory final model before loading checkpoints (reduces VRAM spikes).
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Need to reload the base model for the best checkpoint
        # First, get base model (reuse the already loaded one's base)
        base_model_for_best = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{gpu_id}"},
        )
        best_model = PeftModel.from_pretrained(base_model_for_best, best_save_dir)
        best_model.eval()
        
        # Configure generation for best model
        if hasattr(best_model, "generation_config") and best_model.generation_config is not None:
            best_model.generation_config.do_sample = False
            best_model.generation_config.temperature = 0.0
            best_model.generation_config.top_p = 1.0
            best_model.generation_config.use_cache = True
            best_model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        best_test_results = evaluate_model_full(
            model=best_model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            max_eval_samples=len(test_dataset),
            label="BEST_TEST",
        )
        
        # Print comparison
        print_model_comparison(final_results, best_test_results)
        
        # Save comparison results to JSON
        comparison_results = {
            "final_model_dir": save_dir,
            "best_test_model_dir": best_save_dir,
            "best_intra_model_dir": (best_intra_save_dir if intra_evaluate_active else None),
            "final": {
                "train_loss": final_results["train_loss"],
                "test_loss": final_results["test_loss"],
                "val_loss": final_results.get("val_loss"),
                "train_metrics": final_results["train_metrics"],
                "test_metrics": final_results["test_metrics"],
                "val_metrics": final_results.get("val_metrics"),
            },
            "best_test": {
                "train_loss": best_test_results["train_loss"],
                "test_loss": best_test_results["test_loss"],
                "val_loss": best_test_results.get("val_loss"),
                "train_metrics": best_test_results["train_metrics"],
                "test_metrics": best_test_results["test_metrics"],
                "val_metrics": best_test_results.get("val_metrics"),
            },
        }

        # Optionally evaluate best-intra checkpoint
        if (
            intra_evaluate_active
            and os.path.exists(best_intra_save_dir)
            and os.path.exists(os.path.join(best_intra_save_dir, "adapter_config.json"))
        ):
            # Cleanup best-test model before loading another checkpoint
            try:
                del best_model
                del base_model_for_best
                torch.cuda.empty_cache()
            except Exception:
                pass

            base_model_for_best_intra = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": f"cuda:{gpu_id}"},
            )
            best_intra_model = PeftModel.from_pretrained(base_model_for_best_intra, best_intra_save_dir)
            best_intra_model.eval()

            if hasattr(best_intra_model, "generation_config") and best_intra_model.generation_config is not None:
                best_intra_model.generation_config.do_sample = False
                best_intra_model.generation_config.temperature = 0.0
                best_intra_model.generation_config.top_p = 1.0
                best_intra_model.generation_config.use_cache = True
                best_intra_model.generation_config.pad_token_id = tokenizer.pad_token_id

            best_intra_results = evaluate_model_full(
                model=best_intra_model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                max_eval_samples=len(test_dataset),
                label="BEST_INTRA",
            )

            print_model_comparison(final_results, best_intra_results)
            comparison_results["best_intra"] = {
                "train_loss": best_intra_results["train_loss"],
                "test_loss": best_intra_results["test_loss"],
                "val_loss": best_intra_results.get("val_loss"),
                "train_metrics": best_intra_results["train_metrics"],
                "test_metrics": best_intra_results["test_metrics"],
                "val_metrics": best_intra_results.get("val_metrics"),
            }

            try:
                del best_intra_model
                del base_model_for_best_intra
                torch.cuda.empty_cache()
            except Exception:
                pass

        comparison_path = os.path.join(save_dir, "final_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison results saved to: {comparison_path}")
        
        # Cleanup best model
        try:
            del best_model
            del base_model_for_best
        except Exception:
            pass
    else:
        print(f"[WARNING] Best model checkpoint not found at {best_save_dir}, skipping comparison.")
        # Just print final model results
        print("\n--- Final Model Results ---")
        print(f"Train Loss: {final_results['train_loss']:.4f}")
        print(f"Test Loss:  {final_results['test_loss']:.4f}")
        print(f"Train Overall Accuracy: {final_results['train_metrics']['overall_accuracy']*100:.2f}%")
        print(f"Test Overall Accuracy:  {final_results['test_metrics']['overall_accuracy']*100:.2f}%")
        if final_results.get("val_metrics") is not None:
            print(f"Val Overall Accuracy (intra): {final_results['val_metrics']['overall_accuracy']*100:.2f}%")
    
    # Cleanup
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return final_accuracy


def run_experiment_with_multiple_seeds(
    seeds: List[int] = [42, 123, 456],
    model_path: str = MODEL_PATH,
    data: str = "vanilla",
    output_dir: str = OUTPUT_DIR,
    num_epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 8192,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.005,
    click_weight: float = 0.0,
    gpu_id: int = 2,
    sampling_trajectories: bool = False,
    sampling_trajectories_k: int = 10,
    sampling_pairs: bool = False,
    sampling_pairs_k: int = 50,
):
    """
    Run training with multiple seeds and report aggregate results.
    """
    data_path = get_data_path(data)
    accuracies = []
    
    print("=" * 60)
    print(f"Running experiment with {len(seeds)} seeds: {seeds}")
    print("=" * 60)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{len(seeds)} with seed {seed}")
        print(f"{'='*60}\n")
        
        try:
            accuracy = train_llama_il(
                model_path=model_path,
                data_path=data_path,
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_length=max_length,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                click_weight=click_weight,
                gpu_id=gpu_id,
                seed=seed,
                data=data,
                sampling_trajectories=sampling_trajectories,
                sampling_trajectories_k=sampling_trajectories_k,
                sampling_pairs=sampling_pairs,
                sampling_pairs_k=sampling_pairs_k,
            )
            accuracies.append(accuracy)
            print(f"\n[Run {i}] Accuracy: {accuracy*100:.2f}%")
        except Exception as e:
            import traceback
            print(f"\n[Run {i}] Error: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
    
    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Dataset: {data}")
        print(f"Successful runs: {len(accuracies)}/{len(seeds)}")
        print(f"Accuracies: {[f'{a*100:.2f}%' for a in accuracies]}")
        print(f"Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print("=" * 60)
        
        return mean_acc, std_acc, accuracies
    
    return None, None, []


def main(data: str = "vanilla"):
    """Main entry point."""
    run_experiment_with_multiple_seeds(
        seeds=[42],
        model_path=MODEL_PATH,
        data=data,
        output_dir=OUTPUT_DIR,
        num_epochs=5,
        batch_size=2,
        learning_rate=1e-4,
        gradient_accumulation_steps=4,
        max_length=1024,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.005,
        click_weight=0.0,
        gpu_id=2,
        sampling=False,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LLaMA with LoRA for WebArena IL")
    parser.add_argument("--model", type=str, default="1B", choices=["1B", "8B"],
                        help="Model size: '1B' (Llama-3.2-1B) or '8B' (Llama-3.1-8B)")
    parser.add_argument("--data", type=str, default="vanilla", choices=["vanilla", "safe"],
                        help="Data type to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length (tokens). Training examples are left-padded and (unless --error_on_truncation) truncated from the left.",
    )
    parser.add_argument("--gpu", type=int, default=2, help="GPU ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Sampling controls (train split only)
    # Backwards-compatible alias: --sampling / --sampling_k are trajectory sampling.
    parser.add_argument(
        "--sampling",
        "--sampling_trajectories",
        dest="sampling_trajectories",
        action="store_true",
        help="Subsample TRAIN split by keeping k trajectories per task_template_id (keeps all steps in chosen trajectories).",
    )
    parser.add_argument(
        "--sampling_k",
        "--sampling_trajectories_k",
        dest="sampling_trajectories_k",
        type=int,
        default=5,
        help="When --sampling/--sampling_trajectories is set: keep k trajectories per task_template_id (train split only).",
    )
    parser.add_argument(
        "--sampling_pairs",
        action="store_true",
        help="Subsample TRAIN split within each trajectory by keeping k state-action pairs (steps) per trajectory.",
    )
    parser.add_argument(
        "--sampling_pairs_k",
        type=int,
        default=50,
        help="When --sampling_pairs is set: keep at most k state-action pairs (steps) per trajectory (train split only).",
    )
    parser.add_argument(
        "--intra_evaluate",
        action="store_true",
        help=(
            "If set: evaluate an intra-template VAL set built from UNSAMPLED trajectories within the TRAIN templates. "
            "Requires --sampling_trajectories."
        ),
    )
    parser.add_argument(
        "--val_max_trajectories",
        type=int,
        default=0,
        help=(
            "When --intra_evaluate is set: cap the number of VAL trajectories PER task_template_id. "
            "0 means no cap (use all unsampled trajectories)."
        ),
    )
    parser.add_argument("--split_mode", type=str, default="template", choices=["template", "random"],
                        help="Split mode: 'template' (OOD by task type) or 'random' (ID 80/20)")
    parser.add_argument(
        "--test_template_ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "If provided (and --split_mode template): use these task_template_id values as the FIXED test set templates "
            "(train templates are all remaining templates). If omitted: pick held-out templates randomly but deterministically via --seed."
        ),
    )
    parser.add_argument("--ignore_templates", type=int, help="template ids to ignore, separated by ,",
                        default=[2002, 2007], nargs="+")
    parser.add_argument("--skip_pretrain_evaluation", action="store_true", help="Skip pretrain evaluation")
    parser.add_argument(
        "--sync_timing",
        action="store_true",
        help="Synchronize CUDA for more accurate timing logs (slower). Default: async approximate timing.",
    )
    parser.add_argument(
        "--train_eval_every",
        type=int,
        default=3,
        help="Evaluate/print TRAIN accuracy every N epochs (in addition to final epoch). Set 0 to disable periodic.",
    )
    parser.add_argument(
        "--max_history_actions",
        type=int,
        default=15,
        help="Max number of previous actions to include in the prompt context (most recent).",
    )
    parser.add_argument(
        "--error_on_truncation",
        action="store_true",
        help="If set: raise an error when a sample exceeds --max_length instead of truncating from the left.",
    )
    parser.add_argument(
        "--policy_aware",
        action="store_true",
        help="If set, require --data safe and inject the dataset safety policy into the system prompt.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name for the experiment. Used for organizing checkpoints and results.",
    )

    
    args = parser.parse_args()
    
    train_llama_il(
        model_path=get_model_path(args.model),
        data_path=get_data_path(args.data),
        output_dir=OUTPUT_DIR,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_length=args.max_length,
        gpu_id=args.gpu,
        seed=args.seed,
        data=args.data,
        sampling_trajectories=args.sampling_trajectories,
        sampling_trajectories_k=args.sampling_trajectories_k,
        sampling_pairs=args.sampling_pairs,
        sampling_pairs_k=args.sampling_pairs_k,
        intra_evaluate=args.intra_evaluate,
        val_max_trajectories=args.val_max_trajectories,
        split_mode=args.split_mode,
        test_template_ids=args.test_template_ids,
        ignore_templates=args.ignore_templates,
        skip_pretrain_evaluation=args.skip_pretrain_evaluation,
        sync_timing=args.sync_timing,
        train_eval_every=args.train_eval_every,
        max_history_actions=args.max_history_actions,
        error_on_truncation=args.error_on_truncation,
        policy_aware=args.policy_aware,
        experiment_name=args.experiment_name,
    )

