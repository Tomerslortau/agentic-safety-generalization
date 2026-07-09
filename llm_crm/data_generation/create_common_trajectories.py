#!/usr/bin/env python3
"""
Create common trajectory files for tasks that pass quality checks in BOTH vanilla and safe.
Only tasks that pass ALL criteria in BOTH modes are included.

Criteria:
1. Reward is 1.0
2. Trajectory length >= 4
3. (Safe only) completion_under_policy is True
4. Trajectory length is not very different from other tasks in the same template
5. Only keep templates with at least MIN_GOOD_PER_TEMPLATE tasks that pass ALL criteria in BOTH modes

Outputs:
- training_data/vanilla_trajectories_common.json
- training_data/safe_trajectories_common.json
(Both files contain the SAME set of task IDs)
"""

import json
import copy
from pathlib import Path
from collections import defaultdict
import statistics


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def get_task_id_from_key(key: str) -> int:
    """Extract task_id from key like 'browsergym/STWebAgentBenchEnv.20000'"""
    return int(key.split('.')[-1])


def get_best_trajectory(trajectories: dict, is_safe: bool) -> tuple:
    """
    Find the best (latest) trajectory that meets criteria.
    Returns (traj_id, traj_data, meta, rejection_reason) or (None, None, None, reason) if none qualify.
    """
    if not trajectories:
        return None, None, None, "no_trajectories"

    for traj_id in sorted(trajectories.keys(), key=lambda x: int(x), reverse=True):
        traj = trajectories[traj_id]
        meta = traj.get('meta_data') or traj.get('meta') or {}

        reward = meta.get('reward')
        length = meta.get('length')

        # Check reward
        if reward != 1.0:
            continue

        # Check length >= 4
        if length is None or length < 4:
            continue

        # For safe, check completion_under_policy
        if is_safe:
            completion_under_policy = meta.get('completion_under_policy')
            if not completion_under_policy:
                continue

        # This trajectory passes basic criteria
        return traj_id, traj, meta, None

    # Determine why all trajectories failed - check best trajectory to give a reason
    best_reward = -1
    best_reason = "no_qualifying_trajectory"

    for traj_id in trajectories.keys():
        traj = trajectories[traj_id]
        meta = traj.get('meta_data') or traj.get('meta') or {}
        reward = meta.get('reward', 0)

        if reward > best_reward:
            best_reward = reward
            if reward != 1.0:
                best_reason = "reward_not_1"
            elif meta.get('length', 0) < 4:
                best_reason = "length_less_than_4"
            elif is_safe and not meta.get('completion_under_policy'):
                best_reason = "completion_under_policy_false"

    return None, None, None, best_reason


def compute_template_stats(data: dict, task_to_template: dict, is_safe: bool) -> dict:
    """Compute mean and stdev of trajectory lengths per template."""
    template_lengths = defaultdict(list)

    for key, entry in data.items():
        if not key.startswith('browsergym/STWebAgentBenchEnv.'):
            continue

        tid = get_task_id_from_key(key)
        tmpl = task_to_template.get(tid)
        if not tmpl:
            continue

        trajectories = entry.get('trajectories', {})
        traj_id, traj, meta, _ = get_best_trajectory(trajectories, is_safe)

        if traj_id is not None:
            length = meta.get('length')
            if length:
                template_lengths[tmpl].append(length)

    # Compute stats
    stats = {}
    for tmpl, lengths in template_lengths.items():
        if len(lengths) >= 2:
            mean = statistics.mean(lengths)
            stdev = statistics.stdev(lengths)
        elif len(lengths) == 1:
            mean = lengths[0]
            stdev = 0
        else:
            mean = 0
            stdev = 0
        stats[tmpl] = {'mean': mean, 'stdev': stdev, 'lengths': lengths}

    return stats


def is_length_acceptable(length: int, template_stats: dict, tmpl: str, max_deviation: float = 2.0) -> bool:
    """Check if length is within acceptable range for the template."""
    if tmpl not in template_stats:
        return True  # No stats, accept by default

    stats = template_stats[tmpl]
    mean = stats['mean']
    stdev = stats['stdev']

    if stdev == 0:
        # All same length or single task, use a fixed tolerance
        return abs(length - mean) <= 3

    # Accept if within max_deviation standard deviations
    return abs(length - mean) <= max_deviation * stdev


def main():
    MIN_GOOD_PER_TEMPLATE = 9

    base = Path(__file__).parent

    # Load task definitions to get template mapping
    # Load both task definition files (20000 and 30000 series)
    tasks_path_20000 = base / 'new_data_prob40.json'
    tasks_path_30000 = base / 'new_data_prob40_30000.json'
    
    tasks = []
    if tasks_path_20000.exists():
        tasks_20000 = load_json(tasks_path_20000)
        tasks.extend(tasks_20000)
        print(f"Loaded {len(tasks_20000)} tasks from {tasks_path_20000.name}")
    if tasks_path_30000.exists():
        tasks_30000 = load_json(tasks_path_30000)
        tasks.extend(tasks_30000)
        print(f"Loaded {len(tasks_30000)} tasks from {tasks_path_30000.name}")
    
    task_to_template = {t['task_id']: t['template_id'] for t in tasks}
    all_task_ids = sorted({t['task_id'] for t in tasks})
    print(f"Total unique task IDs: {len(all_task_ids)}")

    # Load trajectory files
    vanilla_path = base / 'training_data' / 'vanilla_trajectories_new.json'
    safe_path = base / 'training_data' / 'safe_trajectories_new.json'

    vanilla_data = load_json(vanilla_path)
    safe_data = load_json(safe_path)

    # Compute template stats for length deviation checks
    print("Computing template statistics...")
    vanilla_stats = compute_template_stats(vanilla_data, task_to_template, is_safe=False)
    safe_stats = compute_template_stats(safe_data, task_to_template, is_safe=True)

    # First pass: find tasks that pass BOTH vanilla and safe, but do not write yet
    passed = {}  # tid -> record with tmpl/key and chosen trajectories
    failed_tasks = defaultdict(list)

    for tid in all_task_ids:
        key = f'browsergym/STWebAgentBenchEnv.{tid}'
        tmpl = task_to_template.get(tid)

        # === Check VANILLA ===
        vanilla_entry = vanilla_data.get(key)
        if not vanilla_entry:
            failed_tasks['vanilla: missing_in_file'].append(tid)
            continue

        v_trajectories = vanilla_entry.get('trajectories', {})
        v_traj_id, v_traj, v_meta, v_reason = get_best_trajectory(v_trajectories, is_safe=False)

        if v_traj_id is None:
            failed_tasks[f'vanilla: {v_reason}'].append(tid)
            continue

        v_length = v_meta.get('length')
        if not is_length_acceptable(v_length, vanilla_stats, tmpl, max_deviation=5.0):
            failed_tasks['vanilla: length_outlier'].append(tid)
            continue

        # === Check SAFE ===
        safe_entry = safe_data.get(key)
        if not safe_entry:
            failed_tasks['safe: missing_in_file'].append(tid)
            continue

        s_trajectories = safe_entry.get('trajectories', {})
        s_traj_id, s_traj, s_meta, s_reason = get_best_trajectory(s_trajectories, is_safe=True)

        if s_traj_id is None:
            failed_tasks[f'safe: {s_reason}'].append(tid)
            continue

        s_length = s_meta.get('length')
        if not is_length_acceptable(s_length, safe_stats, tmpl, max_deviation=5.0):
            failed_tasks['safe: length_outlier'].append(tid)
            continue

        # === BOTH passed - record selection ===
        passed[tid] = {
            "tmpl": tmpl,
            "key": key,
            "vanilla": {"entry": vanilla_entry, "traj_id": v_traj_id, "traj": v_traj},
            "safe": {"entry": safe_entry, "traj_id": s_traj_id, "traj": s_traj},
        }

    # Second pass: keep only templates with at least MIN_GOOD_PER_TEMPLATE tasks
    tmpl_counts = defaultdict(int)
    for tid, rec in passed.items():
        tmpl_counts[rec["tmpl"]] += 1

    good_templates = {tmpl for tmpl, c in tmpl_counts.items() if c >= MIN_GOOD_PER_TEMPLATE}

    # Tasks kept after template filter
    kept_tids = [tid for tid, rec in passed.items() if rec["tmpl"] in good_templates]
    kept_tids.sort()

    # Build final common outputs
    vanilla_common = {}
    safe_common = {}

    for tid in kept_tids:
        rec = passed[tid]
        key = rec["key"]
        tmpl = rec["tmpl"]

        vanilla_entry = rec["vanilla"]["entry"]
        v_traj_id = rec["vanilla"]["traj_id"]
        
        safe_entry = rec["safe"]["entry"]
        s_traj_id = rec["safe"]["traj_id"]
        
        # Make deep copies to avoid modifying original trajectories
        v_traj = copy.deepcopy(rec["vanilla"]["traj"])
        s_traj = copy.deepcopy(rec["safe"]["traj"])
        
        # Add task_template_id to vanilla trajectory meta_data
        if 'meta_data' not in v_traj:
            v_traj['meta_data'] = {}
        elif v_traj['meta_data'] is None:
            v_traj['meta_data'] = {}
        v_traj['meta_data']['task_template_id'] = tmpl
        
        # Add task_template_id to safe trajectory meta_data
        if 'meta_data' not in s_traj:
            s_traj['meta_data'] = {}
        elif s_traj['meta_data'] is None:
            s_traj['meta_data'] = {}
        s_traj['meta_data']['task_template_id'] = tmpl

        vanilla_common[key] = {
            'policy': vanilla_entry.get('policy'),
            'trajectories': {v_traj_id: v_traj}
        }

        safe_common[key] = {
            'policy': safe_entry.get('policy'),
            'trajectories': {s_traj_id: s_traj}
        }

    # Write output files
    vanilla_out_path = base / 'training_data' / 'vanilla_trajectories_common.json'
    safe_out_path = base / 'training_data' / 'safe_trajectories_common.json'

    vanilla_out_path.write_text(json.dumps(vanilla_common, indent=2), encoding='utf-8')
    safe_out_path.write_text(json.dumps(safe_common, indent=2), encoding='utf-8')

    # Count unique templates in final output
    templates_in_output = {rec["tmpl"] for tid, rec in passed.items() if tid in kept_tids}
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMMON TRAJECTORIES (tasks passing BOTH vanilla AND safe)")
    print(f"{'='*60}")
    print(f"Total tasks: {len(all_task_ids)}")
    print(f"Tasks passing all criteria in BOTH (before template filter): {len(passed)}")
    print(f"Templates with >= {MIN_GOOD_PER_TEMPLATE} passing tasks: {len(good_templates)}")
    print(f"Templates in common trajectories: {len(templates_in_output)}")
    print(f"Tasks kept after template filter: {len(kept_tids)}")
    print(f"\nKept task IDs ({len(kept_tids)}):")
    print(f"  {kept_tids}")

    # Optional: show which templates were dropped due to insufficient passing tasks
    dropped_templates = {tmpl for tmpl in tmpl_counts.keys() if tmpl not in good_templates}
    if dropped_templates:
        dropped_sorted = sorted(dropped_templates, key=lambda t: tmpl_counts[t])
        print(f"\nTemplates dropped (< {MIN_GOOD_PER_TEMPLATE} passing tasks): {len(dropped_sorted)}")
        for tmpl in dropped_sorted:
            print(f"  - template {tmpl}: {tmpl_counts[tmpl]} passing tasks")

    print(f"\n{'='*60}")
    print("FAILURE REASONS (why task was NOT added)")
    print(f"{'='*60}")

    for reason in sorted(failed_tasks.keys()):
        tids = failed_tasks[reason]
        print(f"\n{reason}: {len(tids)} tasks")
        for tid in sorted(tids):
            print(f"  - {tid}")

    print(f"\n{'='*60}")
    print(f"OUTPUT FILES (same {len(kept_tids)} tasks in both)")
    print(f"{'='*60}")
    print(f"  {vanilla_out_path}")
    print(f"  {safe_out_path}")


if __name__ == '__main__':
    main()
