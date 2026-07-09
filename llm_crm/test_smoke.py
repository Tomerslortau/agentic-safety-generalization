"""Offline smoke test for the LLM / CRM part (no network, no weights, no live stack).

Exercises the code paths that do NOT need the external stack:
  1. per-seed metrics aggregation into the paper-style table,
  2. trajectory -> (prompt, target-action) construction,
  3. exact-match accuracy scoring.

The prompt/target step uses `train_llama_il.build_prompt_and_target` when its heavy
dependencies (torch/transformers/peft) are importable, and a small built-in fallback
otherwise, so this test runs anywhere. Run: `python test_smoke.py`.
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
sys.path.insert(0, os.path.join(HERE, 'training'))
SAMPLE = os.path.join(HERE, 'data_sample')
VANILLA = os.path.join(SAMPLE, 'vanilla_trajectories_common.json')
SAFE = os.path.join(SAMPLE, 'safe_trajectories_common.json')


def exact_match(pred, target):
    return 1.0 if pred.strip() == target.strip() else 0.0


def _iter_steps(path):
    """Yield (goal, state, action, action_history) per step, mirroring the dataset."""
    with open(path) as f:
        raw = json.load(f)
    for task_id, task in raw.items():
        for tnum, traj in task.get('trajectories', {}).items():
            goal = traj.get('meta_data', {}).get('goal', '')
            data = traj.get('data', {})
            step_keys = sorted([k for k in data if k.startswith('step_')],
                               key=lambda x: int(x.split('_')[1]))
            history = []
            for sk in step_keys:
                step = data[sk]
                state = step.get('state', {})
                action = step.get('action', '')
                if not action:
                    continue
                yield goal, state, action, list(history)
                history.append(action)


def test_results_aggregation():
    from create_experiments_results_per_seed import main as agg_main
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'out.csv')
        rows = agg_main(results_dir=os.path.join(SAMPLE, 'sample_results'),
                        experiment_name='7_trajectories_train', epoch=19, out_path=out)
    assert len(rows) == 4, f"expected 4 rows, got {len(rows)}"
    by = {(r['seed'], r['model_type']): r for r in rows}
    for seed in (0, 1):
        safe_test = by[(seed, 'safe')]['test_accuracy']
        vanilla_test = by[(seed, 'vanilla')]['test_accuracy']
        assert safe_test < vanilla_test, \
            f"seed {seed}: safe test-acc {safe_test} should be < vanilla {vanilla_test}"
    print("PASS: test_results_aggregation (safe test-acc < vanilla, as in the paper)")


def test_trajectory_prompt_target():
    try:
        from train_llama_il import build_prompt_and_target
        builder = "train_llama_il.build_prompt_and_target"
    except Exception as e:  # torch/transformers/peft not installed
        build_prompt_and_target = None
        builder = f"built-in fallback ({type(e).__name__})"

    def build(goal, state, action, history):
        if build_prompt_and_target is not None:
            return build_prompt_and_target(goal, state, action, action_history=history)
        prompt = (f"# Goal\n{goal}\n\n# Previous Actions\n" +
                  ("\n".join(history) if history else "(none)") +
                  f"\n\n# Current Page (Accessibility Tree)\n{state.get('axtree_txt','')}\n\n# Action")
        return prompt, action.strip()

    n = 0
    for path in (VANILLA, SAFE):
        for goal, state, action, history in _iter_steps(path):
            prompt, target = build(goal, state, action, history)
            assert goal in prompt, "goal missing from prompt"
            assert target == action.strip(), "target should be the raw action string"
            assert exact_match(target, action) == 1.0
            n += 1
    assert n > 0, "no steps parsed"
    print(f"PASS: test_trajectory_prompt_target ({n} steps via {builder})")


def test_exact_match_scoring():
    assert exact_match('click("30")', 'click("30")') == 1.0
    assert exact_match('click("30")', 'click("31")') == 0.0
    # A perfect student on the sample scores 1.0 overall.
    preds = targets = [a for _, _, a, _ in _iter_steps(VANILLA)]
    acc = sum(exact_match(p, t) for p, t in zip(preds, targets)) / len(targets)
    assert acc == 1.0
    print(f"PASS: test_exact_match_scoring (self-match accuracy = {acc:.2f})")


def test_safe_teacher_adds_warnings():
    safe_actions = [a for _, _, a, _ in _iter_steps(SAFE)]
    vanilla_actions = [a for _, _, a, _ in _iter_steps(VANILLA)]
    assert any('send_msg_to_user' in a for a in safe_actions), \
        "safe teacher should warn before saving PII"
    assert len(safe_actions) > len(vanilla_actions), \
        "safe trajectories should have more steps (extra warnings)"
    print("PASS: test_safe_teacher_adds_warnings")


if __name__ == '__main__':
    test_results_aggregation()
    test_trajectory_prompt_target()
    test_exact_match_scoring()
    test_safe_teacher_adds_warnings()
    print("\nAll offline smoke tests passed.")
