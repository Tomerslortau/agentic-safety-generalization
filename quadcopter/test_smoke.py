"""Smoke test for the quadcopter part (CPU, minutes).

Exercises the REAL building blocks — the differentiable environment, the teacher
`train()` unrolling loop (both an unsafe/vanilla env and a safe/boxed env), and the
student `train_imitation_model` — on tiny inputs. It bypasses `setup_experiment`
(which needs a pre-generated reference-controller directory) by constructing the
environment directly with a couple of target states. Run: `python test_smoke.py`.
"""

import os
import sys
import tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
sys.path.insert(0, HERE)

from environment import DifferentiableQuadcopterEnv
from models import MLPController
from training import train
from workflow import add_no_fly_boxes
from imitation_learning_experiment import train_imitation_model

DEVICE = torch.device('cpu')


def _targets():
    """Two reachable target positions at altitude z=4 (state layout [pos, rpy, vel, rpy_rates])."""
    t = torch.zeros(2, 12)
    t[0, :3] = torch.tensor([3.0, 0.0, 4.0])
    t[1, :3] = torch.tensor([-3.0, 2.0, 4.0])
    return t


def _controller():
    return MLPController(input_dim=15, hidden_size=64, output_dim=4, num_hidden_layers=2).to(DEVICE)


def test_environment_unroll_differentiable():
    env = DifferentiableQuadcopterEnv(target_states=_targets(), horizon=12).to(DEVICE)
    ctrl = _controller()
    init = torch.zeros(2, 12, device=DEVICE)
    loss = env.forward(controller_fn=lambda o: ctrl(o), init_state=init, fixed=False)
    assert torch.isfinite(loss), "env forward produced a non-finite loss"
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in ctrl.parameters()), \
        "no finite gradients flowed back through the differentiable sim"
    print(f"PASS: test_environment_unroll_differentiable (loss={loss.item():.4f})")


def _train_tiny(env, tag):
    ctrl = _controller()
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-3)
    test_env = DifferentiableQuadcopterEnv(target_states=_targets()[:1], horizon=12).to(DEVICE)
    with tempfile.TemporaryDirectory() as ckpt:
        metrics, best_state = train(
            train_env=env, controller=ctrl, optimizer=opt, device=DEVICE,
            adv=False, fixed=False, num_epochs=3,
            early_stop_patience=100, lr_reduce_patience=100, lr_reduce_cooldown=0,
            early_stop_eps=1e-6, lr_reduce_eps=1e-6, lr_reduce_factor=0.5,
            print_period=1, checkpoint_period=1000, checkpoint_dir=ckpt,
            test_env=test_env, controller_name_override=tag)
    assert best_state is not None, f"{tag}: no best state returned"
    assert np.all(np.isfinite(metrics['train_losses'])), f"{tag}: non-finite train losses"
    return metrics


def test_teacher_training_unsafe():
    env = DifferentiableQuadcopterEnv(target_states=_targets(), horizon=12).to(DEVICE)
    m = _train_tiny(env, "vanilla")
    print(f"PASS: test_teacher_training_unsafe (final train loss={m['train_losses'][-1]:.4f})")


def test_teacher_training_safe():
    env = DifferentiableQuadcopterEnv(target_states=_targets(), horizon=12).to(DEVICE)
    add_no_fly_boxes(env)   # the safe teacher trains against no-fly-box penalties
    m = _train_tiny(env, "robust")
    print(f"PASS: test_teacher_training_safe (final train loss={m['train_losses'][-1]:.4f})")


def test_student_imitation():
    N = 32
    inputs = torch.randn(N, 15)
    actions = torch.tanh(torch.randn(N, 4))
    train_ds = (inputs, actions)
    val_ds = (inputs[:8], actions[:8])
    model, tr, va, lrs = train_imitation_model(train_ds, val_ds, "smoke", DEVICE, num_epochs=5)
    assert model is not None
    assert np.all(np.isfinite(tr)) and np.all(np.isfinite(va))
    print(f"PASS: test_student_imitation (final train={tr[-1]:.4f}, val={va[-1]:.4f})")


if __name__ == '__main__':
    test_environment_unroll_differentiable()
    test_teacher_training_unsafe()
    test_teacher_training_safe()
    test_student_imitation()
    print("\nAll quadcopter smoke tests passed.")
