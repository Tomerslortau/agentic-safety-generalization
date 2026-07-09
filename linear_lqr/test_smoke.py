"""Smoke tests for the united linear_lqr part (CPU, seconds).

Covers the shared solver core, the imitation data-gen + training pipeline, and the
Lipschitz-ratio experiment. Run: `python test_smoke.py` (no pytest needed).
"""

import os
import sys
import shutil
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
sys.path.insert(0, HERE)                                   # core package
sys.path.insert(0, os.path.join(HERE, 'imitation_experiment'))
sys.path.insert(0, os.path.join(HERE, 'lipschitz_experiment'))

from core.solvers import solve_lqr, solve_dgare, find_hinf_gain
from model import ResidualMLP
from generate_database import generate
from train_policy import load_dataset, train_model, normalize_mse, run_job, DEFAULTS

import config as lip_config
from experiments import run_experiment


SMOKE_DATA = os.path.join(HERE, 'data_smoke')


def test_solve_lqr():
    rng = np.random.RandomState(0)
    n = 3
    A, B = rng.randn(n, n) * 0.3, rng.randn(n, n) * 0.3
    P, K = solve_lqr(A, B, np.eye(n), np.eye(n))
    assert np.all(np.linalg.eigvalsh(P) >= -1e-10)
    assert np.max(np.abs(np.linalg.eigvals(A + B @ K))) < 1.0
    print("PASS: test_solve_lqr")


def test_solve_dgare():
    rng = np.random.RandomState(0)
    n = 3
    A, B = rng.randn(n, n) * 0.3, rng.randn(n, n) * 0.3
    ok, P, K = solve_dgare(A, B, np.eye(n), np.eye(n), np.eye(n), gamma=5.0)
    assert ok, "DGARE failed at gamma=5.0"
    assert np.all(np.linalg.eigvalsh(P) >= -1e-10)
    assert np.max(np.abs(np.linalg.eigvals(A + B @ K))) < 1.0
    print("PASS: test_solve_dgare")


def test_find_hinf_gain():
    rng = np.random.RandomState(0)
    n = 3
    A, B = rng.randn(n, n) * 0.3, rng.randn(n, n) * 0.3
    K = find_hinf_gain(A, B, np.eye(n), np.eye(n), np.eye(n), gamma_cap=10.0)
    assert K.shape == (n, n)
    assert np.max(np.abs(np.linalg.eigvals(A + B @ K))) < 1.0
    print("PASS: test_find_hinf_gain")


def test_hinf_differs_from_lqr():
    rng = np.random.RandomState(0)
    n = 3
    A, B = rng.randn(n, n) * 0.3, rng.randn(n, n) * 0.3
    _, K_lqr = solve_lqr(A, B, np.eye(n), np.eye(n))
    K_hinf = find_hinf_gain(A, B, np.eye(n), np.eye(n), np.eye(n), gamma_cap=10.0)
    assert not np.allclose(K_lqr, K_hinf, atol=1e-6)
    print("PASS: test_hinf_differs_from_lqr")


def test_model_forward():
    n = 5
    model = ResidualMLP(n * n + n, n, hidden_dim=64, num_layers=3)
    x = torch.randn(8, n * n + n)
    y = model(x)
    assert y.shape == (8, n)
    assert torch.all(torch.isfinite(y))
    print("PASS: test_model_forward")


def test_generate_database():
    n = 3
    shutil.rmtree(SMOKE_DATA, ignore_errors=True)
    generate(seed=42, n=n, data_root=SMOKE_DATA, systems_per_n2=5)
    data_dir = os.path.join(SMOKE_DATA, 'seed_42', f'n{n}')
    assert os.path.exists(os.path.join(data_dir, 'system.npz'))
    assert os.path.exists(os.path.join(data_dir, 'tasks.npz'))
    assert os.path.exists(os.path.join(data_dir, 'meta.json'))
    tasks = np.load(os.path.join(data_dir, 'tasks.npz'))
    assert tasks['Q'].shape[1] == n * n
    assert tasks['K_lqr'].shape[1:] == (n, n)
    assert tasks['K_hinf'].shape[1:] == (n, n)
    assert len(tasks['Q']) > 0
    print(f"PASS: test_generate_database ({len(tasks['Q'])} tasks)")


def test_load_dataset():
    n = 3
    data_dir = os.path.join(SMOKE_DATA, 'seed_42', f'n{n}')
    train, intask, crosstask = load_dataset(
        data_dir, n, train_states=4, intask_test_states=2,
        train_frac=0.8, state_seed=42)
    for name, data in [('train', train), ('intask', intask), ('crosstask', crosstask)]:
        Q, x, u_lqr, u_hinf = data
        assert Q.shape[1] == n * n, f"{name} Q shape wrong"
        assert x.shape[1] == n, f"{name} x shape wrong"
        assert u_lqr.shape[1] == n, f"{name} u_lqr shape wrong"
        assert u_hinf.shape[1] == n, f"{name} u_hinf shape wrong"
        assert len(Q) > 0, f"{name} empty"
    assert not np.allclose(train[2], train[3]), "LQR and H-inf actions identical"
    print(f"PASS: test_load_dataset (train={len(train[0])}, intask={len(intask[0])}, cross={len(crosstask[0])})")


def test_train_pipeline():
    n = 3
    cfg = dict(DEFAULTS, hidden_dim=32, num_layers=2,
               max_epochs=50, eval_every=10, patience=5, batch_size=16)
    data_dir = os.path.join(SMOKE_DATA, 'seed_42', f'n{n}')
    train, intask, crosstask = load_dataset(
        data_dir, n, train_states=4, intask_test_states=2,
        train_frac=0.8, state_seed=42)
    tr = (train[0], train[1], train[2])
    it = (intask[0], intask[1], intask[2])
    ct = (crosstask[0], crosstask[1], crosstask[2])
    device = torch.device('cpu')
    train_mse, intask_mse, crosstask_mse = train_model(tr, it, ct, n, cfg, device, tag="smoke")
    assert all(np.isfinite(v) and v >= 0 for v in [train_mse, intask_mse, crosstask_mse])
    print(f"PASS: test_train_pipeline (train={train_mse:.4e}, intask={intask_mse:.4e}, cross={crosstask_mse:.4e})")


def test_lipschitz_ratio():
    """Tiny end-to-end Lipschitz-ratio run: finite ratio on a single grid point."""
    orig_num_Q = lip_config.num_Q_for_dim
    lip_config.num_Q_for_dim = lambda n: 15  # shrink for speed
    try:
        cfg = lip_config.select_config('general')
        cfg.n_seeds = 1
        cfg.NORM_A_LIST = [0.2]
        cfg.NORM_B_LIST = [0.5]
        cfg.MIN_EIG_R_LIST = [1]
        cfg.NORM_D_LIST = [1]
        cfg.verbose_outer = False
        df = run_experiment(2, cfg)
    finally:
        lip_config.num_Q_for_dim = orig_num_Q
    assert len(df) == 1
    row = df.iloc[0]
    assert np.isfinite(row['L_lqr_emp']) and np.isfinite(row['L_hinf_emp'])
    ratio = row['L_hinf_emp'] / row['L_lqr_emp']
    assert np.isfinite(ratio) and ratio > 0
    print(f"PASS: test_lipschitz_ratio (L_hinf/L_lqr = {ratio:.3f})")


def test_run_job_gpu():
    if not torch.cuda.is_available():
        print("SKIP: test_run_job_gpu (no GPU)")
        return
    n = 3
    cfg = dict(DEFAULTS, hidden_dim=32, num_layers=2,
               max_epochs=50, eval_every=10, patience=5, batch_size=16,
               data_root=SMOKE_DATA, train_states_per_task=4,
               intask_test_states_per_task=2)
    result = run_job(n, seed=42, cfg=cfg, gpu_id=0)
    for key in ['lqr_train', 'lqr_intask', 'lqr_crosstask',
                'hinf_train', 'hinf_intask', 'hinf_crosstask']:
        assert key in result, f"Missing key: {key}"
        assert np.isfinite(result[key])
    print("PASS: test_run_job_gpu")


if __name__ == '__main__':
    test_solve_lqr()
    test_solve_dgare()
    test_find_hinf_gain()
    test_hinf_differs_from_lqr()
    test_model_forward()
    test_generate_database()
    test_load_dataset()
    test_train_pipeline()
    test_lipschitz_ratio()
    test_run_job_gpu()

    shutil.rmtree(SMOKE_DATA, ignore_errors=True)
    print("\nAll smoke tests passed.")
