import random

import numpy as np
import torch
try:
    import GPUtil
    _HAVE_GPUTIL = True
except Exception:
    GPUtil = None
    _HAVE_GPUTIL = False

from config import system_param_intervals, initial_state_intervals


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_available_device(
    max_load=0.5,
    max_memory=0.5,
    device_index=0
):
    # CPU when there is no CUDA at all.
    if not torch.cuda.is_available():
        return torch.device("cpu")
    # Without GPUtil, just pick a valid CUDA index.
    if not _HAVE_GPUTIL:
        idx = min(max(device_index, 0), torch.cuda.device_count() - 1)
        return torch.device(f'cuda:{idx}')
    # Try to get a GPU that is under the load and memory thresholds.
    available_gpus = GPUtil.getAvailable(
        order='first',
        maxLoad=max_load,
        maxMemory=max_memory,
        limit=torch.cuda.device_count()
    )
    if available_gpus:
        device_index = min([len(available_gpus)-1, device_index])
        return torch.device(f'cuda:{available_gpus[device_index]}')
    else:
        return torch.device("cpu")
    

def draw_goal_sample(device):
    # Sample a 12D goal state (pos, rpy, vel, rpy_rates)
    goal_state = []
    for key in ['pos', 'rpy', 'vel', 'rpy_rates']:
        for low, high in system_param_intervals[key]:
            goal_state.append(np.random.uniform(low, high))
    goal_state = torch.tensor(goal_state, device=device)
    return goal_state


def sample_initial_state(device):
    # Sample a 12D initial state (pos, rpy, vel, rpy_rates) using config intervals
    init_state = []
    for key in ['pos', 'rpy', 'vel', 'rpy_rates']:
        for low, high in initial_state_intervals[key]:
            init_state.append(np.random.uniform(low, high))
    return torch.tensor(init_state, device=device, dtype=torch.float32)


def verify_no_gradients(model, test_func, *args, **kwargs):
    """
    Verify that no gradients are computed during test function execution.
    
    Args:
        model: The model to check
        test_func: Function to test
        *args, **kwargs: Arguments for test_func
    
    Returns:
        bool: True if no gradients were computed
    """
    # Clear any existing gradients
    model.zero_grad()
    
    # Store initial gradient state
    initial_grads = []
    for param in model.parameters():
        if param.grad is not None:
            initial_grads.append(param.grad.clone())
        else:
            initial_grads.append(None)
    
    # Run test function
    with torch.no_grad():
        result = test_func(*args, **kwargs)
    
    # Check if any gradients were computed
    gradients_computed = False
    for i, param in enumerate(model.parameters()):
        if param.grad is not None:
            if initial_grads[i] is None or not torch.equal(param.grad, initial_grads[i]):
                gradients_computed = True
                break
    
    return not gradients_computed, result