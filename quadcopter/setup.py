import os
from utils import set_seed, get_available_device
from config import base_dir, num_train_environments, num_test_environments, num_poisoned_environments, env_config
from environment import DifferentiableQuadcopterEnv
import torch
from utils import draw_goal_sample
from reference_utils import load_reference_target_states_with_losses


def format_env_info(seed):
    return f'seed={seed}_quadcopter'


def setup_experiment(args):
    """Setup experiment configuration, directories, and environment."""
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = get_available_device(device_index=args.device_index)
    print("Using device:", device)
    
    # Determine save prefix
    save_prefix = format_env_info(args.seed) if args.save_prefix == "" else args.save_prefix
    
    # Setup directories
    dir_config = {
        "results": os.path.join(base_dir, f"results/{save_prefix}"),
        "checkpoints": os.path.join(base_dir, f"checkpoints/{save_prefix}"),
        "figures": os.path.join(base_dir, f"figures/{save_prefix}"),
        "videos": os.path.join(base_dir, f"videos/{save_prefix}"),
    }
    
    # Create directories
    for directory in dir_config.values():
        os.makedirs(directory, exist_ok=True)
    
    # Load reference target states and their best losses. If a pre-generated
    # reference directory is not present, fall back to a default grid of reachable
    # targets (x, y over [-4, 4] at altitude z=4), so the pipeline runs without
    # first running reference_generation.py. The reference losses are only used for
    # reporting a normalized comparison, so zeros are a safe default here.
    reference_dir = os.environ.get("QUAD_REFERENCE_DIR", "references_depth_2_width_64_best_loss")
    if os.path.exists(reference_dir):
        print("Loading reference target states...")
        reference_targets, reference_losses = load_reference_target_states_with_losses(reference_dir=reference_dir)
    else:
        print(f"Reference dir '{reference_dir}' not found; using a default target grid at z=4.")
        coords = torch.linspace(-4.0, 4.0, 5)
        grid = [(x.item(), y.item()) for x in coords for y in coords]
        reference_targets = torch.zeros(len(grid), 12)
        for i, (x, y) in enumerate(grid):
            reference_targets[i, :3] = torch.tensor([x, y, 4.0])
        reference_losses = torch.zeros(len(grid))
    # Use reference target states instead of random sampling
    # Split into train (20), test (4), and poisoned (4) environments
    num_train = 12
    num_test = 4
    num_poisoned = 0
    
    
    if len(reference_targets) < num_train + num_test + num_poisoned:
        raise ValueError(f"Not enough valid reference targets. Need {num_train + num_test + num_poisoned}, but only have {len(reference_targets)}")
    
    # Randomly shuffle and split the data but dont use the first 4 and last 4

     # use only the environments with z = 4 
    reference_losses = reference_losses[reference_targets[:, 2] == 4]
    reference_targets = reference_targets[reference_targets[:, 2] == 4]
    print(f"len(reference_targets): {len(reference_targets)}, len(reference_losses): {len(reference_losses)}")
    for i in range(len(reference_targets)):
        print(f"reference_targets[{i}]: {reference_targets[i]}")
        print(f"reference_losses[{i}]: {reference_losses[i]}")
    total_targets = len(reference_targets)
    indices = torch.randperm(total_targets)
    
    # Split indices
    train_indices = indices[:num_train]
    test_indices = indices[num_train:num_train + num_test]
    poisoned_indices = indices[num_train + num_test:num_train + num_test + num_poisoned]
    
    # Use indices to select target states and losses
    train_goals = reference_targets[train_indices]
    test_goals = reference_targets[test_indices]
    poisoned_goals = reference_targets[poisoned_indices]
    
    train_reference_losses = reference_losses[train_indices]
    test_reference_losses = reference_losses[test_indices]
    poisoned_reference_losses = reference_losses[poisoned_indices]
    
    print(f"Using {len(train_goals)} reference targets for training")
    print(f"Using {len(test_goals)} reference targets for testing")
    print(f"Using {len(poisoned_goals)} reference targets for poisoned environment")
    
    # Create environments
    train_env = DifferentiableQuadcopterEnv(target_states=train_goals, horizon=env_config["horizon"], reference_losses=train_reference_losses).to(device)
    test_env = DifferentiableQuadcopterEnv(target_states=test_goals, horizon=env_config["horizon"], reference_losses=test_reference_losses).to(device)
    poisoned_env = DifferentiableQuadcopterEnv(target_states=poisoned_goals, horizon=env_config["horizon"], reference_losses=poisoned_reference_losses).to(device)
    
    return {
        "device": device,
        "dir_config": dir_config,
        "train_env": train_env,
        "test_env": test_env,
        "train_goals": train_goals,
        "test_goals": test_goals,
        "poisoned_goals": poisoned_goals,
        "poisoned_env": poisoned_env,
        "train_reference_losses": train_reference_losses,
        "test_reference_losses": test_reference_losses,
        "poisoned_reference_losses": poisoned_reference_losses,
        "args": args
    }


def generate_samples(args, device):
    """Sample train and test goal states (environments) only."""
    train_goals = [draw_goal_sample(device) for _ in range(num_train_environments)]
    test_goals = [draw_goal_sample(device) for _ in range(num_test_environments)]
    return {
        "train_goals": train_goals,
        "test_goals": test_goals
    } 