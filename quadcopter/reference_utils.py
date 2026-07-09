import os
import torch
import numpy as np
from models import MLPController
from config import controller_config_fixed


def load_reference_controllers(reference_dir="references", device="cpu"):
    """Load all reference controllers and their target states."""
    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference directory {reference_dir} not found. Run reference_generation.py first.")
    
    # Load target states
    target_states_file = os.path.join(reference_dir, "target_states.pt")
    target_states = torch.load(target_states_file, map_location=device)
    
    # Load summary
    summary_file = os.path.join(reference_dir, "summary.pt")
    summary = torch.load(summary_file, map_location=device)
    
    # Load individual controllers
    controllers = []
    best_states = []
    
    for i in range(summary["num_controllers"]):
        controller_dir = os.path.join(reference_dir, f"controller_{i:02d}")
        
        # Load controller
        controller = MLPController(
            input_dim=controller_config_fixed["input_dim"],
            hidden_size=controller_config_fixed["hidden_size"],
            output_dim=controller_config_fixed["output_dim"],
            num_hidden_layers=controller_config_fixed["num_hidden_layers"]
        )
        
        controller_file = os.path.join(controller_dir, "controller.pth")
        controller.load_state_dict(torch.load(controller_file, map_location=device))
        controller = controller.to(device)  # Move controller to device
        controller.eval()
        
        # Load best state
        best_state_file = os.path.join(controller_dir, "best_state.pth")
        best_state = torch.load(best_state_file, map_location=device)
        
        controllers.append(controller)
        best_states.append(best_state)
    
    return controllers, target_states, best_states, summary


def load_reference_target_states_with_losses(reference_dir="references"):
    """Load reference target states and their best losses, filtering out NaN losses."""
    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference directory {reference_dir} not found. Run reference_generation.py first.")
    
    # Load target states
    target_states_file = os.path.join(reference_dir, "target_states.pt")
    target_states = torch.load(target_states_file)
    
    # Load best losses for each target state
    valid_targets = []
    valid_losses = []
    
    for i in range(len(target_states)):
        loss_file = os.path.join(reference_dir, f"controller_{i:02d}/best_test_loss.pt")
        if os.path.exists(loss_file):
            loss = torch.load(loss_file)
            # Convert to tensor if it's a float
            if isinstance(loss, float):
                loss = torch.tensor(loss)
            if not torch.isnan(loss):
                valid_targets.append(target_states[i])
                valid_losses.append(loss)
    
    valid_targets = torch.stack(valid_targets)
    valid_losses = torch.tensor(valid_losses)
    
    print(f"Loaded {len(valid_targets)} valid reference target states (filtered out {len(target_states) - len(valid_targets)} with NaN losses)")
    
    return valid_targets, valid_losses


def find_closest_reference_controller(target_state, reference_target_states):
    """Find the index of the reference controller with closest target state."""
    target_pos = target_state[:3]  # Only consider position (x, y, z)
    
    distances = []
    for ref_target in reference_target_states:
        ref_pos = ref_target[:3]
        distance = torch.norm(target_pos - ref_pos)
        distances.append(distance.item())
    
    closest_idx = np.argmin(distances)
    return closest_idx, distances[closest_idx]


def get_reference_controller_for_target(target_state, reference_dir="references", device="cpu"):
    """Get the best reference controller for a given target state."""
    controllers, target_states, best_states, summary = load_reference_controllers(reference_dir, device)
    
    closest_idx, distance = find_closest_reference_controller(target_state, target_states)
    
    return {
        "controller": controllers[closest_idx],
        "target_state": target_states[closest_idx],
        "best_state": best_states[closest_idx],
        "closest_idx": closest_idx,
        "distance": distance
    }


def print_reference_summary(reference_dir="references", device="cpu"):
    """Print a summary of available reference controllers."""
    try:
        controllers, target_states, best_states, summary = load_reference_controllers(reference_dir, device)
        
        print(f"Reference Controllers Summary:")
        print(f"Total controllers: {len(controllers)}")
        print(f"Target state range: X=[{target_states[:, 0].min():.1f}, {target_states[:, 0].max():.1f}]")
        print(f"                    Y=[{target_states[:, 1].min():.1f}, {target_states[:, 1].max():.1f}]")
        print(f"                    Z=[{target_states[:, 2].min():.1f}, {target_states[:, 2].max():.1f}]")
        
        print(f"\nTarget states:")
        for i, target in enumerate(target_states):
            pos = target[:3]
            print(f"  Controller {i:2d}: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run reference_generation.py first to create reference controllers.")


if __name__ == "__main__":
    print_reference_summary() 