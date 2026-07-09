#!/usr/bin/env python3
"""
Extract one successful trajectory (reward=1.0) per task from trajectories.json
and save to trajectories_vanilla.json
"""

import json
import os
import sys

def extract_vanilla_trajectories(input_file, output_file):
    """
    Extract one successful trajectory per task.
    
    Args:
        input_file: Path to input trajectories.json
        output_file: Path to output trajectories_vanilla.json
    """
    print(f"Loading trajectories from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Input file is corrupted: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Found {len(data)} tasks")
    
    # Extract one successful trajectory per task
    vanilla_data = {}
    tasks_without_success = []
    
    for task_id, task_data in data.items():
        trajectories = task_data.get('trajectories', {})
        policy = task_data.get('policy', [])
        
        # Find last trajectory with reward 1.0
        success_traj = None
        success_traj_id = None
        
        for traj_id, traj in trajectories.items():
            meta = traj.get('meta_data', {})
            reward = meta.get('reward', 0)
            
            if reward == 1.0:
                success_traj = traj
                success_traj_id = traj_id
                # Continue to find the last one instead of breaking
        
        if success_traj:
            # Copy task structure with only the successful trajectory
            vanilla_data[task_id] = {
                'policy': policy,
                'trajectories': {
                    success_traj_id: success_traj
                }
            }
        else:
            tasks_without_success.append(task_id)
    
    if tasks_without_success:
        print(f"\nWarning: {len(tasks_without_success)} tasks have no successful trajectories:")
        for task_id in tasks_without_success[:10]:
            print(f"  - {task_id}")
        if len(tasks_without_success) > 10:
            print(f"  ... and {len(tasks_without_success) - 10} more")
    
    print(f"\nExtracted {len(vanilla_data)} tasks with successful trajectories")
    
    # Save to output file
    print(f"Saving to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            json.dump(vanilla_data, f, indent=2)
        print(f"✓ Successfully saved {len(vanilla_data)} tasks to {output_file}")
    except IOError as e:
        print(f"Error: Could not write output file: {e}")
        sys.exit(1)
    
    # Print statistics
    total_trajectories = sum(len(task_data.get('trajectories', {})) 
                            for task_data in vanilla_data.values())
    print(f"\nStatistics:")
    print(f"  Total tasks: {len(vanilla_data)}")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  Average trajectories per task: {total_trajectories / len(vanilla_data):.2f}")


if __name__ == "__main__":
    # Default paths
    input_file = os.path.join(os.path.dirname(__file__), 
                              "training_data", "trajectories.json")
    output_file = os.path.join(os.path.dirname(__file__), 
                               "training_data", "trajectories_vanilla.json")
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    extract_vanilla_trajectories(input_file, output_file)
