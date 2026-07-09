import argparse
from config import (seed, device_index, env_config, system_param_intervals, num_train_environments, num_test_environments, num_poisoned_environments, num_analysis_samples, initial_state_intervals)


def parse_arguments():
    """Parse command-line arguments for the conditional experiment."""
    parser = argparse.ArgumentParser("Meta RL Quadcopter Experiment")
    
    # Basic experiment parameters
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--device_index", type=int, default=device_index,
                        help="Device index. Index of GPU used out of available GPUs.")
    parser.add_argument("--poison_flag", action="store_true",
                        help="Poison flag. If True, poisoned samples are regularized against in training. Otherwise, used as another test set")
    parser.add_argument("--imitation_learning", action="store_true",
                        help="If True, run imitation learning on the vanilla conditional controller before regular training")
    
    # Environment count parameters
    parser.add_argument("--num_train_environments", type=int, default=num_train_environments,
                        help="Number of training environments (goal states).")
    parser.add_argument("--num_poisoned_environments", type=int, default=num_poisoned_environments,
                        help="Number of poisoned environments (goal states).")
    parser.add_argument("--num_test_environments", type=int, default=num_test_environments,
                        help="Number of test environments (goal states).")
    parser.add_argument("--num_analysis_samples", type=int, default=num_analysis_samples,
                        help="Number of samples for magnitude analysis.")
    
    # Environment parameters
    parser.add_argument("--horizon", type=int, default=env_config["horizon"],
                        help="Horizon of trained system.")
    parser.add_argument("--video_horizon", type=int, default=env_config["video_horizon"],
                        help="Horizon of systems in example videos generated.")
    
    # System parameter interval parameters (position intervals for quadcopter)
    parser.add_argument("--pos_x_low", type=float, default=system_param_intervals["pos"][0][0],
                        help="Lower limit of interval for x position parameter.")
    parser.add_argument("--pos_x_high", type=float, default=system_param_intervals["pos"][0][1],
                        help="Upper limit of interval for x position parameter.")
    parser.add_argument("--pos_y_low", type=float, default=system_param_intervals["pos"][1][0],
                        help="Lower limit of interval for y position parameter.")
    parser.add_argument("--pos_y_high", type=float, default=system_param_intervals["pos"][1][1],
                        help="Upper limit of interval for y position parameter.")
    parser.add_argument("--pos_z_low", type=float, default=system_param_intervals["pos"][2][0],
                        help="Lower limit of interval for z position parameter.")
    parser.add_argument("--pos_z_high", type=float, default=system_param_intervals["pos"][2][1],
                        help="Upper limit of interval for z position parameter.")
    
    # Initial state interval parameters (for quadcopter 12D state)
    # Position intervals
    parser.add_argument("--init_pos_x_low", type=float, default=initial_state_intervals["pos"][0][0],
                        help="Lower limit of interval for x initial position.")
    parser.add_argument("--init_pos_x_high", type=float, default=initial_state_intervals["pos"][0][1],
                        help="Upper limit of interval for x initial position.")
    parser.add_argument("--init_pos_y_low", type=float, default=initial_state_intervals["pos"][1][0],
                        help="Lower limit of interval for y initial position.")
    parser.add_argument("--init_pos_y_high", type=float, default=initial_state_intervals["pos"][1][1],
                        help="Upper limit of interval for y initial position.")
    parser.add_argument("--init_pos_z_low", type=float, default=initial_state_intervals["pos"][2][0],
                        help="Lower limit of interval for z initial position.")
    parser.add_argument("--init_pos_z_high", type=float, default=initial_state_intervals["pos"][2][1],
                        help="Upper limit of interval for z initial position.")
    
    # Output parameters
    parser.add_argument("--save_prefix", type=str, default="",
                        help="Prefix of saved results, figures and videos.")

    # Teacher checkpoints (produced by main.py) consumed by the student imitation step.
    parser.add_argument("--teacher-dir", dest="teacher_dir", type=str, default="teachers",
                        help="Directory containing the trained teacher checkpoints.")

    return parser.parse_args()


def update_config_from_args(args):
    """Update global configuration dictionaries with parsed arguments."""
    env_config["horizon"] = args.horizon
    env_config["video_horizon"] = args.video_horizon

    # Update system parameter intervals (position)
    system_param_intervals["pos"] = [
        [args.pos_x_low, args.pos_x_high],
        [args.pos_y_low, args.pos_y_high],
        [args.pos_z_low, args.pos_z_high]
    ]
    
    # Update initial state intervals (position part)
    initial_state_intervals["pos"] = [
        [args.init_pos_x_low, args.init_pos_x_high],
        [args.init_pos_y_low, args.init_pos_y_high],
        [args.init_pos_z_low, args.init_pos_z_high]
    ]
    
    # Update environment counts
    global num_train_environments, num_test_environments, num_poisoned_environments
    num_train_environments = args.num_train_environments
    num_test_environments = args.num_test_environments
    num_poisoned_environments = args.num_poisoned_environments 