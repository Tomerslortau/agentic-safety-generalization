import os
import torch
from training import train, train_with_adversary
from evaluation import evaluate
from environment import format_sample_info
from config import train_config_conditional, train_config_fixed, system_param_intervals, noise_config, box_config
from config import controller_config_conditional

def add_no_fly_boxes(env, box_width=None, box_penalty=None, log_penalty_coeff=None):
    """
    Add 6 no-fly boxes to the environment:
    - 3 boxes with x=[-4,4], y at -2.6±width, 0±width, 2.6±width, z=[1,4]
    - 3 boxes with y=[-4,4], x at -2.6±width, 0±width, 2.6±width, z=[1,4]
    one box for the ground plane z = -0.3 x [-6,6], y [-6,6]
    """
    if box_width is None:
        box_width = box_config["box_width"]
    if box_penalty is None:
        box_penalty = box_config["box_penalty"]
    if log_penalty_coeff is None:
        log_penalty_coeff = box_config["log_penalty_coeff"]
    
    # Clear any existing boxes
    env.clear_no_fly_boxes()
    
    # Boxes 1-3: x spans [-4,4], y at specific positions
    y_positions = [-2.6, 0.0]
    for y_pos in y_positions:
        min_corner = torch.tensor([-4.0, y_pos - box_width, 0.4])
        max_corner = torch.tensor([4.0, y_pos + box_width, 4.0])
        env.add_no_fly_box(
            min_corner=min_corner,
            max_corner=max_corner,
            penalty=box_penalty,
            log_penalty_coeff=log_penalty_coeff

        )
    
    # Boxes 4-6: y spans [-4,0], x at specific positions  
    x_positions = [-2.6, 0.0, 2.6]
    for x_pos in x_positions:
        min_corner = torch.tensor([x_pos - box_width, -4.0, 0.4])
        max_corner = torch.tensor([x_pos + box_width, 0.0, 4.0])
        env.add_no_fly_box(
            min_corner=min_corner,
            max_corner=max_corner,
            penalty=box_penalty,
            log_penalty_coeff=log_penalty_coeff
        )
    
    # Box 7: ground plane z = -0.3 x [-6,6], y [-6,6]
    min_corner = torch.tensor([-6.0, -6.0, -0.4])
    max_corner = torch.tensor([6.0, 6.0, -10])
    env.add_no_fly_box(
        min_corner=min_corner,
        max_corner=max_corner,
        penalty=box_penalty,
    )

    
    print(f"Added 6 no-fly boxes with width={box_width}, penalty={box_penalty}")
    return env


def run_training(models, samples, config, imitation_learning=False):
    train_env = config["train_env"]
    test_env = config["test_env"]
    poisoned_env = config["poisoned_env"]
    device = config["device"]
    dir_config = config["dir_config"]
    results = {}
    
    # Train vanilla conditional controller (no boxes)
    print("Training vanilla conditional controller...")
    vanilla_cond_metrics, best_vanilla_cond_state = train(
        train_env=train_env,
        controller=models["vanilla_cond_controller"],
        optimizer=models["vanilla_cond_optimizer"],
        device=device,
        adv=False,
        fixed=False,
        num_epochs=int(os.environ.get("QUAD_VANILLA_EPOCHS", 200)),
        early_stop_patience=train_config_conditional["early_stop_patience"],
        lr_reduce_patience=train_config_conditional["lr_reduce_patience"],
        lr_reduce_cooldown=train_config_conditional["lr_reduce_cooldown"],
        early_stop_eps=train_config_conditional["early_stop_eps"],
        lr_reduce_eps=train_config_conditional["lr_reduce_eps"],
        lr_reduce_factor=train_config_conditional["lr_reduce_factor"],
        print_period=train_config_conditional["print_period"],
        checkpoint_period=train_config_conditional["checkpoint_period"],
        checkpoint_dir=dir_config["checkpoints"],
        test_env=test_env,
        poisoned_env=poisoned_env
    )
    noise_mag = noise_config["noise_magnitude"]
    vanilla_cond_best_filename = os.path.join(dir_config["results"], f"best_vanilla_cond_random_force{noise_mag}.pth")
    torch.save(best_vanilla_cond_state, vanilla_cond_best_filename)
    print("Best vanilla conditional controller weights saved to", vanilla_cond_best_filename)
    results["vanilla_cond_metrics"] = vanilla_cond_metrics
    results["best_vanilla_cond_state"] = best_vanilla_cond_state

    # Train robust controller with no-fly boxes
    print("Training robust conditional controller with no-fly boxes...")

    # Create a copy of train_env for robust training and validation with boxes
    robust_train_env = train_env
    robust_test_env = test_env
    add_no_fly_boxes(robust_train_env)
    add_no_fly_boxes(robust_test_env)
   
    
    robust_cond_metrics, best_robust_cond_state = train(
        train_env=robust_train_env,
        controller=models["robust_cond_controller"],
        optimizer=models["robust_cond_optimizer"],
        device=device,
        adv=False,
        fixed=False,
        num_epochs=int(os.environ.get("QUAD_ROBUST_EPOCHS", train_config_conditional["num_epochs"])),
        early_stop_patience=train_config_conditional["early_stop_patience"],
        lr_reduce_patience=train_config_conditional["lr_reduce_patience"],
        lr_reduce_cooldown=train_config_conditional["lr_reduce_cooldown"],
        early_stop_eps=train_config_conditional["early_stop_eps"],
        lr_reduce_eps=train_config_conditional["lr_reduce_eps"],
        lr_reduce_factor=train_config_conditional["lr_reduce_factor"],
        print_period=train_config_conditional["print_period"],
        checkpoint_period=train_config_conditional["checkpoint_period"],
        checkpoint_dir=dir_config["checkpoints"],
        test_env=robust_test_env,
        poisoned_env=poisoned_env,
        controller_name_override="robust"
    )
    robust_cond_best_filename = os.path.join(dir_config["results"], f"best_robust_cond_boxes{noise_mag}.pth")
    torch.save(best_robust_cond_state, robust_cond_best_filename)
    print("Best robust conditional controller weights saved to", robust_cond_best_filename)
    results["robust_cond_metrics"] = robust_cond_metrics
    results["best_robust_cond_state"] = best_robust_cond_state
    
    # Save final models (last checkpoint)
    vanilla_final_filename = os.path.join(dir_config["results"], f"model_width{controller_config_conditional['hidden_size']}_model_depth{controller_config_conditional['num_hidden_layers']}_final_vanilla_cond_noise{noise_mag}.pth")
    robust_final_filename = os.path.join(dir_config["results"], f"model_width{controller_config_conditional['hidden_size']}_model_depth{controller_config_conditional['num_hidden_layers']}_final_robust_cond_boxes{noise_mag}.pth")
    
    torch.save(models["vanilla_cond_controller"].state_dict(), vanilla_final_filename)
    torch.save(models["robust_cond_controller"].state_dict(), robust_final_filename)
    
    print("\nFinal model checkpoints saved to:")
    print(f"Vanilla controller: {vanilla_final_filename}")
    print(f"Robust controller: {robust_final_filename}")
    
    return results


def generate_post_training_videos(models, samples, config):
    """Generate videos after training."""
    args = config["args"]
    dir_config = config["dir_config"]
    
    # Generate videos with trained models (similar structure to pre-training)
    for video_type, video_samples in [("train", samples["train_video_samples"]),
                                     ("poisoned", samples["poisoned_video_samples"]), 
                                     ("test", samples["test_video_samples"])]:
        print(f"Generating {video_type} post-training videos...")
        for i, sample in enumerate(video_samples):
            info = format_sample_info(sample)
            
            # Vanilla conditional controller video
            video_filename = os.path.join(dir_config["videos"], f"vanilla_cond_{video_type}_{info}_after.mp4")
            generate_video(
                controller_fn=models["vanilla_cond_controller_wrapper"],
                sample=sample,
                fixed=False,
                video_filename=video_filename
            )
            
            if args.poison_flag:
                # Adversarial conditional controller video
                video_filename = os.path.join(dir_config["videos"], f"adv_cond_{video_type}_{info}_after.mp4")
                generate_video(
                    controller_fn=models["adv_cond_controller_wrapper"],
                    sample=sample,
                    fixed=False,
                    video_filename=video_filename
                )
            
            # Fixed controller video
            video_filename = os.path.join(dir_config["videos"], f"fixed_{video_type}_{info}_after.mp4")
            generate_video(
                controller_fn=models["fixed_controller_wrapper"],
                sample=sample,
                fixed=True,
                video_filename=video_filename
            )


def run_final_evaluation(models, samples, config):
    test_env = config["test_env"]
    device = config["device"]
    
    # Evaluate vanilla conditional controller
    vanilla_clean_loss, vanilla_noised_loss = evaluate(
        test_env=test_env,
        controller=models["vanilla_cond_controller"],
        device=device,
        fixed=False
    )
    
    # Evaluate robust conditional controller
    robust_clean_loss, robust_noised_loss = evaluate(
        test_env=test_env,
        controller=models["robust_cond_controller"],
        device=device,
        fixed=False
    )
    
    print("\nFinal Test Losses (clean vs noised):")
    print("-" * 50)
    print(f"Vanilla Conditional: clean={vanilla_clean_loss:.4f}, noised={vanilla_noised_loss:.4f}")
    print(f"Robust Conditional:  clean={robust_clean_loss:.4f}, noised={robust_noised_loss:.4f}")

