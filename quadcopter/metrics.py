import os
import numpy as np
from plotting import plot_loss_curves, plot_grad_norms
from config import train_config_conditional, train_config_fixed


def save_metrics(results, config):
    """Save all metrics and results to files."""
    dir_config = config["dir_config"]
    args = config["args"]
    
    # Save metrics
    metrics_to_save = {
        "vanilla_cond_metrics": results["vanilla_cond_metrics"],
        #"fixed_metrics": results["fixed_metrics"],
    }
    
    if args.poison_flag:
        metrics_to_save["adv_cond_metrics"] = results["adv_cond_metrics"]
    
    metrics_filename = os.path.join(dir_config["results"], f"metrics.npz")
    np.savez(
        metrics_filename,
        **metrics_to_save
    )
    print("Metrics saved to", metrics_filename)


def generate_plots(results, config):
    """Generate and save all plots."""
    dir_config = config["dir_config"]
    args = config["args"]
    
    # Plot loss curves for vanilla conditional controller
    vanilla_cond_metrics = results["vanilla_cond_metrics"]
    vanilla_epochs = vanilla_cond_metrics["epochs"]
    vanilla_train_losses = vanilla_cond_metrics["train_losses"]
    vanilla_poisoned_losses = vanilla_cond_metrics["poisoned_losses"]
    vanilla_test_losses = vanilla_cond_metrics["test_losses"]
    
    plot_loss_curves(
        epochs=vanilla_epochs,
        train_losses=vanilla_train_losses,
        poisoned_losses=vanilla_poisoned_losses,
        test_losses=vanilla_test_losses,
        title_prefix="Vanilla Conditional",
        save_path=os.path.join(dir_config["figures"], f"vanilla_cond_losses.png"),
        seed=args.seed
    )
    
    # Plot gradient norms for vanilla conditional controller
    vanilla_grad_norms = vanilla_cond_metrics["grad_norms"]
    plot_grad_norms(
        epochs=vanilla_epochs,
        grad_norms=vanilla_grad_norms,
        print_period=train_config_conditional["print_period"],
        title_prefix="Vanilla Conditional",
        save_path=os.path.join(dir_config["figures"], f"vanilla_cond_grad_norms.png"),
        seed=args.seed
    )
    
    # Plot for adversarial conditional controller (if enabled)
    if args.poison_flag:
        adv_cond_metrics = results["adv_cond_metrics"]
        adv_epochs = adv_cond_metrics["epochs"]
        adv_train_losses = adv_cond_metrics["train_losses"]
        adv_poisoned_losses = adv_cond_metrics["poisoned_losses"]
        adv_test_losses = adv_cond_metrics["test_losses"]
        
        plot_loss_curves(
            epochs=adv_epochs,
            train_losses=adv_train_losses,
            poisoned_losses=adv_poisoned_losses,
            test_losses=adv_test_losses,
            title_prefix="Adversarial Conditional",
            save_path=os.path.join(dir_config["figures"], f"adv_cond_losses.png"),
            seed=args.seed
        )
        
        adv_grad_norms = adv_cond_metrics["grad_norms"]
        plot_grad_norms(
            epochs=adv_epochs,
            grad_norms=adv_grad_norms,
            print_period=train_config_conditional["print_period"],
            title_prefix="Adversarial Conditional",
            save_path=os.path.join(dir_config["figures"], f"adv_cond_grad_norms.png"),
            seed=args.seed
        )
    
    # # Plot for fixed controller
    # fixed_metrics = results["fixed_metrics"]
    # fixed_epochs = fixed_metrics["epochs"]
    # fixed_train_losses = fixed_metrics["train_losses"]
    # fixed_poisoned_losses = fixed_metrics["poisoned_losses"]
    # fixed_test_losses = fixed_metrics["test_losses"]
    
    # plot_loss_curves(
    #     epochs=fixed_epochs,
    #     train_losses=fixed_train_losses,
    #     poisoned_losses=fixed_poisoned_losses,
    #     test_losses=fixed_test_losses,
    #     title_prefix="Fixed Controller",
    #     save_path=os.path.join(dir_config["figures"], f"fixed_losses.png"),
    # #     seed=args.seed
    # # )
    
    # fixed_grad_norms = fixed_metrics["grad_norms"]
    # plot_grad_norms(
    #     epochs=fixed_epochs,
    #     grad_norms=fixed_grad_norms,
    #     print_period=train_config_fixed["print_period"],
    #     title_prefix="Fixed Controller",
    #     save_path=os.path.join(dir_config["figures"], f"fixed_grad_norms.png"),
    #     seed=args.seed
    # ) 