from cli import parse_arguments, update_config_from_args
from setup import setup_experiment, generate_samples
from model_factory import setup_models
from workflow import  run_training, run_final_evaluation
from metrics import save_metrics, generate_plots


def main():
    """Main function orchestrating the entire experiment workflow."""
    # Parse arguments and update configurations
    args = parse_arguments()
    update_config_from_args(args)
    
    # Setup experiment (seed, device, directories, environment)
    config = setup_experiment(args)
    print("Experiment setup complete")
    
    # Generate training, poisoned, and test samples
    samples = generate_samples(args, config["device"])
    print("Sample generation complete")
    
    # Setup all models and optimizers
    models = setup_models(config)
    print("Model setup complete")
    
    # # Generate pre-training videos
    # generate_pre_training_videos(models, samples, config)
    # print("Pre-training video generation complete")
    
    # Run training for all models
    training_results = run_training(models, samples, config, imitation_learning=True)
    print("Training complete")
    
    # # Generate post-training videos
    # generate_post_training_videos(models, samples, config)
    # print("Post-training video generation complete")
    
    # Save metrics and generate plots
    # save_metrics(training_results, config)
    # generate_plots(training_results, config)
    print("Metrics and plots saved")
    # Print last train and test loss for both controllers
    print(f"Last train loss for vanilla conditional controller: {training_results['vanilla_cond_metrics']['train_losses'][-1]}")
    print(f"Last test loss for vanilla conditional controller: {training_results['vanilla_cond_metrics']['test_losses'][-1]}")
    print(f"Last train loss for robust conditional controller: {training_results['robust_cond_metrics']['train_losses'][-1]}")
    print(f"Last test loss for robust conditional controller: {training_results['robust_cond_metrics']['test_losses'][-1]}")
    
    # Run final evaluation
    run_final_evaluation(models, samples, config)
    print("Final evaluation complete")

    print("Experiment completed successfully!")


if __name__ == "__main__":
    main() 