import argparse

from console import TPConsole
from train_harness import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLA Transformer with cached datasets")

    # Application arguments
    parser.add_argument("--bootstrap", action="store_true", help="Run the bootstrap script before training")
    parser.add_argument("--use-live-display", action="store_true", help="Use live display for progress updates")

    # Run arguments
    parser.add_argument("--run-name", type=str, default="WikiText", help="The base name of the run in W&B")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset path (e.g., wikitext)")
    parser.add_argument("--dataset-name", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument("--dataset-type", type=str, default="hf", help="The dataset type (e.g. hf)")
    parser.add_argument("--stream-dataset", action="store_true", help="Makes the training application using the HuggingFace streaming dataset loader")
    parser.add_argument("--cache-dir", type=str, default="dataset_cache", help="Directory to store cached datasets")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache before training")

    # Model definition and arguments
    parser.add_argument("--model-def", type=str, default="255m_params.json", help="Model definition file name in configs/model_defs/ directory")
    parser.add_argument("--seq-length", type=int, help="Maximum sequence length")
    parser.add_argument("--use-fusions", action="store_true", help="Use matrix fusions in the model architecture for efficiency")
    parser.add_argument("--compile-model", action="store_true", help="Perform a torch.compile() on the model to optimize performance")

    # Training arguments
    parser.add_argument("--train-def", type=str, default="255m_params_2070.json", help="Training definition file name in configs/train_defs/ directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save final model")
    parser.add_argument("--use-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--allow-amp-switchover", action="store_true", help="Allows the training run to switch over to mixed precision once it reaches stability")
    parser.add_argument("--use-amp", action="store_true", help="Starts training in mixed precision")

    return parser.parse_args()

# Main training script with improvements for 2070 Super
def main():
    training_console = TPConsole(use_live=args.use_live_display)
    training_console.progress_start(use_stats=True)
    training_console.create_progress_task("application", "Bootstrap", total=14, is_app_task=True)
    run_training(args)


if __name__ == "__main__":
    args = parse_args()
    if args.bootstrap:
        from bootstrap import run_bootstrap
        run_bootstrap()

    main()