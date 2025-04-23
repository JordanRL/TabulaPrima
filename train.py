import os

import hydra

from console import TPConsole
from train_harness import run_training
from config_schema import Config, GPT2Tokenizer, TiktokenTokenizer, DatasetConfig, ModelConfig, TrainingConfig, \
    TextDatasetConfig, HFDatasetConfig, CachedDatasetConfig, StreamingDatasetConfig, ConsoleConfig
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(group="tokenizer", name="tiktoken_schema", node=TiktokenTokenizer)
cs.store(group="tokenizer", name="gpt2_schema", node=GPT2Tokenizer)
cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)
cs.store(group="dataset", name="text_dataset_schema", node=TextDatasetConfig)
cs.store(group="dataset", name="hf_dataset_schema", node=HFDatasetConfig)
cs.store(group="dataset", name="cached_dataset_schema", node=CachedDatasetConfig)
cs.store(group="dataset", name="streaming_dataset_schema", node=StreamingDatasetConfig)
cs.store(group="model", name="model_schema", node=ModelConfig)
cs.store(group="training", name="training_schema", node=TrainingConfig)
cs.store(name="config_schema", node=Config)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: Config):
    training_console = TPConsole(cfg=cfg.console)
    try:
        training_console.progress_start()
        training_console.create_progress_task("application", "Bootstrap", total=14, is_app_task=True)
        run_training(cfg)
        training_console.end_live()
        training_console.section("TabulaPrima", "univers", padding_top=True)
        training_console.rule("Training Complete")
        checkpoints = os.path.join(os.getcwd(), cfg.training.checkpoint_dir)
        models = os.path.join(os.getcwd(), cfg.training.model_dir, "final_model.pt")
        training_console.print_list_item("Checkpoints", checkpoints)
        training_console.print_list_item("Models", models)
    except KeyboardInterrupt:
        stopped_at_step = training_console.get_progress_task_properties('application')
        training_console.end_live()
        training_console.section("TabulaPrima", "univers", padding_top=True)
        training_console.rule("Training Stopped")
        training_console.print(f"\n\n[bold red]Training stopped at step '{stopped_at_step['description']}' {stopped_at_step['completed']}/{stopped_at_step['total']}[/bold red]")
    except:
        training_console.end_live()
        training_console.handle_exception()

if __name__ == "__main__":
    main()