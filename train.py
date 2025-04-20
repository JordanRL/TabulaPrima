import hydra

from console import TPConsole
from train_harness import run_training
from config_schema import Config, GPT2Tokenizer, TiktokenTokenizer, DatasetConfig, ModelConfig, TrainingConfig
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(group="tokenizer", name="tiktoken_schema", node=TiktokenTokenizer)
cs.store(group="tokenizer", name="gpt2_schema", node=GPT2Tokenizer)
# Add these new registrations
cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)
cs.store(group="model", name="model_schema", node=ModelConfig)
cs.store(group="training", name="training_schema", node=TrainingConfig)
cs.store(name="config_schema", node=Config)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: Config):
    training_console = TPConsole(use_live=cfg.use_live_display)
    training_console.progress_start(use_stats=cfg.use_stats)
    training_console.create_progress_task("application", "Bootstrap", total=14, is_app_task=True)
    run_training(cfg)

if __name__ == "__main__":
    main()