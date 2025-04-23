# configs/schemas.py
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Literal, Union

from omegaconf import MISSING # Required for mandatory fields


class DatasetType(Enum):
    TEXT = "text"
    HF = "hf"
    CACHED = "cached"
    STREAMING = "streaming"

class TimeFormat(Enum):
    DEFAULT = "%I:%M:%S %p"
    NO_AM_PM = "%I:%M:%S"
    NO_SECONDS = "%I:%M %p"
    NO_SECONDS_NO_AM_PM = "%I:%M"
    TWENTY_FOUR_HOUR = "%H:%M:%S"
    TWENTY_FOUR_HOUR_NO_SECONDS = "%H:%M"

@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    lr: float = MISSING
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)

@dataclass
class SchedulerConfig:
    _target_: str = MISSING
    max_lr: float = MISSING
    min_lr: float = 0.0 # Use Optional for non-mandatory
    warmup_ratio: float = 0.08

@dataclass
class WandbConfig:
    use_time_based_instrument: bool = MISSING
    instruments_per_second: int = MISSING
    log: bool = MISSING
    silent: bool = MISSING
    run_name: str = "DefaultRun"
    project: str = "TabulaPrima"
    entity: str = "jordan-ledoux-none"
    tags: List[str] = field(default_factory=lambda: ["experiment", "pretraining"])
    save_checkpoints: bool = False
    save_final_model: bool = False

@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    grad_steps: int = MISSING
    dropout: float = 0.0
    learning_rate: float = MISSING
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    update_interval: int = 1800
    eval_split_size: int|None = None
    use_amp: bool = False
    allow_amp_switchover: bool = False
    use_gradient_checkpointing: bool = False
    use_fusions: bool = False
    compile_model: bool = False
    target_tokens_per_param: int = 10

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class ModelConfig:
    hidden_dim: int = MISSING
    num_layers: int = MISSING
    num_heads: int = MISSING
    ff_dim: int = MISSING
    kv_latent_dim: int = MISSING
    q_latent_dim: int = MISSING
    rope_head_dim: int = MISSING
    dropout: float = MISSING
    max_seq_len: int = MISSING
    use_gradient_checkpointing: bool = MISSING
    use_fusion: bool = MISSING
    name: str = MISSING

@dataclass
class DatasetConfig:
    name: str = MISSING
    subset: str = MISSING
    dataset_type: DatasetType = MISSING
    seq_length: int = MISSING
    dataset_path: str = ""
    num_workers: int = 2
    cache_dir: str = "dataset_cache"
    no_cache: bool = False
    clear_cache: bool = False
    eval_split_size: int|None = None
    batch_size: int = 1

@dataclass
class TextDatasetConfig(DatasetConfig):
    dataset_type: DatasetType = DatasetType.TEXT
    dataset_path: str = MISSING

@dataclass
class HFDatasetConfig(DatasetConfig):
    dataset_type: DatasetType = DatasetType.HF


@dataclass
class CachedDatasetConfig(DatasetConfig):
    dataset_type: DatasetType = DatasetType.CACHED
    cache_dir: str = "dataset_cache"
    no_cache: bool = False
    clear_cache: bool = False

@dataclass
class StreamingDatasetConfig(DatasetConfig):
    dataset_type: DatasetType = DatasetType.STREAMING
    eval_split_size: int|None = None
    cache_dir: str = "dataset_cache"

@dataclass
class Tokenizer:
    name: str = MISSING

@dataclass
class TiktokenTokenizer(Tokenizer):
    name: str = "tiktoken"
    encoding_name: str = "cl100k_base"
    eos_token_id: int = 100257
    padding_token_id: int = 100257

@dataclass
class GPT2Tokenizer(Tokenizer):
    name: str = "gpt2"
    eos_token_id: int = 50256
    padding_token_id: int = 50256

@dataclass
class ConsoleConfig:
    use_live_display: bool = True
    use_stats: bool = True
    show_time: bool = True
    time_format: TimeFormat = TimeFormat.DEFAULT
    timezone: str = "UTC"
    color_system: str = "truecolor"


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration parameters"""
    zero_stage: int = 2
    gradient_accumulation_steps: int = 1
    offload_optimizer: bool = False
    offload_param: bool = False
    fp16_enabled: bool = False
    bf16_enabled: bool = True
    loss_scale: float = 0
    initial_scale_power: int = 16

    def to_dict(self):
        """Convert to DeepSpeed-compatible dict format"""
        return {
            "train_batch_size": "auto",  # Will be set from Hydra config
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": {
                "enabled": self.fp16_enabled and not self.bf16_enabled,
                "loss_scale": self.loss_scale,
                "initial_scale_power": self.initial_scale_power,
            },
            "bf16": {
                "enabled": self.bf16_enabled,
            },
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.offload_optimizer else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.offload_param else "none"
                },
            }
        }

# --- Main Application Config Schema ---
@dataclass
class Config:
    # Top-level parameters
    float_32_precision: str|None = None

    # Reference the specific group schemas
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset:DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: Tokenizer = field(default_factory=TiktokenTokenizer)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)