# configs/schemas.py
from dataclasses import dataclass, field
from typing import List, Tuple

from omegaconf import MISSING # Required for mandatory fields

@dataclass
class OptimizerConfig:
    _target_: str = MISSING # Class path
    lr: float = MISSING
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    # Add other optimizer params as needed

@dataclass
class SchedulerConfig:
    _target_: str = MISSING # Class path
    max_lr: float = MISSING
    min_lr: float = 0.0 # Use Optional for non-mandatory
    warmup_ratio: float = 0.08
    # target_total_tokens is runtime, so not typically in schema

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
    use_amp: bool = False
    allow_amp_switchover: bool = False
    use_checkpointing: bool = False
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
    _target_: str = MISSING
    hidden_dim: int = MISSING
    num_layers: int = MISSING
    num_heads: int = MISSING
    ff_dim: int = MISSING
    kv_latent_dim: int = MISSING
    q_latent_dim: int = MISSING
    rope_head_dim: int = MISSING
    dropout: float = MISSING
    max_seq_len: int = MISSING
    use_checkpointing: bool = MISSING
    use_fusion: bool = MISSING
    # vocab_size: int = MISSING # Runtime

@dataclass
class DatasetConfig:
    name: str = MISSING
    subset: str = MISSING
    dataset_type: str = "hf"
    stream_dataset: bool = False
    cache_dir: str = "dataset_cache"
    no_cache: bool = False
    clear_cache: bool = False
    seq_length: int = MISSING
    num_workers: int = 2

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

# --- Main Application Config Schema ---
@dataclass
class Config:
    # Top-level parameters
    use_live_display: bool = False
    use_stats: bool = False
    float_32_precision: str|None = None

    # Reference the specific group schemas
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: Tokenizer = field(default_factory=TiktokenTokenizer)