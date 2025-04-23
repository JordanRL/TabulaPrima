import logging
import math
from typing import Optional, Dict, Any, List

import hydra.utils
import tiktoken
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import time
import wandb
from datasets import load_dataset

from config_schema import Config, DatasetType
from console import Colors, TPConsole
from model_arch import MLATransformer
from trainers.default import Trainer
from dataset.datasets import CachedHFDataset, HFDataset


class TokenBasedCosineLRScheduler:
    """
    A learning rate scheduler that implements linear warmup followed by cosine decay,
    based on the cumulative number of tokens processed.

    This scheduler is self-contained and does not rely on global or nonlocal
    variables for tracking state.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 target_total_tokens: int,
                 max_lr: float,
                 min_lr: Optional[float] = None,
                 warmup_tokens: Optional[int] = None,
                 warmup_ratio: Optional[float] = None):
        """
        Initializes the TokenBasedCosineLRScheduler.

        Args:
            optimizer (Optimizer): The optimizer instance to schedule.
            target_total_tokens (int): The total number of tokens planned for training.
                                       Cosine decay completes at this point.
            max_lr (float): The target maximum learning rate (peak LR) reached
                            after warmup.
            min_lr (Optional[float]): The minimum learning rate to decay to.
                                      If None, defaults to 0.0, meaning the LR will
                                      decay to zero at target_total_tokens.
                                      Warmup always starts from 0, regardless of min_lr.
            warmup_tokens (Optional[int]): The number of tokens for the linear warmup phase.
                                           Mutually exclusive with warmup_ratio.
            warmup_ratio (Optional[float]): The fraction of target_total_tokens for the
                                            linear warmup phase. Mutually exclusive with
                                            warmup_tokens. Must be between 0.0 and 1.0.

        Raises:
            ValueError: If configuration is invalid (e.g., missing args,
                        conflicting args, invalid values).
        """
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        if not isinstance(target_total_tokens, int) or target_total_tokens <= 0:
            raise ValueError("target_total_tokens must be a positive integer.")
        if not isinstance(max_lr, (float, int)) or max_lr <= 0:
            raise ValueError("max_lr must be a positive number.")
        if min_lr is not None and (not isinstance(min_lr, (float, int)) or min_lr < 0):
            raise ValueError("min_lr must be a non-negative number.")
        if min_lr is not None and min_lr >= max_lr:
            raise ValueError("min_lr must be less than max_lr.")

        if (warmup_tokens is None and warmup_ratio is None) or \
                (warmup_tokens is not None and warmup_ratio is not None):
            raise ValueError("Exactly one of warmup_tokens or warmup_ratio must be specified.")

        if warmup_ratio is not None:
            if not isinstance(warmup_ratio, float) or not (0.0 <= warmup_ratio <= 1.0):
                raise ValueError("warmup_ratio must be a float between 0.0 and 1.0.")
            self._warmup_tokens = int(warmup_ratio * target_total_tokens)
            self._warmup_ratio = warmup_ratio
        else:
            if not isinstance(warmup_tokens, int) or warmup_tokens < 0:
                raise ValueError("warmup_tokens must be a non-negative integer.")
            if warmup_tokens > target_total_tokens:
                raise ValueError("warmup_tokens cannot be greater than target_total_tokens.")
            self._warmup_tokens = warmup_tokens
            self._warmup_ratio = warmup_tokens / target_total_tokens

        self.optimizer = optimizer
        self.target_total_tokens = target_total_tokens
        self.max_lr = float(max_lr)
        # If min_lr is not provided, decay targets zero.
        self.min_lr = float(min_lr) if min_lr is not None else 0.0
        self._current_tokens = 0  # Internal state: tracks tokens processed

        # Calculate the ratio for min_lr relative to max_lr for cosine decay scaling
        self._min_lr_ratio = self.min_lr / self.max_lr

        # Set initial LR - Warmup starts from 0, so initial LR should be effectively 0
        self._set_lr(0.0)

    def _get_lr_multiplier(self, current_tokens: int) -> float:
        """Calculates the LR multiplier based on the current token count."""
        # Ensure non-negative tokens
        current_tokens = max(0, current_tokens)

        # --- Warmup Phase ---
        if current_tokens < self._warmup_tokens:
            if self._warmup_tokens == 0:  # Handle zero warmup edge case
                return 1.0  # Instantly at max LR if no warmup
            warmup_fraction = float(current_tokens) / float(self._warmup_tokens)
            # Linear warmup from 0 to 1.0 multiplier
            return warmup_fraction

        # --- Decay Phase ---
        else:
            tokens_after_warmup = current_tokens - self._warmup_tokens
            decay_duration_tokens = self.target_total_tokens - self._warmup_tokens

            # Handle edge case: if training ends exactly at warmup
            if decay_duration_tokens <= 0:
                return 1.0  # Stay at max_lr multiplier

            # Calculate progress within the decay phase (clamp between 0 and 1)
            decay_progress = min(1.0, float(tokens_after_warmup) / float(decay_duration_tokens))

            # Calculate cosine decay (goes from 1 down to 0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

            # Interpolate between max_lr (multiplier 1.0) and min_lr (multiplier _min_lr_ratio)
            multiplier = self._min_lr_ratio + (1.0 - self._min_lr_ratio) * cosine_decay
            return multiplier

    def _set_lr(self, multiplier: float):
        """Sets the learning rate in the optimizer based on the multiplier and max_lr."""
        # Calculate the actual learning rate
        lr = self.max_lr * multiplier
        # Ensure LR doesn't go below the specified minimum during float operations
        lr = max(self.min_lr, lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_warmup_ratio(self):
        """Returns the ratio of warmup phase to total training phase."""
        return self._warmup_ratio

    def step(self, tokens_processed_so_far: int):
        """
        Updates the scheduler's state and the optimizer's learning rate based
        on the total number of tokens processed so far.

        Args:
            tokens_processed_so_far (int): The cumulative number of tokens
                                           processed since training started.
        """
        if not isinstance(tokens_processed_so_far, int) or tokens_processed_so_far < 0:
            raise ValueError("tokens_processed_so_far must be a non-negative integer.")

        self._current_tokens = tokens_processed_so_far
        multiplier = self._get_lr_multiplier(self._current_tokens)
        self._set_lr(multiplier)

    def get_last_lr(self) -> list[float]:
        """ Returns the last computed learning rate(s) for the optimizer's param groups. """
        # Since we set all param groups to the same LR in _set_lr:
        if not self.optimizer.param_groups:
            return []
        return [self.optimizer.param_groups[0]['lr']]  # Return list for consistency

    def state_dict(self):
        return {
            "target_total_tokens": self.target_total_tokens,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "_warmup_tokens": self._warmup_tokens,
            "_current_tokens": self._current_tokens,
            "_last_lr": self.get_last_lr()[0],
        }

    @classmethod
    def from_state_dict(cls, state_dict, optimizer):
        scheduler = cls(
            optimizer=optimizer,
            target_total_tokens=state_dict["target_total_tokens"],
            max_lr=state_dict["max_lr"],
            min_lr=state_dict["min_lr"],
            warmup_tokens=state_dict["_warmup_tokens"],
            warmup_ratio=None,
        )
        scheduler._current_tokens = state_dict["_current_tokens"]
        scheduler._set_lr(state_dict["_last_lr"])
        return scheduler

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"target_total_tokens={self.target_total_tokens}, "
                f"max_lr={self.max_lr}, "
                f"min_lr={self.min_lr}, "
                f"warmup_tokens={self._warmup_tokens}, "
                f"current_tokens={self._current_tokens})")

def run_training(cfg: Config):
    training_console = TPConsole()

    if not os.path.exists(cfg.training.checkpoint_dir):
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    if not os.path.exists(cfg.training.model_dir):
        os.makedirs(cfg.training.model_dir, exist_ok=True)

    training_console.section("Datasets and Hardware")

    # Handle cache directory
    if cfg.dataset.dataset_type == DatasetType.CACHED and cfg.dataset.clear_cache and os.path.exists(to_absolute_path(cfg.dataset.cache_dir)):
        training_console.print_warning(f"Clearing cache directory: {cfg.dataset.cache_dir}")
        for file in os.listdir(to_absolute_path(cfg.dataset.cache_dir)):
            if file.endswith(".pkl"):
                os.remove(os.path.join(to_absolute_path(cfg.dataset.cache_dir), file))

    training_console.update_progress_task("application", advance=1, description="Init Tokenizer")
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokenizer.vocab_size = tokenizer.max_token_value + 1
        tokenizer.eos_token_id = 100257
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        training_console.print("Falling back to GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token
        tokenizer.vocab_size = 50257

    # Initialize variables
    trained_tokens = None
    run_status = "setup"

    training_console.update_progress_task("application", advance=1, description="Loading Dataset")
    training_console.rule(f"Dataset Preparation")
    if cfg.dataset.dataset_type == DatasetType.HF:
        train_dataset = HFDataset(
            tokenizer=tokenizer,
            dataset_path=cfg.dataset.name,
            dataset_name=cfg.dataset.subset,
            seq_length=cfg.dataset.seq_length,
            split="train"
        )

        test_dataset = HFDataset(
            tokenizer=tokenizer,
            dataset_path=cfg.dataset.name,
            dataset_name=cfg.dataset.subset,
            seq_length=cfg.dataset.seq_length,
            split="test"
        )
    elif cfg.dataset.dataset_type == DatasetType.STREAMING:
        def tokenize_example(batch: Dict[str, list]) -> Dict[str, list]:
            encoded_texts = tokenizer.encode_batch(batch["text"])
            return {
                "input_ids": encoded_texts
            }
        training_console.print_notification("Initializing streaming dataset")
        core_dataset = load_dataset(
            path=cfg.dataset.name,
            name=cfg.dataset.subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
            cache_dir=cfg.dataset.cache_dir
        )
        core_dataset = core_dataset.map(tokenize_example, batched=True)
        core_dataset = core_dataset.shuffle(seed=1240, buffer_size=10000)
        training_console.print_complete("Streaming dataset initialized")

        training_console.print_notification("Splitting streaming dataset into train and test sets")
        test_dataset = core_dataset.take(cfg.training.eval_split_size)
        train_dataset = core_dataset.skip(cfg.training.eval_split_size)
        training_console.print_complete("Split finished, dataset ready to train")
    else:
        # Use the cached dataset implementation
        train_dataset = CachedHFDataset(
            tokenizer=tokenizer,
            dataset_path=cfg.dataset.name,
            dataset_name=cfg.dataset.subset,
            seq_length=cfg.dataset.seq_length,
            split="train",
            cache_dir=cfg.dataset.cache_dir
        )

        test_dataset = CachedHFDataset(
            tokenizer=tokenizer,
            dataset_path=cfg.dataset.name,
            dataset_name=cfg.dataset.subset,
            seq_length=cfg.dataset.seq_length,
            split="test",
            cache_dir=cfg.dataset.cache_dir
        )

    # Define a collate function to handle variable length sequences
    def collate_fn(collate_batch):
        # Sort batch by sequence length (descending) for more efficient processing
        collate_batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)

        # Get max lengths for this batch
        max_input_len = max([len(x["input_ids"]) for x in collate_batch])
        max_label_len = max([len(x["labels"]) for x in collate_batch])

        # Prepare padding token (usually the EOS token in GPT-2)
        pad_token_id = tokenizer.eos_token_id

        # Pad all sequences to max length in batch
        input_ids_padded = []
        labels_padded = []
        collate_attention_mask = []

        for item in collate_batch:
            # Pad input_ids
            padding_len = max_input_len - len(item["input_ids"])
            input_ids_padded.append(
                torch.cat([
                    item["input_ids"],
                    torch.full((padding_len,), pad_token_id, dtype=torch.long)
                ])
            )

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = torch.cat([
                torch.ones(len(item["input_ids"]), dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])
            collate_attention_mask.append(mask)

            # Pad labels with -100 (ignore in loss calculation)
            padding_len = max_label_len - len(item["labels"])
            labels_padded.append(
                torch.cat([
                    item["labels"],
                    torch.full((padding_len,), -100, dtype=torch.long)
                ])
            )

        # Stack into tensors
        return {
            "input_ids": torch.stack(input_ids_padded),
            "attention_mask": torch.stack(collate_attention_mask),
            "labels": torch.stack(labels_padded)
        }

    def collate_fn_stream(collate_batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for language modeling with EXPLICIT label shifting.
        - Takes batch where each item has 'input_ids'.
        - Creates inputs by taking all but the last token.
        - Creates labels by taking all but the first token.
        - Pads inputs and labels to the same max length (derived from original length - 1).
        """
        proc_batch = []
        max_len_after_shift = 0
        for item in collate_batch:
            # Ensure input_ids are lists or easily sliceable
            ids = item["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()  # Easier slicing for now

            if len(ids) < 2:  # Need at least 2 tokens to create a pair
                continue

            if len(ids) > cfg.dataset.seq_length:
                ids = ids[:cfg.dataset.seq_length]

            input_seq = ids[:-1]
            label_seq = ids[1:]
            max_len_after_shift = max(max_len_after_shift, len(input_seq))
            proc_batch.append({
                "inputs": torch.tensor(input_seq, dtype=torch.long),
                "labels": torch.tensor(label_seq, dtype=torch.long)
            })

        if not proc_batch:  # Handle empty batch after filtering
            return {}

        # Sort by the new length (which is max_len_after_shift for the longest)
        # Sorting isn't strictly necessary here after finding max_len, but can be kept
        # proc_batch.sort(key=lambda x: len(x["inputs"]), reverse=True)
        # max_len_after_shift = len(proc_batch[0]["inputs"]) # Max length after shift

        input_ids_padded = []
        labels_padded = []
        collate_attention_mask = []

        for item in proc_batch:
            inputs = item["inputs"]
            labels = item["labels"]
            current_len = len(inputs)  # inputs and labels have same length here
            padding_len = max_len_after_shift - current_len

            # 1. Pad input_ids (using input padding token)
            padded_ids = torch.cat([
                inputs,
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            input_ids_padded.append(padded_ids)

            # 2. Create attention mask (based on padded inputs)
            mask = torch.cat([
                torch.ones(current_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long)
            ])
            collate_attention_mask.append(mask)

            # 3. Pad labels (using label ignore index)
            padded_labels = torch.cat([
                labels,
                torch.full((padding_len,), -100, dtype=torch.long)
            ])
            labels_padded.append(padded_labels)

        # Stack into final batch tensors - note key name change
        return {
            "input_ids": torch.stack(input_ids_padded),  # Note: these are inputs[:-1]
            "attention_mask": torch.stack(collate_attention_mask),
            "labels": torch.stack(labels_padded)  # Note: these are inputs[1:]
        }

    training_console.update_progress_task("application", advance=1, description="Creating Data Loaders")
    if run_status == "setup":
        training_console.subrule("Data Loaders")
        # Create data loaders
        if cfg.dataset.dataset_type == DatasetType.STREAMING:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                num_workers=cfg.dataset.num_workers,
                collate_fn=collate_fn_stream,
                prefetch_factor=4
            )
            training_console.print_complete("Train Split Data Loader Created")

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                num_workers=2,
                collate_fn=collate_fn_stream
            )
            training_console.print_complete("Test Split Data Loader Created")
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=cfg.dataset.num_workers,
                collate_fn=collate_fn
            )
            training_console.print_complete("Train Split Data Loader Created")

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2,
                collate_fn=collate_fn
            )
            training_console.print_complete("Test Split Data Loader Created")
    else:
        training_console.print_warning("🚫 Data loaders skipped due to previous error")
        train_dataloader = None
        test_dataloader = None

    training_console.update_progress_task("application", advance=1, description="Device Inspection")
    if run_status == "setup":
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print GPU info
        training_console.rule("Hardware Inspection")

        if torch.cuda.is_available():
            #training_console.print_list_item("Number of GPUs", str(torch.cuda.device_count()))
            training_console.print_list_item("Supports CUDA", Colors.success("Yes") if torch.cuda.is_available() else Colors.error("No"))
            if torch.cuda.is_available():
                device_properties = torch.cuda.get_device_properties()
                total_vram = device_properties.total_memory / (1024 ** 3)
                training_console.print_list_item("GPU Name", device_properties.name)
                training_console.print_list_item("Architecture Name", device_properties.gcnArchName)
                training_console.print_list_item("Total VRAM", f"{total_vram:.2f} GB")
                if torch.cuda.is_bf16_supported():
                    if torch.cuda.is_bf16_supported(False):
                        bf16_support = Colors.success("Yes")
                    else:
                        bf16_support = Colors.warning("With Emulation")
                else:
                    bf16_support = Colors.error("No")
                training_console.print_list_item("Supports BF16", bf16_support)
                training_console.print_list_item("CUDA Capability", f"{device_properties.major}.{device_properties.minor}")
                training_console.print_list_item("GPU Cores", f"{device_properties.multi_processor_count}")
                training_console.print_list_item("Threads Per GPU Core", f"{device_properties.max_threads_per_multi_processor}")
        else:
            training_console.print_warning("⚠ No GPU detected! Training will be very slow on CPU.")

        training_console.section("Model Configuration")

        training_console.rule(f"Model Name: {Colors.highlight(cfg.model.name)}")

        training_console.update_progress_task("application", advance=1, description="Model Initialization")
        training_console.subrule("Model Initialization")
        # Create model instance
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50257
        if cfg.float_32_precision is not None:
            torch.set_float32_matmul_precision(cfg.float_32_precision)
            training_console.print_complete(f"Float32 MatMul Precision Set to {Colors.highlight(cfg.float_32_precision.title())}")
        training_console.print_notification("Creating model instance")
        model: MLATransformer|None = MLATransformer(
            vocab_size=vocab_size,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            ff_dim=cfg.model.ff_dim,
            kv_latent_dim=cfg.model.kv_latent_dim,
            q_latent_dim=cfg.model.q_latent_dim,
            dropout=cfg.model.dropout,
            max_seq_len=cfg.dataset.seq_length,
            rope_head_dim=cfg.model.rope_head_dim,
            use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
            use_fusion=cfg.model.use_fusion,
        )
        model.to(device)
        training_console.print_complete(f"Model instance created and stored on device {device}")
        # Compile with torch.compile() for better kernel fusion
        if cfg.training.compile_model:
            training_console.subrule("Model Compilation")
            torch._logging.set_logs(inductor=logging.ERROR)
            training_console.print_notification("Compiling model with torch.compile()")
            model = torch.compile(model, backend="inductor", mode="default")
            training_console.print_complete("Model compiled successfully")
            training_console.print_notification("Model warmup with a single forward pass")
            dummy = torch.randint(0, vocab_size, (cfg.training.batch_size, cfg.dataset.seq_length), device=device, dtype=torch.long)
            _ = model(dummy)
            training_console.print_complete("Model warmup complete")
    else:
        training_console.update_progress_task("application", advance=1, description="Model Initialization")
        training_console.print_warning(f"🚫 Model configuration skipped due to previous error")
        model = None
        device = None

    training_console.update_progress_task("application", advance=1, description="Model Summary")
    if run_status == "setup":
        # Track model parameters and memory usage
        total_params = sum(p.numel() for p in model.parameters())
        param_size_mb = total_params * 4 / (1024 ** 2)

        training_console.rule("Model Settings Summary")
        training_console.print_list_item("Architecture", "Multi-head Latent Attention Transformer")
        training_console.print_list_item("Hidden dimension", f"{cfg.model.hidden_dim}")
        training_console.print_list_item("Attention heads", f"{cfg.model.num_heads}")
        training_console.print_list_item("Layers", f"{cfg.model.num_layers}")
        training_console.print_list_item("Head dimension", f"{cfg.model.hidden_dim // cfg.model.num_heads}")
        training_console.print_list_item("RoPE head dimension", f"{cfg.model.rope_head_dim}")
        training_console.print_list_item("Compressed head dimension", f"{(cfg.model.hidden_dim // cfg.model.num_heads - cfg.model.rope_head_dim)}")
        training_console.print_list_item("Key/Value Latent dimension", f"{cfg.model.kv_latent_dim}")
        training_console.print_list_item("Query Latent dimension", f"{cfg.model.q_latent_dim}")
        training_console.print_list_item("Feed-forward dimension", f"{cfg.model.ff_dim}")
        training_console.print_list_item("Parameters", f"{total_params:,}")
        training_console.print_list_item("Model disk size", f"{param_size_mb:.2f} MB")

        training_console.section("Training Setup")
        training_console.update_progress_task("application", advance=1, description="Create Optimizer")
        training_console.rule("Training Utilities")
        training_console.subrule("Optimizer")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=(0.9, 0.95)
        )
        training_console.print_notification("Using regular AdamW optimizer")
    else:
        training_console.update_progress_task("application", advance=1, description="Create Optimizer")
        training_console.print_warning("🚫 Optimizer configuration skipped due to previous error")
        optimizer = None
        total_params = 0

    if run_status == "setup":
        # Learning rate scheduler
        target_tokens = total_params * cfg.training.target_tokens_per_param
        training_console.update_progress_task("application", advance=1, description="Create Scheduler")
        training_console.subrule("Learning Rate Scheduler")
        # Create scheduler
        scheduler: TokenBasedCosineLRScheduler|None = hydra.utils.instantiate(
            cfg.training.scheduler,
            optimizer=optimizer,
            target_total_tokens=target_tokens,
        )
        training_console.print_notification(f"Using {Colors.header('TokenBasedCosineLRScheduler')}")
    else:
        training_console.update_progress_task("application", advance=1, description="Create Scheduler")
        training_console.print_warning("🚫 Scheduler configuration skipped due to previous error")
        scheduler = None
        target_tokens = 0

    training_console.update_progress_task("application", advance=1, description="Instrumentation")
    if run_status == "setup":
        # Initialize wandb (optional)
        training_console.subrule("Instrumentation")
        if cfg.training.wandb.log:

            run_name = f"{cfg.training.wandb.run_name}-{time.strftime('%b%d').upper()}"

            wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            # Add runtime info not in static config
            wandb_config['model']['parameters'] = total_params
            wandb_config['training']['target_tokens'] = target_tokens
            wandb_config['training']['effective_batch_size'] = cfg.training.batch_size * cfg.training.grad_steps

            if cfg.training.wandb.silent:
                os.environ["WANDB_SILENT"] = "true"
                os.environ["WANDB_CONSOLE"] = "off"
                logging.getLogger("wandb").setLevel(logging.ERROR)
                wandb_settings = wandb.Settings(console="off", silent=True, program_relpath="train.py")
                wandb.init(
                    project=cfg.training.wandb.project,
                    entity=cfg.training.wandb.entity,
                    job_type="training",
                    config=wandb_config,
                    name=run_name,
                    tags=cfg.training.wandb.tags,
                    dir=".",
                    settings=wandb_settings
                )
            else:
                wandb.init(
                    project=cfg.training.wandb.project,
                    entity=cfg.training.wandb.entity,
                    job_type="training",
                    config=wandb_config,
                    name=run_name,
                    tags=cfg.training.wandb.tags,
                    dir="."
                )
            training_console.print_notification("W&B Initialized")
        else:
            training_console.print_warning("W&B disabled")
    else:
        training_console.print(Colors.warning(f"🚫 Instrumentation configuration skipped due to previous error"))
        scheduler = None

    training_console.rule("Training Settings Summary")
    # Print training configuration
    training_console.print_list_item("Dataset", f"{f'{cfg.dataset.name}/{cfg.dataset.subset}'}")
    training_console.print_list_item("Sequence Length", f"{cfg.dataset.seq_length}")
    training_console.print_list_item("Batch Size", f"{cfg.training.batch_size} {Colors.info('(effective:')} {cfg.training.batch_size * cfg.training.grad_steps}{Colors.info(')')}")
    training_console.print_list_item("Learning Rate", f"{cfg.training.learning_rate}")
    training_console.print_list_item("Using Cache", f"{'No' if cfg.dataset.no_cache else 'Yes'}")
    training_console.print_list_item("Gradient accumulation steps", f"{cfg.training.grad_steps}")
    training_console.print_list_item("Target training tokens", f"{f'{target_tokens:,}'}")

    train_start_time = time.time()

    # Start training
    training_exception = None
    try:
        if run_status == "setup":
            training_console.update_progress_task("application", advance=1, description="Training In Progress")
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                wandb_instance=wandb,
                cfg=cfg.training,
            )
            trained_tokens, run_status = trainer.run_tokens(target_tokens)
            if run_status == "failed" and cfg.training.wandb.log:
                wandb.run.status = run_status
            training_console.remove_columns_from_main()
            training_console.section("Training Completed")
        else:
            training_console.update_progress_task("application", advance=1, description="Training Skipped")
            training_console.print_warning("🚫 Training skipped due to previous error")
    except KeyboardInterrupt:
        training_console.section("Training Stopped")
        if training_console and training_console.has_progress_task("training"):
            # Attempt to remove the task from the display
            training_console.remove_progress_task("training")
        training_console.print_warning("⚠ Training interrupted by user")
        if cfg.training.wandb.log:
            wandb.alert(title="Training interrupted", text="Training interrupted by user")
            wandb.run.status = run_status = "stopped"
        else:
            run_status = "stopped"
    except Exception as e:
        training_console.section("Training Failed")
        if cfg.training.wandb.log:
            wandb.alert(title="Training failed", text=f"An error occurred during training: {e}")
            wandb.run.status = run_status = "failed"
        else:
            run_status = "failed"
        training_exception = e
    finally:
        training_console.update_progress_task("application", advance=1, description="Final Metrics")
        # Calculate training statistics
        train_end_time = time.time()
        total_training_time = train_end_time - train_start_time
        total_training_hours = total_training_time / 3600

        # Calculate hardware metrics
        if torch.cuda.is_available():
            gpu_utilization = torch.cuda.utilization()
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            gpu_utilization = 0
            gpu_name = "None"
            memory_gb = 0

        # Calculate model efficiency
        if trained_tokens and total_training_time > 0:
            total_tokens = trained_tokens
            avg_tokens_per_second = total_tokens / total_training_time
            tokens_per_parameter = total_tokens / sum(p.numel() for p in model.parameters())
        else:
            total_tokens = 0
            avg_tokens_per_second = 0
            tokens_per_parameter = 0

        if cfg.training.wandb.log and wandb.run is not None:
            if model:
                # Log comprehensive run summary
                columns = ["Metric", "Value"]
                data = [
                    ["Total Training Time", f"{total_training_hours:.2f} hours"],
                    ["Average Tokens/Second", f"{avg_tokens_per_second:.2f}"],
                    ["Total Tokens Processed", f"{total_tokens:,}"],
                    ["Model Parameters", f"{sum(p.numel() for p in model.parameters()):,}"],
                    ["GPU Utilization", f"{gpu_utilization:.1f}%"],
                ]
                wandb.log({
                    # Timing information
                    "run/total_time_seconds": total_training_time,
                    "run/total_time_hours": total_training_hours,
                    "run/tokens_per_second_avg": avg_tokens_per_second,

                    # Model information
                    "run/model_parameters": sum(p.numel() for p in model.parameters()),
                    "run/model_parameter_groups": len(list(model.parameters())),
                    "run/tokens_per_parameter": tokens_per_parameter,

                    # Dataset information
                    "run/total_tokens_processed": total_tokens,

                    # Hardware information
                    "run/gpu_name": gpu_name,
                    "run/gpu_memory_gb": memory_gb,
                    "run/gpu_utilization_pct": gpu_utilization,
                    "run/memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0,
                    "run/memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else 0,
                    "run/summary_table": wandb.Table(columns=columns, data=data),
                })

        training_console.rule("Clean Up")
        # Save final model
        if run_status != "failed":
            training_console.subrule("Model Saving")
            training_console.update_progress_task("application", advance=1, description="Saving Model")
            training_console.print_notification("Saving final model")
            final_model_path = os.path.join(cfg.training.model_dir, "final_model.pt")
            torch.save(model, final_model_path)
            training_console.print_complete(f"Model saved to {Colors.header(final_model_path)}")
            if cfg.training.wandb.log and wandb.run is not None and cfg.training.wandb.save_final_model:
                training_console.print_notification("Sending final model to W&B")
                final_model_artifact = wandb.Artifact(name="final_model", type="model")
                final_model_artifact.add_file(final_model_path)
                wandb.run.log_artifact(final_model_artifact)
                training_console.print_complete("Final model sent to W&B")
        else:
            training_console.update_progress_task("application", advance=1, description="Skipping Model Saving")

        training_console.subrule("Shutting Down")
        if cfg.training.wandb.log and wandb.run is not None:
            wandb.finish()
            training_console.print_notification("W&B Closed Gracefully")

        if run_status == "failed":
            training_console.print_error("❌ Training process experienced an error ❌")
        elif run_status == "stopped":
            training_console.print_warning("🚫 Training process was stopped manually 🚫")
        elif run_status == "terminated":
            training_console.print_warning("🛑 Training process was stopped automatically 🛑")
        else:
            training_console.print_success("✨ Training process completed successfully ✨")
        training_console.update_progress_task("application", completed=14, description="Finished")
        training_console.progress_stop()
        time.sleep(0.1)
        training_console.end_live()
        if training_exception is not None:
            raise training_exception