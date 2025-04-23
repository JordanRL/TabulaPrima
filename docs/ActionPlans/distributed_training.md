# Implementation Plan for Multi-GPU Training in TabulaPrima

## Overview

This document outlines a comprehensive implementation plan for adding multi-GPU training capabilities to the TabulaPrima project. The plan addresses two specific tasks from the improvement list:

1. **Implement model parallelism for training larger models across multiple GPUs**
2. **Implement distributed training with DeepSpeed or FSDP**

The implementation will leverage PyTorch's native distributed capabilities along with popular libraries for efficient large-scale model training.

## Current System Analysis

Based on the examination of the existing codebase:

- **Model Architecture**: TabulaPrima uses an MLATransformer (Multi-head Latent Attention) architecture implemented in `model_arch.py`. The model follows a standard transformer structure but uses a specialized attention mechanism with latent representations.

- **Training System**: The training loop is implemented in `trainer.py`, which handles optimization, checkpointing, evaluation, and logging. The `train_harness.py` orchestrates the entire training process including data loading and model initialization.

- **Current Limitations**: The code currently only supports single-GPU training. It lacks distributed data loading, gradient synchronization, and model sharding capabilities necessary for multi-GPU training.

## Implementation Plan 1: Model Parallelism

Model parallelism will allow dividing large models across multiple GPUs when a single GPU cannot hold the entire model.

### 1.1 Add Tensor Parallelism Support

Tensor parallelism splits individual layers across multiple GPUs, particularly useful for attention and feed-forward layers.

```python
# New file: model_parallel.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class ParallelLinear(torch.nn.Module):
    """Linear layer implementation for tensor parallelism."""
    def __init__(self, in_features, out_features, process_group=None):
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)
        self.rank = dist.get_rank(self.process_group)
        
        # Split output features across GPUs
        self.out_features_per_gpu = out_features // self.world_size
        self.linear = torch.nn.Linear(in_features, self.out_features_per_gpu)
        
    def forward(self, x):
        # Local computation
        local_output = self.linear(x)
        
        # Gather outputs from all GPUs
        gathered_output = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(gathered_output, local_output, self.process_group)
        
        # Concatenate along feature dimension
        return torch.cat(gathered_output, dim=-1)
```

### 1.2 Modify MLATransformer for Tensor Parallelism

Update the model architecture to support splitting attention heads and feed-forward networks across GPUs:

```python
# Modifications to model_arch.py

class ParallelMultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, kv_latent_dim, q_latent_dim, rope_head_dim, 
                 dropout, max_seq_len, use_fusion=False, process_group=None):
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group) if process_group else 1
        self.rank = dist.get_rank(process_group) if process_group else 0
        
        # Assign a subset of attention heads to each GPU
        self.num_local_heads = num_heads // self.world_size
        if num_heads % self.world_size != 0:
            raise ValueError(f"Number of heads ({num_heads}) must be divisible by world size ({self.world_size})")
        
        # Rest of initialization with self.num_local_heads instead of num_heads
        # ...
        
    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, is_causal=True, use_cache=False, sin_cos=None):
        # Local attention computation
        # ...
        
        # All-gather across GPUs to get the full context
        # ...
```

### 1.3 Implement Pipeline Parallelism

Pipeline parallelism will partition the model vertically by layers, allowing very deep models:

```python
# New file: pipeline_parallel.py

import torch
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe

def convert_model_to_pipe(model, num_gpus):
    """Convert an MLATransformer model to use pipeline parallelism."""
    # Split the model into equal segments
    num_layers = len(model.layers)
    layers_per_gpu = num_layers // num_gpus
    
    # Create balanced partitions
    partitions = []
    for i in range(num_gpus):
        start_idx = i * layers_per_gpu
        end_idx = (i + 1) * layers_per_gpu if i < num_gpus - 1 else num_layers
        
        # Create a sequential module for this partition
        partition = torch.nn.Sequential(
            model.embedding if i == 0 else torch.nn.Identity(),
            *model.layers[start_idx:end_idx],
            model.norm if i == num_gpus - 1 else torch.nn.Identity(),
            model.lm_head if i == num_gpus - 1 else torch.nn.Identity()
        )
        partitions.append(partition)
    
    # Create a Pipe model
    pipe_model = Pipe(torch.nn.Sequential(*partitions), chunks=8)
    return pipe_model
```

### 1.4 Add Device Placement Utilities

Create utilities to manage device placement for different parallelism strategies:

```python
# New file: device_utils.py

import torch
import torch.distributed as dist

def initialize_model_parallel(tensor_parallel_size=1, pipeline_parallel_size=1):
    """
    Initialize the necessary process groups for model parallelism.
    
    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of GPUs for pipeline parallelism
    
    Returns:
        Tuple of process groups (tensor_pg, pipeline_pg)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    if tensor_parallel_size * pipeline_parallel_size > world_size:
        raise ValueError(f"Requested parallelism ({tensor_parallel_size}x{pipeline_parallel_size}) exceeds available GPUs ({world_size})")
    
    # Create process groups
    ranks = list(range(world_size))
    tensor_groups = [ranks[i:i+tensor_parallel_size] for i in range(0, world_size, tensor_parallel_size)]
    pipeline_groups = []
    
    for i in range(tensor_parallel_size):
        pg = [tensor_groups[j][i] for j in range(pipeline_parallel_size)]
        pipeline_groups.append(pg)
    
    # Create actual process groups
    tensor_pg = dist.new_group(tensor_groups[dist.get_rank() // tensor_parallel_size])
    pipeline_pg = dist.new_group(pipeline_groups[dist.get_rank() % tensor_parallel_size])
    
    return tensor_pg, pipeline_pg
```

### 1.5 Update Training Loop for Model Parallelism

Modify the `Trainer` class to support model parallelism:

```python
# Modifications to trainer.py

class ModelParallelTrainer(Trainer):
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, scheduler, device, cfg, wandb_instance=None, tensor_parallel=False, pipeline_parallel=False):
        super().__init__(model, train_dataloader, test_dataloader, optimizer, scheduler, device, cfg, wandb_instance)
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        
    def _forward(self, input_ids, labels, attention_mask):
        # Handle different forward pass based on parallelism strategy
        if self.pipeline_parallel:
            # Pipeline parallel forward pass needs to handle microbatches
            # ...
        else:
            # Regular or tensor parallel forward pass
            return super()._forward(input_ids, labels, attention_mask)
```

## Implementation Plan 2: Distributed Training with DeepSpeed/FSDP

Implementing distributed training with DeepSpeed or FSDP will enable efficient training across multiple GPUs with minimal code changes.

### 2.1 DeepSpeed Integration

#### 2.1.1 Add DeepSpeed Configuration

Create a config template for DeepSpeed:

```json
// New file: configs/deepspeed_config.json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "weight_decay": 0.01,
      "bias_correction": true
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 0.001,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true,
    "auto_cast": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8
  }
}
```

#### 2.1.2 Create DeepSpeed Trainer

Implement a DeepSpeed-compatible trainer:

```python
# New file: deepspeed_trainer.py

import os
import torch
import deepspeed
from trainer import Trainer, TrainingState

class DeepSpeedTrainer(Trainer):
    def __init__(self, model, train_dataloader, test_dataloader, device, cfg, wandb_instance=None):
        # Initialize without optimizer and scheduler as DeepSpeed will handle those
        super().__init__(model, train_dataloader, test_dataloader, None, None, device, cfg, wandb_instance)
        
        # Load DeepSpeed config
        ds_config_path = os.path.join(os.path.dirname(__file__), "configs/deepspeed_config.json")
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config_path,
            training_data=train_dataloader.dataset
        )
        
        self.model = model_engine
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_deepspeed = True
        
        # Update training state for DeepSpeed
        self.training_state.use_amp = self.model.fp16_enabled()
        
    def _bidirectional(self):
        # DeepSpeed handles gradients differently
        logits, loss = self._forward(
            input_ids=self.input_ids,
            labels=self.labels,
            attention_mask=self.attention_mask
        )
        
        # DeepSpeed handles loss scaling internally
        self.model.backward(loss)
        return loss
        
    def _backprop(self, grad_clip_value):
        # DeepSpeed handles gradient clipping and optimizer steps
        self.model.step()
```

#### 2.1.3 Update Training Harness to Support DeepSpeed

Modify `train_harness.py` to initialize distributed training with DeepSpeed:

```python
# Modifications to train_harness.py

def initialize_distributed():
    """Initialize distributed training environment."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    return world_size, local_rank, device

def run_training(cfg: Config):
    # ... existing code ...
    
    # Initialize distributed if using DeepSpeed or FSDP
    if cfg.training.distributed_backend in ["deepspeed", "fsdp"]:
        world_size, local_rank, device = initialize_distributed()
        training_console.print_notification(f"Initialized distributed training with world size {world_size}, local rank {local_rank}")
    else:
        # Original single-GPU code
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ... existing code for data loading ...
    
    # Choose the appropriate trainer based on config
    if cfg.training.distributed_backend == "deepspeed":
        from deepspeed_trainer import DeepSpeedTrainer
        trainer = DeepSpeedTrainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            cfg=cfg.training,
            wandb_instance=wandb if cfg.training.wandb.log else None
        )
    elif cfg.training.distributed_backend == "fsdp":
        # Use FSDP trainer (implemented below)
        from fsdp_trainer import FSDPTrainer
        trainer = FSDPTrainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg.training,
            wandb_instance=wandb if cfg.training.wandb.log else None
        )
    else:
        # Original trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cfg=cfg.training,
            wandb_instance=wandb if cfg.training.wandb.log else None
        )
    
    # ... rest of the function ...
```

### 2.2 FSDP Integration

#### 2.2.1 Create FSDP Wrapper Utilities

Add utilities to wrap model components with FSDP:

```python
# New file: fsdp_utils.py

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from model_arch import TransformerLayer, FeedForward, MultiHeadLatentAttention

def get_mixed_precision_policy():
    """Return mixed precision policy based on hardware capabilities."""
    if torch.cuda.is_bf16_supported():
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    else:
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )

def wrap_model_with_fsdp(model, device):
    """Wrap a model with FSDP."""
    # Define auto-wrap policy for transformer layers
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={
            TransformerLayer,
            FeedForward,
            MultiHeadLatentAttention
        }
    )
    
    # Configure FSDP wrapping
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=get_mixed_precision_policy(),
        device_id=device.index,
        cpu_offload=CPUOffload(offload_params=True),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )
    
    return fsdp_model
```

#### 2.2.2 Create FSDP Trainer

Implement an FSDP-compatible trainer:

```python
# New file: fsdp_trainer.py

import torch
from trainer import Trainer, TrainingState
from fsdp_utils import wrap_model_with_fsdp
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig,
)

class FSDPTrainer(Trainer):
    def __init__(self, model, train_dataloader, test_dataloader, optimizer, scheduler, device, cfg, wandb_instance=None):
        # Initialize with base trainer
        super().__init__(model, train_dataloader, test_dataloader, None, None, device, cfg, wandb_instance)
        
        # Wrap model with FSDP
        self.model = wrap_model_with_fsdp(model, device)
        
        # Create optimizer after wrapping with FSDP
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Initialize scheduler with FSDP-wrapped optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=cfg.target_tokens_per_param * sum(p.numel() for p in self.model.parameters()),
            eta_min=cfg.min_learning_rate
        )
        
        # Update training state for FSDP
        self.training_state.use_amp = True
        
    def save_checkpoint(self):
        """Save FSDP model checkpoint with state dict handling."""
        self.console.rule(f"Saving FSDP Checkpoint", style="highlight")
        
        # Configure FSDP state dict settings
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        # Save only on rank 0
        if torch.distributed.get_rank() == 0:
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config, optim_state_dict_config
            ):
                # Get state dicts
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = FSDP.optim_state_dict(self.model, self.optimizer)
                
                # Save checkpoint
                checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"fsdp_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
                torch.save({
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer_state_dict,
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "training_state_state_dict": self.training_state.state_dict(),
                    "trainer_state_dict": self.state_dict(),
                    "wandb_run_id": self.wandb.run.id if self.wandb is not None else None,
                }, checkpoint_path)
                self.console.print_complete(f"FSDP Checkpoint saved to file {checkpoint_path}")
        
        # Synchronize all processes
        torch.distributed.barrier()
```

### 2.3 Update Configuration Schema

Update `config_schema.py` to include new configuration options for distributed training:

```python
# Modifications to config_schema.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class DistributedBackend(Enum):
    NONE = "none"
    DEEPSPEED = "deepspeed"
    FSDP = "fsdp"
    MODEL_PARALLEL = "model_parallel"

@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    backend: DistributedBackend = DistributedBackend.NONE
    # DeepSpeed specific
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False
    # FSDP specific
    checkpoint_activation: bool = False
    cpu_offload: bool = False
    # Model parallel specific
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

@dataclass
class TrainingConfig:
    # Existing fields...
    
    # Add distributed configuration
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
```

### 2.4 Modify Data Loading for Distributed Training

Update data loading to support distributed training:

```python
# Modifications to train_harness.py

from torch.utils.data.distributed import DistributedSampler

def create_data_loaders(dataset_train, dataset_test, cfg, world_size=1, rank=0):
    """Create data loaders with proper distributed samplers if needed."""
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            dataset_test,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=cfg.training.batch_size,
            sampler=train_sampler,
            num_workers=cfg.dataset.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=1,
            sampler=test_sampler,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        # Original single-GPU data loaders
        train_dataloader = DataLoader(
            dataset_train,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            collate_fn=collate_fn
        )
        
        test_dataloader = DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
    
    return train_dataloader, test_dataloader
```

## Implementation Order and Timeline

### Phase 1: Preparation and Basic Integration (2 weeks)

1. Update configuration schema to include distributed training options
2. Set up distributed initialization in the training harness
3. Modify data loading to use DistributedSampler
4. Create basic distributed wrappers and utilities

### Phase 2: FSDP Implementation (2 weeks)

1. Create FSDP utilities and wrappers
2. Implement FSDP trainer
3. Add checkpoint saving/loading for FSDP
4. Test and optimize FSDP implementation

### Phase 3: DeepSpeed Integration (2 weeks)

1. Create DeepSpeed configuration file
2. Implement DeepSpeed trainer
3. Update training harness to support DeepSpeed
4. Test and optimize DeepSpeed implementation

### Phase 4: Model Parallelism Implementation (3 weeks)

1. Implement tensor parallelism utilities
2. Modify model architecture to support tensor parallelism
3. Implement pipeline parallelism
4. Create device placement utilities
5. Update training loop for model parallelism
6. Test and optimize model parallelism implementation

### Phase 5: Testing and Refinement (1 week)

1. Create benchmarking utilities for multi-GPU training
2. Compare different distributed strategies
3. Fine-tune configurations for optimal performance
4. Document distributed training usage

## Validation and Testing

To validate the implementation:

1. **Functional Testing**: Ensure that models converge with the same or better loss curves compared to single-GPU training.

2. **Performance Testing**: Measure throughput (tokens/second) and memory usage with different configurations.

3. **Scalability Testing**: Verify that training scales efficiently with the number of GPUs.

4. **Recovery Testing**: Test checkpoint saving and loading in distributed settings.

## Documentation Updates

1. Update README with distributed training instructions
2. Add example configuration files for different distributed strategies
3. Create a separate guide for scaling models and choosing the right distributed strategy
4. Document the tradeoffs between different approaches

## Conclusion

This implementation plan provides a comprehensive approach to enabling multi-GPU training in TabulaPrima. By implementing both model parallelism and distributed training with DeepSpeed/FSDP, the project will be able to handle much larger models and datasets efficiently. The modular approach allows for flexibility in choosing the most appropriate strategy based on model size, available hardware, and training requirements.