import argparse
import math
import sys
import datetime
import pickle
import hashlib
import traceback
from traceback import FrameSummary

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import os
import time
from tqdm import tqdm
import wandb  # Optional for logging
from datasets import load_dataset
from model_arch import MLATransformer
from trainer import Trainer

# Console colors for better readability
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def header(text):
        return f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}"
    
    @staticmethod
    def info(text):
        return f"{Colors.BLUE}{text}{Colors.ENDC}"
    
    @staticmethod
    def success(text):
        return f"{Colors.GREEN}{text}{Colors.ENDC}"
    
    @staticmethod
    def warning(text):
        return f"{Colors.YELLOW}{text}{Colors.ENDC}"
    
    @staticmethod
    def error(text):
        return f"{Colors.RED}{text}{Colors.ENDC}"
    
    @staticmethod
    def highlight(text):
        return f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}"

# Model defaults
HIDDEN_DIM = 1152
NUM_LAYERS = 12
NUM_HEADS = 9
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
FF_DIM = 4608
MLA_LATENT_DIM = 288
MAX_SEQ_LENGTH = 512
ROPE_HEAD_DIM = HEAD_DIM // 4
COMPRESSED_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
KV_LATENT_DIM = MLA_LATENT_DIM
Q_LATENT_DIM = MLA_LATENT_DIM

# Training defaults
DROPOUT = 0.1
VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size
BATCH_SIZE = 1
GRAD_STEPS = 8
LEARNING_RATE = 5e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLA Transformer with cached datasets")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset path (e.g., wikitext)")
    parser.add_argument("--dataset-name", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument("--run-name", type=str, default="WikiText", help="The base name of the run in W&B")
    parser.add_argument("--cache-dir", type=str, default="dataset_cache", help="Directory to store cached datasets")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache before training")

    # Model arguments
    parser.add_argument("--seq-length", type=int, default=MAX_SEQ_LENGTH, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--grad-acc-steps", type=int, default=GRAD_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")

    # Training arguments
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save final model")
    parser.add_argument("--use-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--allow-amp-switchover", action="store_true", help="Allows the training run to switch over to mixed precision once it reaches stability")

    return parser.parse_args()

# Define RoPE dimension per head (d_h^R in paper)
# Let's use the user's original rope_dim calculation for this example


# Define compressed content head dimension (d_h^C)
 # e.g., 128 - 32 = 96

# Define latent dimensions (d_c for KV, d'_c for Q in paper)
# Let's assume they are the same for simplicity, using user's MLA_LATENT_DIM


# Training Dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load and tokenize data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize the full text
        tokens = tokenizer.encode(text)

        # Create examples with max_length tokens
        for i in range(0, len(tokens) - max_length, max_length // 2):  # 50% overlap
            self.examples.append(tokens[i:i + max_length])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {
            'input_ids': torch.tensor(tokens[:-1]),
            'labels': torch.tensor(tokens[1:])
        }

class HFDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, dataset_name=None, seq_length=MAX_SEQ_LENGTH, split="train"):
        """
        Initialize the dataset from Hugging Face.

        Args:
            tokenizer: Tokenizer to use for encoding text
            dataset_path: Dataset path (e.g., "wikitext")
            dataset_name: Dataset name (e.g., "wikitext-2-raw-v1")
            seq_length: Maximum sequence length
            split: Either "train" or "test"
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        print(Colors.header(f"\n{'='*50}"))
        print(Colors.header(f" Loading {split.upper()} Dataset: {dataset_path}/{dataset_name or ''}"))
        print(Colors.header(f"{'='*50}"))
        
        # Load dataset
        self.raw_dataset = load_dataset(dataset_path, dataset_name, trust_remote_code=True)
        self.split = split
        
        # Process raw dataset to create examples
        self.examples = []
        
        # Process each text item individually to avoid exceeding max length
        total_tokens = 0
        skipped_short = 0
        text_items = 0
        
        # Progress bar for dataset processing
        texts = self.raw_dataset[split]["text"]
        for text in tqdm(texts, desc=f"Processing {split} texts", unit="text"):
            if not text.strip():
                continue
                
            text_items += 1
            # Tokenize the current text item
            encodings = tokenizer(text, return_tensors="pt", truncation=False)
            input_ids = encodings.input_ids[0]
            total_tokens += len(input_ids)
            
            # Skip if this text is too short
            if len(input_ids) < 4:  # Need at least a few tokens
                skipped_short += 1
                continue
                
            # Split into examples with stride
            for i in range(0, max(1, len(input_ids) - seq_length), seq_length // 2):
                end_idx = min(i + seq_length, len(input_ids))
                if end_idx - i < seq_length // 4:  # Skip if too short
                    continue
                
                # Get the example with consistent length whenever possible
                if end_idx - i == seq_length:
                    # Full-length example, just clone it
                    self.examples.append(input_ids[i:end_idx].clone())
                else:
                    # This example is at the end of the text and shorter than seq_length
                    example = input_ids[i:end_idx].clone()
                    # Ensure all examples have at least 4 tokens for training
                    if len(example) >= 4:
                        self.examples.append(example)

        # Ensure we have at least one example
        if len(self.examples) == 0:
            print(Colors.error(f"Warning: No examples found in {split} set, creating a dummy example"))
            self.examples.append(torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id]))

        self.total_tokens = total_tokens

        # Print summary information
        print(Colors.success(f"\n✓ Loaded {split} dataset:"))
        print(Colors.info(f"  • Dataset: {dataset_path}/{dataset_name or ''}"))
        print(Colors.info(f"  • Text items processed: {text_items} (skipped {skipped_short} short items)"))
        print(Colors.info(f"  • Training examples: {Colors.highlight(f'{len(self.examples):,}')}"))
        print(Colors.info(f"  • Total tokens: {Colors.highlight(f'{total_tokens:,}')}"))
        print(Colors.info(f"  • Avg tokens per example: {total_tokens / max(1, len(self.examples)):.1f}"))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        
        # Create inputs and labels for causal language modeling
        # Input: all tokens except the last one
        # Labels: all tokens except the first one
        return {
            "input_ids": input_ids[:-1],
            "labels": input_ids[1:]
        }


class CachedHFDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, dataset_name=None, seq_length=2048, split="train",
                 cache_dir="dataset_cache"):
        """
        Initialize the dataset from Hugging Face with caching for tokenized examples.

        Args:
            tokenizer: Tokenizer to use for encoding text
            dataset_path: Dataset path (e.g., "wikitext")
            dataset_name: Dataset name (e.g., "wikitext-2-raw-v1")
            seq_length: Maximum sequence length
            split: Either "train" or "test"
            cache_dir: Directory to store cached tokenized datasets
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.cache_dir = cache_dir

        # Create a unique cache key based on tokenizer, dataset, and sequence length
        # This ensures we regenerate cache if any of these change
        tokenizer_name = tokenizer.__class__.__name__
        self.vocab_size = len(tokenizer)
        cache_key = f"{tokenizer_name}_{self.vocab_size}_{dataset_path}_{dataset_name or ''}_{split}_{seq_length}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        self.cache_file = os.path.join(cache_dir, f"{cache_hash}.pkl")

        print(f"\n{'=' * 50}")
        print(f" Dataset: {dataset_path}/{dataset_name or ''} ({split})")
        print(f"{'=' * 50}")

        # Try to load from cache first
        self.examples = self._load_from_cache()
        was_cached = self._was_cached()

        # If not in cache, process the dataset and save to cache
        if self.examples is None:
            from datasets import load_dataset

            print(f"Cache miss. Loading and tokenizing dataset...")
            self.raw_dataset = load_dataset(dataset_path, dataset_name, trust_remote_code=True)
            self.examples = []
            self.total_tokens = 0

            # Process each text item to create examples
            skipped_short = 0
            text_items = 0

            # Progress bar for dataset processing
            texts = self.raw_dataset[split]["text"]
            for text in tqdm(texts, desc=f"Processing {split} texts", unit="text"):
                if not text.strip():
                    continue

                text_items += 1
                # Tokenize the current text item
                encodings = tokenizer(text, return_tensors="pt", truncation=False)
                input_ids = encodings.input_ids[0]
                self.total_tokens += len(input_ids)

                # Skip if this text is too short
                if len(input_ids) < 4:  # Need at least a few tokens
                    skipped_short += 1
                    continue

                # Split into examples with stride
                for i in range(0, max(1, len(input_ids) - seq_length), seq_length // 2):
                    end_idx = min(i + seq_length, len(input_ids))
                    if end_idx - i < seq_length // 4:  # Skip if too short
                        continue

                    # Get the example with consistent length whenever possible
                    if end_idx - i == seq_length:
                        # Full-length example, just clone it
                        self.examples.append(input_ids[i:end_idx].clone())
                    else:
                        # This example is at the end of the text and shorter than seq_length
                        example = input_ids[i:end_idx].clone()
                        # Ensure all examples have at least 4 tokens for training
                        if len(example) >= 4:
                            self.examples.append(example)

            # Ensure we have at least one example
            if len(self.examples) == 0:
                print(f"Warning: No examples found in {split} set, creating a dummy example")
                self.examples.append(torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id]))

            # Save processed examples to cache
            self._save_to_cache()

        # Print summary information
        print(f"\n✓ Loaded {split} dataset:")
        print(f"  • Dataset: {dataset_path}/{dataset_name or ''}")
        print(f"  • Split: {split}")
        print(f"  • Examples: {len(self.examples):,}")
        if hasattr(self, 'total_tokens'):
            print(f"  • Total tokens: {self.total_tokens:,}")
            print(f"  • Avg tokens per example: {self.total_tokens / max(1, len(self.examples)):.1f}")
        print(f"  • Sequence length: {seq_length}")
        if was_cached:
            print(f"  • Loaded from cache: Yes ✓")
        else:
            print(f"  • Saved to cache: Yes ✓")

    def _load_from_cache(self):
        """Try to load the dataset from cache file"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            return None

        if os.path.exists(self.cache_file):
            try:
                print(f"Loading cached dataset from {self.cache_file}...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.total_tokens = cache_data['total_tokens']
                    return cache_data['examples']
            except Exception as e:
                print(f"Error loading cache: {e}")
                return None
        return None

    def _save_to_cache(self):
        """Save the processed dataset to cache file"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Saving dataset to cache at {self.cache_file}...")
        with open(self.cache_file, 'wb') as f:
            cache_data = {
                'examples': self.examples,
                'total_tokens': self.total_tokens
            }
            pickle.dump(cache_data, f)

    def _was_cached(self):
        """Check if dataset was loaded from cache"""
        return os.path.exists(self.cache_file)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]

        # Create inputs and labels for causal language modeling
        # Input: all tokens except the last one
        # Labels: all tokens except the first one
        return {
            "input_ids": input_ids[:-1],
            "labels": input_ids[1:]
        }

def check_grad_enabled(model):
    no_grad_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            no_grad_params.append(name)

    if no_grad_params:
        print(Colors.error(f"❌ WARNING: The following parameters don't require gradients:"))
        for param in no_grad_params:
            print(Colors.error(f"  • {param}"))
        print(Colors.error(f"Training will not work properly. Exiting."))
        sys.exit(1)
    else:
        print(Colors.success(f"✓ All parameters require gradients - Ready for training!"))

# Main training script with improvements for 2070 Super
def main():
    args = parse_args()

    def display_frame_info(frame_info: FrameSummary):
        print(Colors.info(f"  File: {Colors.highlight(frame_info.filename)}"))
        print(Colors.info(f"  Line: {Colors.highlight(frame_info.lineno)}"))
        print(Colors.info(f"  Function: {Colors.highlight(frame_info.name)}"))
        print(Colors.info(f"  Code: {Colors.highlight(frame_info.line.strip() if frame_info.line else 'N/A')}"))

    def display_exception(exception: Exception, msg: str = "❌ Training failed with error"):
        print(Colors.error(f"\n{msg}: {exception}"))
        tb = exception.__traceback__
        if tb:

            stack_summary = traceback.extract_tb(tb)
            print(f"\nFull stack depth: {len(stack_summary)}")

            if len(stack_summary) > 0:
                print(Colors.error(f"\n  ⓘ Outermost Frame (Frame 0):"))
                frame = stack_summary[0]
                display_frame_info(frame)

            if len(stack_summary) > 1:
                print(Colors.error(f"\n  ⓘ Caller of Error Function (Frame -2):"))
                # -1 is where error occurred (func_z)
                # -2 is the caller (func_y)
                frame = stack_summary[-2]
                display_frame_info(frame)

            if len(stack_summary) > 0:
                print(Colors.error(f"\n  ⓘ Error Location (Frame -1):"))
                frame = stack_summary[-1]
                display_frame_info(frame)

    # Handle cache directory
    if args.clear_cache and os.path.exists(args.cache_dir):
        print(Colors.warning(f"Clearing cache directory: {args.cache_dir}"))
        for file in os.listdir(args.cache_dir):
            if file.endswith(".pkl"):
                os.remove(os.path.join(args.cache_dir, file))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=args.seq_length)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize variables
    trained_tokens = None
    run_status = "setup"

    # Print training configuration
    print(Colors.header(f"\n{'=' * 50}"))
    print(Colors.header(f" Training Configuration"))
    print(Colors.header(f"{'=' * 50}"))
    print(Colors.info(f"  • Dataset: {Colors.highlight(f'{args.dataset}/{args.dataset_name}')}"))
    print(Colors.info(f"  • Sequence Length: {Colors.highlight(str(args.seq_length))}"))
    print(Colors.info(f"  • Batch Size: {Colors.highlight(str(args.batch_size))} " +
                      f"(effective: {Colors.highlight(str(args.batch_size * args.grad_acc_steps))})"))
    print(Colors.info(f"  • Learning Rate: {Colors.highlight(str(args.learning_rate))}"))
    print(Colors.info(f"  • Using Cache: {Colors.highlight('No' if args.no_cache else 'Yes')}"))

    try:
        # Create cached datasets
        if args.no_cache:
            train_dataset = HFDataset(
                tokenizer=tokenizer,
                dataset_path=args.dataset,
                dataset_name=args.dataset_name,
                seq_length=args.seq_length,
                split="train"
            )

            test_dataset = HFDataset(
                tokenizer=tokenizer,
                dataset_path=args.dataset,
                dataset_name=args.dataset_name,
                seq_length=args.seq_length,
                split="test"
            )
        else:
            # Use the cached dataset implementation
            train_dataset = CachedHFDataset(
                tokenizer=tokenizer,
                dataset_path=args.dataset,
                dataset_name=args.dataset_name,
                seq_length=args.seq_length,
                split="train",
                cache_dir=args.cache_dir
            )

            test_dataset = CachedHFDataset(
                tokenizer=tokenizer,
                dataset_path=args.dataset,
                dataset_name=args.dataset_name,
                seq_length=args.seq_length,
                split="test",
                cache_dir=args.cache_dir
            )
    except Exception as e:
        display_exception(exception=e, msg="💥 Error processing datasets")
        run_status = "failed"
        train_dataset = None
        test_dataset = None

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

    try:
        if run_status == "setup":
            # Create data loaders
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,  # No multiprocessing to save memory
                collate_fn=collate_fn
            )

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
        else:
            print(Colors.warning(f"\n  🚫 Data loaders skipped due to previous error"))
            train_dataloader = None
            test_dataloader = None
    except Exception as e:
        display_exception(exception=e, msg="💥 Error creating data loaders")
        run_status = "failed"
        train_dataloader = None
        test_dataloader = None

    try:
        if run_status == "setup":
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Print GPU info
            print(Colors.header(f"\n{'='*50}"))
            print(Colors.header(f" Hardware Configuration"))
            print(Colors.header(f"{'='*50}"))

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(Colors.success(f"✓ GPU detected: {Colors.highlight(gpu_name)}"))
                print(Colors.info(f"  • Total VRAM: {Colors.highlight(f'{total_vram:.2f} GB')}"))
            else:
                print(Colors.warning("⚠ No GPU detected! Training will be very slow on CPU."))

            print(Colors.header(f"\n{'='*50}"))
            print(Colors.header(f" Model Configuration"))
            print(Colors.header(f"{'='*50}"))

            # Create model instance
            model = MLATransformer(
                vocab_size=train_dataset.vocab_size,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                kv_latent_dim=KV_LATENT_DIM,
                q_latent_dim=Q_LATENT_DIM,
                dropout=DROPOUT,
                max_seq_len=args.seq_length,
                rope_head_dim=ROPE_HEAD_DIM,
                use_checkpointing=args.use_checkpointing
            )
            model.to(device)
        else:
            print(Colors.warning(f"\n  🚫 Model configuration skipped due to previous error"))
            model = None
            device = None
    except Exception as e:
        display_exception(exception=e, msg="❌ Unable to load model")
        run_status = "failed"
        model = None
        device = None

    try:
        if run_status == "setup":
            # Track model parameters and memory usage
            total_params = sum(p.numel() for p in model.parameters())
            param_size_mb = total_params * 4 / (1024 ** 2)

            print(Colors.info(f"  • Architecture: Multi-head Latent Attention Transformer"))
            print(Colors.info(f"  • Hidden dimension: {Colors.highlight(str(HIDDEN_DIM))}"))
            print(Colors.info(f"  • Attention heads: {Colors.highlight(str(NUM_HEADS))}"))
            print(Colors.info(f"  • Layers: {Colors.highlight(str(NUM_LAYERS))}"))
            print(Colors.info(f"  • Latent dimension: {Colors.highlight(str(MLA_LATENT_DIM))}"))
            print(Colors.info(f"  • Parameters: {Colors.highlight(f'{total_params:,}')}"))
            print(Colors.info(f"  • Model size: {Colors.highlight(f'{param_size_mb:.2f} MB')}"))

            print(Colors.header(f"\n{'='*50}"))
            print(Colors.header(f" Training Configuration"))
            print(Colors.header(f"{'='*50}"))

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            # Initialize optimizer with weight decay and 8-bit precision
            #try:
                # Try to use 8-bit Adam if available (reduces optimizer memory by 75%)
                #from bitsandbytes.optim import Adam8bit
                #optimizer = Adam8bit(optimizer_grouped_parameters, lr=learning_rate, weight_decay=0.01)
                #print(Colors.success(f"✓ Using 8-bit Adam optimizer for memory efficiency"))
            #except ImportError:
                # Fall back to regular AdamW
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
            print(Colors.warning(f"\n  ⚠️ Using regular AdamW optimizer (8-bit not available)"))
        else:
            print(Colors.warning(f"\n  🚫 Optimizer configuration skipped due to previous error"))
            optimizer = None
            total_params = 0
    except Exception as e:
        display_exception(exception=e, msg="❌ Unable to create optimizer")
        run_status = "failed"
        optimizer = None
        total_params = 0

    if run_status == "setup":
        # Learning rate scheduler
        target_tokens = total_params * 10
        total_steps = target_tokens // (train_dataset.total_tokens / (len(train_dataloader) // args.grad_acc_steps))
        eval_interval = 100

        # Training configuration info
        print(Colors.info(f"  • Batch size: {Colors.highlight(str(args.batch_size))} (effective: {Colors.highlight(str(args.batch_size * args.grad_acc_steps))})"))
        print(Colors.info(f"  • Learning rate: {Colors.highlight(str(args.learning_rate))}"))
        print(Colors.info(f"  • Gradient accumulation steps: {Colors.highlight(str(args.grad_acc_steps))}"))
        print(Colors.info(f"  • Estimated total optimizer steps: {Colors.highlight(f'{total_steps:,}')}"))
        print(Colors.info(f"  • Target training tokens: {Colors.highlight(f'{target_tokens:,}')}"))

        # Measure the actual tokens per second using a warm-up phase
        print(Colors.header(f"\n{'='*50}"))
        print(Colors.header(f" Measuring Performance"))
        print(Colors.header(f"{'='*50}"))

        print(Colors.info("  • Running warm-up iterations to measure tokens per second..."))
    else:
        target_tokens = 0
        total_steps = 0
        eval_interval = None

    try:
        if run_status == "setup":
            # Do a few warm-up steps to measure performance
            model.train()

            # Create small dataloader with a few batches for measurement
            measure_batch_size = args.batch_size
            measure_dataset = torch.utils.data.Subset(train_dataset, list(range(min(20, len(train_dataset)))))
            measure_dataloader = DataLoader(
                measure_dataset,
                batch_size=measure_batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

            # Run a few iterations and measure speed
            start_time = time.time()
            total_tokens_processed = 0

            with torch.no_grad():  # No need for gradients during measurement
                for batch in measure_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Count non-padding tokens
                    actual_tokens = attention_mask.sum().item()
                    total_tokens_processed += actual_tokens

                    # Forward pass only (no backprop needed for measurement)
                    _ = model(input_ids, attention_mask=attention_mask)

                    # Break after processing a few batches
                    if total_tokens_processed > 500000:  # Enough for a good measurement
                        break

            measurement_time = time.time() - start_time
            measured_tokens_per_second = total_tokens_processed / measurement_time

            # Apply a conservative factor to account for backpropagation and later epoch slowdown
            estimated_tokens_per_second = measured_tokens_per_second * 0.55  # 55% of measured forward-only speed

            print(Colors.success(f"  • Measured forward pass speed: {Colors.highlight(f'{measured_tokens_per_second:.1f}')} tokens/sec"))
            print(Colors.success(f"  • Estimated training speed: {Colors.highlight(f'{estimated_tokens_per_second:.1f}')} tokens/sec"))

            # Calculate training time estimate
            estimated_hours = (target_tokens / estimated_tokens_per_second) / 3600

            # Format time estimate nicely
            if estimated_hours < 1:
                time_str = f"{estimated_hours * 60:.1f} minutes"
            else:
                days = int(estimated_hours // 24)
                hours = int(estimated_hours % 24)
                minutes = int((estimated_hours * 60) % 60)
                if days > 0:
                    time_str = f"{days}d {hours}h {minutes}m"
                else:
                    time_str = f"{hours}h {minutes}m"

            print(Colors.info(f"  • Estimated training time: {Colors.highlight(time_str)}"))
        else:
            print(Colors.warning(f"\n  🚫 Training time estimation skipper due to previous error"))
    except Exception as e:
        display_exception(exception=e, msg="❌ Unable to estimate training time")
        run_status = "failed"

    try:
        if run_status == "setup":
            # Calculate warmup steps (e.g., 8% of total steps)
            warmup_steps = int(0.08 * total_steps)

            # Create scheduler
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            # Memory usage
            print(Colors.header(f"\n{'='*50}"))
            print(Colors.header(f" Memory Usage"))
            print(Colors.header(f"{'='*50}"))

            # Check gradient enabled for params
            check_grad_enabled(model)

            # Initialize wandb (optional)
            wandb.init(
                project="TabulaPrima",
                entity="jordan-ledoux-none",
                job_type="training",
                tags=["experiment","generic-dataset","pretraining"],
                config={
                    "job_name": args.run_name+"-"+datetime.datetime.now().strftime("%b%d"),
                    "model_def": {
                        "parameters": total_params,
                        "seq_length": args.seq_length,
                        "hidden_dim": HIDDEN_DIM,
                        "num_heads": NUM_HEADS,
                        "num_layers": NUM_LAYERS,
                        "head_dim": HEAD_DIM,
                        "ff_dim": FF_DIM,
                        "latent_dim": MLA_LATENT_DIM,
                        "rope_head_dim": ROPE_HEAD_DIM,
                    },
                    "training_def": {
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "gradient_accumulation_steps": args.grad_acc_steps,
                        "optimizer": "Adam8Bit" if "8bit" in optimizer.__class__.__name__ else "AdamW",
                        "dropout": DROPOUT,
                        "grad_checkpoints": args.use_checkpointing,
                        "allow_amp_switch": args.allow_amp_switchover,
                        "target_tokens": target_tokens,
                    },
                },
                name=args.run_name+"-"+datetime.datetime.now().strftime("%b%d").upper(),
            )
            """
            wandb.watch(
                model,
                log="all",
                log_freq=100
            )
            """
        else:
            print(Colors.warning(f"\n  🚫 Scheduler configuration skipped due to previous error"))
            scheduler = None
    except Exception as e:
        display_exception(exception=e, msg="❌ Unable to create scheduler")
        run_status = "failed"
        scheduler = None

    train_start_time = time.time()

    # Start training
    try:
        if run_status == "setup":
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                total_parameters=total_params,
                gradient_accumulation_steps=args.grad_acc_steps,
                checkpoint_dir="checkpoints",
                use_amp=False,
                eval_interval=eval_interval,
                global_steps=total_steps,
                wandb=wandb,
                allow_amp_switch=args.allow_amp_switchover,
            )
            run_status = "training"
            trained_tokens, run_status = trainer.run()
            if run_status == "failed":
                wandb.run.status = run_status
        else:
            print(Colors.warning(f"\n  🚫 Training skipped due to previous error"))
    except KeyboardInterrupt:
        print(Colors.warning("\n⚠ Training interrupted by user"))
        wandb.alert(title="Training interrupted", text="Training interrupted by user")
        wandb.run.status = run_status = "stopped"
    except Exception as e:
        display_exception(exception=e, msg="❌ Training failed with error")
        wandb.alert(title="Training failed", text=f"An error occurred during training: {e}")
        wandb.run.status = run_status = "failed"
    finally:
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

        if wandb.run is not None:
            if model:
                # Log comprehensive run summary
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
                    "run/dataset_size_tokens": train_dataset.total_tokens if hasattr(train_dataset, "total_tokens") else 0,
                    "run/dataset_size_examples": len(train_dataset),

                    # Hardware information
                    "run/gpu_name": gpu_name,
                    "run/gpu_memory_gb": memory_gb,
                    "run/gpu_utilization_pct": gpu_utilization,
                    "run/memory_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0,
                    "run/memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3) if torch.cuda.is_available() else 0,
                })

                # Create a final summary table
                if wandb.run is not None:
                    columns = ["Metric", "Value"]
                    data = [
                        ["Total Training Time", f"{total_training_hours:.2f} hours"],
                        ["Average Tokens/Second", f"{avg_tokens_per_second:.2f}"],
                        ["Total Tokens Processed", f"{total_tokens:,}"],
                        ["Model Parameters", f"{sum(p.numel() for p in model.parameters()):,}"],
                        ["GPU Utilization", f"{gpu_utilization:.1f}%"],
                    ]
                    wandb.log({"run/summary_table": wandb.Table(columns=columns, data=data)})

            wandb.finish()

    print(Colors.header(f"\n{'='*50}"))
    print(Colors.header(f" Training Complete"))
    print(Colors.header(f"{'='*50}"))

    # Save final model
    if run_status != "failed":
        final_model_path = os.path.join("models", f"tabula_prima_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
        torch.save(model, final_model_path)
        print(Colors.success(f"✓ Saved final model to: {final_model_path}"))

    if run_status == "failed":
        print(Colors.error(f"\n❌ Training process experienced an error ❌"))
    elif run_status == "stopped":
        print(Colors.warning(f"\n🚫 Training process was stopped manually 🚫"))
    elif run_status == "terminated":
        print(Colors.warning(f"\n🛑 Training process was stopped automatically 🛑"))
    else:
        print(Colors.success(f"\n✨ Training process completed successfully ✨"))


# TensorBoard callback for monitoring (alternative to Wandb)
class TensorBoardCallback:
    def __init__(self, log_dir="tensorboard_logs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            self.log_dir = log_dir
            self.enabled = True
        except ImportError:
            print("TensorBoard not available, install with 'pip install tensorboard'")
            self.enabled = False

    def log(self, metrics, step):
        if not self.enabled:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        if self.enabled:
            self.writer.close()


if __name__ == "__main__":
    main()