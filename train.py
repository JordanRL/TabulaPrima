import argparse
import json
import math
import sys
import datetime
import pickle
import hashlib
import traceback
from traceback import FrameSummary

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.align import Align
from rich.panel import Panel
from rich.table import Column
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, PreTrainedTokenizerFast
import os
import time
from tqdm import tqdm
import wandb
from datasets import load_dataset
from model_arch import MLATransformer
from trainer import Trainer
from rich.console import Console
from rich.progress import Progress, TextColumn, TimeRemainingColumn, TimeElapsedColumn, TaskProgressColumn, BarColumn, \
    SpinnerColumn
from pyfiglet import Figlet
from colour import Color

# Console colors for better readability
class Colors:
    HEADER = '#B87FD9'
    BLUE = '#61AFEF'
    CYAN = '#56B6C2'
    GREEN = '#98C379'
    YELLOW = '#E5C07B'
    RED = '#E06C75'
    BOLD = 'bold'
    UNDERLINE = 'underline'
    
    @staticmethod
    def header(text):
        return f"[{Colors.HEADER} {Colors.BOLD}]{text}[/{Colors.HEADER} {Colors.BOLD}]"
    
    @staticmethod
    def info(text):
        return f"[{Colors.BLUE}]{text}[/{Colors.BLUE}]"
    
    @staticmethod
    def success(text):
        return f"[{Colors.GREEN}]{text}[/{Colors.GREEN}]"
    
    @staticmethod
    def warning(text):
        return f"[{Colors.YELLOW}]{text}[/{Colors.YELLOW}]"
    
    @staticmethod
    def error(text):
        return f"[{Colors.RED}]{text}[/{Colors.RED}]"
    
    @staticmethod
    def highlight(text):
        return f"[{Colors.CYAN} {Colors.BOLD}]{text}[/{Colors.CYAN} {Colors.BOLD}]"

    @staticmethod
    def bold(text):
        return f"[{Colors.BOLD}]{text}[/{Colors.BOLD}]"

    @staticmethod
    def underline(text):
        return f"[{Colors.UNDERLINE}]{text}[/{Colors.UNDERLINE}]"

    @staticmethod
    def apply_gradient(text, color_top, color_bottom, padding_top=False, padding_bottom=False):
        # 1. Split the string into a list of lines
        lines = text.splitlines()

        # 1a. Filter out empty or whitespace-only lines BEFORE counting
        #     We keep the original line content, but filter based on the stripped version
        non_empty_lines = [line for line in lines if line.strip()]

        # 2. Count the number of NON-EMPTY lines for the gradient
        num_lines = len(non_empty_lines)

        # Handle edge case: no non-empty lines found after filtering
        if num_lines == 0:
            # Requirement is to add empty line top/bottom, even if no content
            return "\n\n"

        # Handle edge case: only one non-empty line
        if num_lines == 1:
            start_color_hex_only = color_top.lstrip('#').upper()
            formatted_line = f"[#{start_color_hex_only}]{non_empty_lines[0]}[/#{start_color_hex_only}]"
            # Add empty line padding top and bottom
            return f"\n{formatted_line}\n"

        # --- Proceed if 2 or more non-empty lines ---

        # 3. Create the gradient based on the count of non-empty lines
        start_color = Color(color_top)
        end_color = Color(color_bottom)
        gradient_objects = list(start_color.range_to(end_color, num_lines))
        gradient_hex = [color.hex_l for color in gradient_objects]

        # 4. Apply the gradient to each NON-EMPTY line using BBCode style formatting
        formatted_lines = []
        for i, line in enumerate(non_empty_lines):
            line_hex_code = gradient_hex[i].lstrip('#').upper()
            formatted_line = f"[#{line_hex_code}]{line}[/#{line_hex_code}]"
            formatted_lines.append(formatted_line)

        # 5. Rejoin the formatted (non-empty) lines with newline characters
        final_result = "\n".join(formatted_lines)

        # 5a. Add one empty line padding at the top and bottom
        if padding_top:
            final_result = f"\n{final_result}"
        if padding_bottom:
            final_result = f"{final_result}\n"

        return final_result


# Default model definition params
# These will be overwritten after parsing arguments and loading the model def
model_def = {}
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
BATCH_SIZE = 1
GRAD_STEPS = 8
LEARNING_RATE = 5e-5
TOK_PER_PARAM = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLA Transformer with cached datasets")

    # Run arguments
    parser.add_argument("--run-name", type=str, default="WikiText", help="The base name of the run in W&B")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset path (e.g., wikitext)")
    parser.add_argument("--dataset-name", type=str, default="wikitext-103-raw-v1", help="Dataset name")
    parser.add_argument("--dataset-type", type=str, default="hf", help="The dataset type (e.g. hf)")
    parser.add_argument("--cache-dir", type=str, default="dataset_cache", help="Directory to store cached datasets")
    parser.add_argument("--no-cache", action="store_true", help="Disable dataset caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear existing cache before training")

    # Model definition and arguments
    parser.add_argument("--model-def", type=str, default="255m_params.json", help="Model definition file name in configs/model_defs/ directory")
    parser.add_argument("--seq-length", type=int, default=MAX_SEQ_LENGTH, help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--grad-acc-steps", type=int, default=GRAD_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save final model")
    parser.add_argument("--use-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--allow-amp-switchover", action="store_true", help="Allows the training run to switch over to mixed precision once it reaches stability")

    return parser.parse_args()


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
    def __init__(self, tokenizer: PreTrainedTokenizerFast, dataset_path, dataset_name=None, seq_length=2048, split="train",
                 cache_dir="dataset_cache", console: Console|None=None):
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
        self.console = console or Console()
        self.total_raw_tokens = 0
        self.total_tokens = 0
        self.effective_total_tokens = 0

        # Create a unique cache key based on tokenizer, dataset, and sequence length
        # This ensures we regenerate cache if any of these change
        tokenizer_name = tokenizer.__class__.__name__
        if tokenizer_name == "PreTrainedTokenizerFast":
            self.vocab_size = len(tokenizer)
        else:
            self.vocab_size = tokenizer.vocab_size
        cache_key = f"{tokenizer_name}_{self.vocab_size}_{dataset_path}_{dataset_name or ''}_{split}_{seq_length}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        self.cache_file = os.path.join(cache_dir, f"{cache_hash}.pkl")

        self.console.rule(Colors.highlight(f"Dataset: {dataset_path}/{dataset_name or ''} ({split})"), style=Colors.HEADER)

        # Try to load from cache first
        self.examples = self._load_from_cache()
        was_cached = self._was_cached()

        # If not in cache, process the dataset and save to cache
        if self.examples is None:
            from datasets import load_dataset

            self.console.print(Colors.info(f"Cache miss. Loading and tokenizing dataset..."))
            self.raw_dataset = load_dataset(dataset_path, dataset_name, trust_remote_code=True)
            self.examples = []

            # Process each text item to create examples
            skipped_short = 0
            skipped_long = 0
            text_items = 0

            # Progress bar for dataset processing
            texts = self.raw_dataset[split]["text"]

            spinner_col = SpinnerColumn(table_column=Column(max_width=3))
            desc_col = TextColumn(text_format="{task.description}", style=Colors.CYAN,
                                  table_column=Column(max_width=30))
            bar_col = BarColumn(bar_width=None, complete_style="#4B6BFF")
            pct_col = TaskProgressColumn(table_column=Column(max_width=10))
            time_elapsed_col = TimeElapsedColumn(table_column=Column(max_width=15))
            time_remaining_col = TimeRemainingColumn(table_column=Column(max_width=15))
            metadata_col = TextColumn(
                text_format="Total: {task.fields[tok_total]} Avg: {task.fields[avg]}",
                style=Colors.BLUE, table_column=Column(max_width=40))
            dataset_progress = Progress(
                spinner_col, desc_col, bar_col, pct_col, time_elapsed_col, time_remaining_col, metadata_col,
                console=self.console, transient=True, expand=True
            )
            tokenizing_task = dataset_progress.add_task(description=f"Tokenizing {split} dataset", total=len(texts), tok_total=0, avg=0)
            dataset_progress.start()
            for text in texts:
                if not text.strip():
                    dataset_progress.update(tokenizing_task, advance=1, tok_total=f"{self.total_tokens:,}", avg=f"{self.total_tokens/(text_items if text_items > 0 else 1):.2f}")
                    continue

                text_items += 1
                # Tokenize the current text item
                if tokenizer_name == "PreTrainedTokenizerFast":
                    encodings = tokenizer(text, return_tensors="pt", truncation=False, max_length=None)
                    input_ids = encodings.input_ids[0]
                else:
                    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
                self.total_tokens += len(input_ids)
                dataset_progress.update(tokenizing_task, advance=1, tok_total=f"{self.total_tokens:,}", avg=f"{self.total_tokens/text_items:.2f}")

                if len(input_ids) > self.seq_length:
                    skipped_long += 1
                    continue

                # Skip if this text is too short
                if len(input_ids) < 4:  # Need at least a few tokens
                    skipped_short += 1
                    continue

                # Split into examples with stride
                for i in range(0, max(1, len(input_ids) - seq_length), seq_length // 2):
                    end_idx = min(i + seq_length, len(input_ids))
                    if end_idx - i < seq_length // 4:  # Skip if too short
                        continue

                    self.effective_total_tokens += len(input_ids)
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

            dataset_progress.stop()

            # Ensure we have at least one example
            if len(self.examples) == 0:
                self.console.print(f"Warning: No examples found in {split} set, creating a dummy example")
                self.examples.append(torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id]))

            # Save processed examples to cache
            self._save_to_cache()

        # Print summary information
        self.console.print(f"\n  ✓ Loaded {split} dataset:")
        self.console.print(f"  • Dataset: {dataset_path}/{dataset_name or ''}")
        self.console.print(f"  • Split: {split}")
        self.console.print(f"  • Examples: {len(self.examples):,}")
        if hasattr(self, 'total_tokens'):
            self.console.print(f"  • Total tokens: {self.total_tokens:,}")
        if hasattr(self, 'effective_total_tokens'):
            self.console.print(f"  • Effective total tokens: {self.effective_total_tokens:,}")
            self.console.print(f"  • Avg tokens per example: {self.effective_total_tokens / max(1, len(self.examples)):.1f}")
        self.console.print(f"  • Sequence length: {seq_length}")
        if was_cached:
            self.console.print(f"  • Loaded from cache: Yes ✓")
        else:
            self.console.print(f"  • Saved to cache: Yes ✓")

    def _load_from_cache(self):
        """Try to load the dataset from cache file"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            return None

        if os.path.exists(self.cache_file):
            try:
                self.console.print(f"Loading cached dataset from {self.cache_file}...")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.total_tokens = cache_data['total_tokens']
                    self.effective_total_tokens = cache_data['effective_total_tokens']
                    return cache_data['examples']
            except Exception as e:
                self.console.print(f"Error loading cache: {e}")
                return None
        return None

    def _save_to_cache(self):
        """Save the processed dataset to cache file"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self.console.print(f"Saving dataset to cache at {self.cache_file}...")
        with open(self.cache_file, 'wb') as f:
            cache_data = {
                'examples': self.examples,
                'total_tokens': self.total_tokens,
                'effective_total_tokens': self.effective_total_tokens
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

# Main training script with improvements for 2070 Super
def main():
    training_console = Console(soft_wrap=True)
    fig_epic = Figlet(font='epic', width=160)
    title_panel = Panel(
        Align.center(Colors.apply_gradient(
            fig_epic.renderText('Tabula\n   Prima'),
            Colors.HEADER,
            Colors.BLUE,
            padding_top=True,
            padding_bottom=True
        )),
        title="Multi-head Latent Attention LLM",
        subtitle="Pre-Training Utility",
        border_style=Colors.HEADER,
    )
    training_console.print(title_panel)

    args = parse_args()

    # Load model definition from JSON file
    global model_def, HIDDEN_DIM, NUM_LAYERS, NUM_HEADS, HEAD_DIM, FF_DIM, MLA_LATENT_DIM
    global MAX_SEQ_LENGTH, ROPE_HEAD_DIM, COMPRESSED_HEAD_DIM, KV_LATENT_DIM, Q_LATENT_DIM
    
    model_def_path = os.path.join('configs', 'model_defs', args.model_def)
    try:
        with open(model_def_path, 'r') as f:
            model_def = json.load(f)
        training_console.print(Colors.success(f"✓ Successfully loaded model definition from {args.model_def}"))
        
        # Update model parameters from the loaded definition
        HIDDEN_DIM = model_def.get('HIDDEN_DIM', HIDDEN_DIM)
        NUM_LAYERS = model_def.get('NUM_LAYERS', NUM_LAYERS)
        NUM_HEADS = model_def.get('NUM_HEADS', NUM_HEADS)
        HEAD_DIM = model_def.get('HEAD_DIM', HIDDEN_DIM // NUM_HEADS)
        FF_DIM = model_def.get('FF_DIM', FF_DIM)
        MLA_LATENT_DIM = model_def.get('MLA_LATENT_DIM', MLA_LATENT_DIM)
        MAX_SEQ_LENGTH = model_def.get('MAX_SEQ_LENGTH', MAX_SEQ_LENGTH)
        ROPE_HEAD_DIM = model_def.get('ROPE_HEAD_DIM', ROPE_HEAD_DIM)
        COMPRESSED_HEAD_DIM = model_def.get('COMPRESSED_HEAD_DIM', HEAD_DIM - ROPE_HEAD_DIM)
        KV_LATENT_DIM = model_def.get('KV_LATENT_DIM', MLA_LATENT_DIM)
        Q_LATENT_DIM = model_def.get('Q_LATENT_DIM', MLA_LATENT_DIM)
    except Exception as e:
        training_console.print(Colors.error(f"❌ Failed to load model definition from {model_def_path}: {e}"))
        training_console.print(Colors.warning(f"Using default model parameters instead"))

    def display_frame_info(frame_info: FrameSummary):
        training_console.print(Colors.info(f"  File: {Colors.highlight(frame_info.filename)}"))
        training_console.print(Colors.info(f"  Line: {Colors.highlight(frame_info.lineno)}"))
        training_console.print(Colors.info(f"  Function: {Colors.highlight(frame_info.name)}"))
        training_console.print(Colors.info(f"  Code: {Colors.highlight(frame_info.line.strip() if frame_info.line else 'N/A')}"))

    def display_exception(exception: Exception, msg: str = "❌ Training failed with error"):
        training_console.print(Colors.error(f"\n{msg}: {exception}"))
        training_console.print_exception()
        """
        tb = exception.__traceback__
        if tb:

            stack_summary = traceback.extract_tb(tb)
            training_console.print(f"\nFull stack depth: {len(stack_summary)}")

            if len(stack_summary) > 0:
                training_console.print(Colors.error(f"\n  ⓘ Outermost Frame (Frame 0):"))
                frame = stack_summary[0]
                display_frame_info(frame)

            if len(stack_summary) > 1:
                training_console.print(Colors.error(f"\n  ⓘ Caller of Error Function (Frame -2):"))
                frame = stack_summary[-2]
                display_frame_info(frame)

            if len(stack_summary) > 0:
                training_console.print(Colors.error(f"\n  ⓘ Error Location (Frame -1):"))
                frame = stack_summary[-1]
                display_frame_info(frame)
        """

    # Handle cache directory
    if args.clear_cache and os.path.exists(args.cache_dir):
        training_console.print(Colors.warning(f"Clearing cache directory: {args.cache_dir}"))
        for file in os.listdir(args.cache_dir):
            if file.endswith(".pkl"):
                os.remove(os.path.join(args.cache_dir, file))

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # Using the ChatGPT/GPT-4 encoding
        # For compatibility with HF interface
        tokenizer.vocab_size = tokenizer.max_token_value + 256  # Add special tokens count
        tokenizer.eos_token_id = 100257  # Default in tiktoken
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        print("Falling back to GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize variables
    trained_tokens = None
    run_status = "setup"

    try:
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
                cache_dir=args.cache_dir,
                console=training_console
            )

            test_dataset = CachedHFDataset(
                tokenizer=tokenizer,
                dataset_path=args.dataset,
                dataset_name=args.dataset_name,
                seq_length=args.seq_length,
                split="test",
                cache_dir=args.cache_dir,
                console=training_console
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
            training_console.print(Colors.warning(f"\n  🚫 Data loaders skipped due to previous error"))
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
            training_console.rule(Colors.highlight("Hardware Configuration"), style=Colors.HEADER)

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                training_console.print(Colors.success(f"  ✓ GPU detected: {Colors.highlight(gpu_name)}"))
                training_console.print(Colors.info(f"  • Total VRAM: {Colors.highlight(f'{total_vram:.2f} GB')}"))
            else:
                training_console.print(Colors.warning("  ⚠ No GPU detected! Training will be very slow on CPU."))

            training_console.rule(Colors.highlight("Model Configuration"), style=Colors.HEADER)

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
            training_console.print(Colors.warning(f"\n  🚫 Model configuration skipped due to previous error"))
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

            training_console.print(Colors.info(f"  • Architecture: Multi-head Latent Attention Transformer"))
            training_console.print(Colors.info(f"  • Model definition: {Colors.highlight(args.model_def)}"))
            training_console.print(Colors.info(f"  • Hidden dimension: {Colors.highlight(str(HIDDEN_DIM))}"))
            training_console.print(Colors.info(f"  • Attention heads: {Colors.highlight(str(NUM_HEADS))}"))
            training_console.print(Colors.info(f"  • Layers: {Colors.highlight(str(NUM_LAYERS))}"))
            training_console.print(Colors.info(f"  • Head dimension: {Colors.highlight(str(HEAD_DIM))}"))
            training_console.print(Colors.info(f"  • RoPE head dimension: {Colors.highlight(str(ROPE_HEAD_DIM))}"))
            training_console.print(Colors.info(f"  • Compressed head dimension: {Colors.highlight(str(COMPRESSED_HEAD_DIM))}"))
            training_console.print(Colors.info(f"  • Latent dimension: {Colors.highlight(str(MLA_LATENT_DIM))}"))
            training_console.print(Colors.info(f"  • Feed-forward dimension: {Colors.highlight(str(FF_DIM))}"))
            training_console.print(Colors.info(f"  • Parameters: {Colors.highlight(f'{total_params:,}')}"))
            training_console.print(Colors.info(f"  • Model size: {Colors.highlight(f'{param_size_mb:.2f} MB')}"))

            training_console.rule(Colors.highlight("Training Configuration"), style=Colors.HEADER)

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
            training_console.print(Colors.warning(f"\n  ⚠️ Using regular AdamW optimizer (8-bit not available)"))
        else:
            training_console.print(Colors.warning(f"\n  🚫 Optimizer configuration skipped due to previous error"))
            optimizer = None
            total_params = 0
    except Exception as e:
        display_exception(exception=e, msg="❌ Unable to create optimizer")
        run_status = "failed"
        optimizer = None
        total_params = 0

    if run_status == "setup":
        # Learning rate scheduler
        target_tokens = total_params * TOK_PER_PARAM
        total_steps = target_tokens // (train_dataset.effective_total_tokens / (len(train_dataloader) // args.grad_acc_steps))
        eval_interval = 100

        # Print training configuration
        training_console.print(Colors.info(f"  • Dataset: {Colors.highlight(f'{args.dataset}/{args.dataset_name}')}"))
        training_console.print(Colors.info(f"  • Sequence Length: {Colors.highlight(str(args.seq_length))}"))
        training_console.print(Colors.info(f"  • Batch Size: {Colors.highlight(str(args.batch_size))} " +
                                           f"(effective: {Colors.highlight(str(args.batch_size * args.grad_acc_steps))})"))
        training_console.print(Colors.info(f"  • Learning Rate: {Colors.highlight(str(args.learning_rate))}"))
        training_console.print(Colors.info(f"  • Using Cache: {Colors.highlight('No' if args.no_cache else 'Yes')}"))
        training_console.print(Colors.info(f"  • Gradient accumulation steps: {Colors.highlight(str(args.grad_acc_steps))}"))
        training_console.print(Colors.info(f"  • Estimated total optimizer steps: {Colors.highlight(f'{total_steps:,}')}"))
        training_console.print(Colors.info(f"  • Target training tokens: {Colors.highlight(f'{target_tokens:,}')}"))
        training_console.print(Colors.info(f"  • Estimated average tokens per optimizer step: {Colors.highlight(f'{(target_tokens/total_steps):.2f}')}"))

        # Measure the actual tokens per second using a warm-up phase
        training_console.rule(Colors.highlight("Performance Estimation"), style=Colors.HEADER)
        training_console.print(Colors.info("  • Running warm-up iterations to measure tokens per second..."))
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

            measure_target = 500000

            measure_progress = Progress(console=training_console, transient=True, expand=True)
            measure_task = measure_progress.add_task(description="Measuring", total=measure_target)

            # Run a few iterations and measure speed
            start_time = time.time()
            total_tokens_processed = 0
            measure_progress.start()

            with torch.no_grad():  # No need for gradients during measurement
                for batch in measure_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    # Count non-padding tokens
                    actual_tokens = attention_mask.sum().item()
                    total_tokens_processed += actual_tokens
                    measure_progress.update(measure_task, advance=actual_tokens)

                    # Forward pass only (no backprop needed for measurement)
                    _ = model(input_ids, attention_mask=attention_mask)

                    # Break after processing a few batches
                    if total_tokens_processed > measure_target:  # Enough for a good measurement
                        break

            measure_progress.stop()
            measurement_time = time.time() - start_time
            measured_tokens_per_second = total_tokens_processed / measurement_time

            # Apply a conservative factor to account for backpropagation and later epoch slowdown
            estimated_tokens_per_second = measured_tokens_per_second * 0.55  # 55% of measured forward-only speed

            training_console.print(Colors.success(f"  • Measured forward pass speed: {Colors.highlight(f'{measured_tokens_per_second:.1f}')} tokens/sec"))
            training_console.print(Colors.success(f"  • Estimated training speed: {Colors.highlight(f'{estimated_tokens_per_second:.1f}')} tokens/sec"))

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

            training_console.print(Colors.info(f"  • Estimated training time: {Colors.highlight(time_str)}"))
        else:
            training_console.print(Colors.warning(f"\n  🚫 Training time estimation skipper due to previous error"))
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
                        "compressed_head_dim": COMPRESSED_HEAD_DIM,
                        "kv_latent_dim": KV_LATENT_DIM,
                        "q_latent_dim": Q_LATENT_DIM,
                        "source": "255m_params.json" if model_def else "default values"
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
            training_console.print(Colors.warning(f"\n  🚫 Scheduler configuration skipped due to previous error"))
            scheduler = None
    except Exception as e:
        display_exception(exception=e, msg="❌ Unable to create scheduler")
        run_status = "failed"
        scheduler = None

    train_start_time = time.time()

    # Start training
    try:
        if run_status == "setup":
            # Memory usage
            training_console.rule(Colors.highlight("Training Started"), style=Colors.HEADER)
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
                console=training_console,
                dataset_size=train_dataset.effective_total_tokens,
            )
            run_status = "training"
            trained_tokens, run_status = trainer.run()
            if run_status == "failed":
                wandb.run.status = run_status
        else:
            training_console.print(Colors.warning(f"\n  🚫 Training skipped due to previous error"))
    except KeyboardInterrupt:
        training_console.print(Colors.warning("\n⚠ Training interrupted by user"))
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

    training_console.rule(Colors.highlight("Training completed"), style=Colors.HEADER)

    # Save final model
    if run_status != "failed":
        final_model_path = os.path.join("models", f"tabula_prima_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
        torch.save(model, final_model_path)
        training_console.print(Colors.success(f"✓ Saved final model to: {final_model_path}"))

    if run_status == "failed":
        training_console.print(Colors.error(f"\n❌ Training process experienced an error ❌"))
    elif run_status == "stopped":
        training_console.print(Colors.warning(f"\n🚫 Training process was stopped manually 🚫"))
    elif run_status == "terminated":
        training_console.print(Colors.warning(f"\n🛑 Training process was stopped automatically 🛑"))
    else:
        training_console.print(Colors.success(f"\n✨ Training process completed successfully ✨"))


if __name__ == "__main__":
    main()