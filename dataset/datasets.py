import pickle
import hashlib
import torch.distributed as dist
import itertools
import logging
import json
import hydra.utils
import torch
import os

from datasets.iterable_dataset import _BaseExamplesIterable
from smart_open import smart_open
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.utils.data._utils.worker import WorkerInfo
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from datasets import load_dataset, StreamingDownloadManager, get_dataset_infos, DownloadConfig, DatasetDict, Dataset, \
    IterableDatasetDict, IterableDataset

from config_schema import DatasetConfig
from console import Colors, TPConsole
from typing import List, Optional, Dict, Any, Iterable
from datasets.utils.logging import set_verbosity_error as set_datasets_verbosity_error
from pprint import pprint


class TPDataset(Dataset):
    def __init__(self, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

class TPIterableDataset(IterableDataset):
    worker_info: WorkerInfo | None

    def __init__(self, tokenizer, cfg: DatasetConfig, ex_iterable: _BaseExamplesIterable):
        super().__init__(ex_iterable)
        self.tokenizer = tokenizer
        self.max_seq_length = cfg.seq_length

        # --- Get Distributed Information ---
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.using_dist = True
        else:
            self.rank = 0
            self.world_size = 1
            self.using_dist = False

        self.worker_info = get_worker_info()

    def __iter__(self):
        return self

class TPStreamingDataset(TPIterableDataset):
    stream_handle: IterableDatasetDict | IterableDataset
    current_file_index: int
    current_record_index: int
    text_column: str
    split: str
    dataset_name: str
    dataset_path: str

    def __init__(
            self,
            tokenizer: Any,
            split: str,
            cfg: DatasetConfig,
            text_column: str = "text",
            shuffle_seed: int = 42,
    ) -> None:

        super().__init__(tokenizer=tokenizer, cfg=cfg)
        self.dataset_path = cfg.dataset_path
        self.dataset_name = cfg.name
        self.split = split
        self.text_column = text_column

        if self.worker_info is not None:
            self.num_workers = self.worker_info.num_workers if self.worker_info else 1
            self.effective_rank = self.rank * self.num_workers + self.worker_info.id
            self.effective_world_size = self.world_size * self.num_workers
        else:
            self.num_workers = 1
            self.effective_rank = self.rank
            self.effective_world_size = self.world_size

        self.current_record_index = 0
        self.current_file_index = 0
        self.stream_handle = load_dataset(
            self.dataset_path,
            self.dataset_name,
            split=self.split,
            download_config=DownloadConfig(disable_tqdm=True),
            streaming=True,
            cache_dir=None,
        )

        self.stream_handle.shuffle(seed=shuffle_seed)

        # Rank 0 Worker 0 is the only one that retrieves the files
        if self.using_dist and self.effective_rank == 0:
            dl_manager = StreamingDownloadManager(
                base_path=self.dataset_path,
                download_config=DownloadConfig(disable_tqdm=True),
                dataset_name=self.dataset_name,
            )

            # TODO: Finish implementation

    def _parse_and_tokenize(self, line: str) -> Optional[Dict[str, torch.Tensor]]:
        """Parses a line (assuming JSONL) and tokenizes it."""
        try:
            data = json.loads(line)
            text = data.get(self.text_column)
            if text is None:
                # logger.warning(f"Skipping line missing text column '{self.text_column}': {line[:100]}...")
                return None

            tokenized = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt" # Return PyTorch tensors
            )
            # Squeeze to remove the batch dimension added by tokenizer (usually batch size 1)
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                # Add token_type_ids if needed for your model
                # "token_type_ids": tokenized["token_type_ids"].squeeze(0),
            }
        except json.JSONDecodeError:
            # logger.warning(f"Skipping invalid JSON line: {line[:100]}...")
            return None
        except Exception as e:
            logger.warning(f"Skipping line due to error during parsing/tokenization: {e}")
            return None

    def __iter__(self):
        """
        The core iterator logic. Determines sharding, iterates through URLs,
        streams lines, applies modulus check, parses, tokenizes, and yields.
        """
        rank = 0
        world_size = 1
        worker_id = 0
        num_workers = 1

        # --- Get Distributed Information ---
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        worker_info = get_worker_info()
        if worker_info is not None:
            # DataLoader is using multiple workers per process
            worker_id = worker_info.id
            num_workers = worker_info.num_workers # Total workers per rank
        else:
            # Single worker process (or num_workers=0 in DataLoader)
            pass

        # Calculate the effective rank and world size across all processes and workers
        effective_rank = rank * num_workers + worker_id
        effective_world_size = world_size * num_workers

        logger.info(f"[Rank {rank}/Worker {worker_id}] Effective Rank: {effective_rank}, Effective World Size: {effective_world_size}")

        # --- Streaming and Sharding Logic ---
        current_global_record_index = 0
        files_processed_this_worker = 0
        records_yielded_this_worker = 0

        # Iterate through all the file URLs associated with the dataset split
        for url in self.file_urls:
            logger.debug(f"[Rank {rank}/Worker {worker_id}] Opening URL: {url}")
            try:
                # smart_open handles http/https and common compressions (gz, bz2, xz) based on extension
                # Use 'rt' for reading text, it handles decoding
                # If files are binary or need specific encoding, adjust mode and encoding
                with smart_open(url, 'rt', encoding='utf-8') as file_handle:
                    files_processed_this_worker += 1
                    for line in file_handle:
                        # Check if this record index belongs to the current worker
                        if current_global_record_index % effective_world_size == effective_rank:
                            # Process the line
                            processed_data = self._parse_and_tokenize(line)
                            if processed_data:
                                records_yielded_this_worker += 1
                                yield processed_data
                                # Optional: Log progress periodically
                                # if records_yielded_this_worker % 1000 == 0:
                                #    logger.info(f"[Rank {rank}/Worker {worker_id}] Yielded {records_yielded_this_worker} records...")

                        # Increment global counter *after* checking modulus
                        # This ensures every record index is considered across all workers.
                        current_global_record_index += 1

            except EOFError:
                 logger.warning(f"[Rank {rank}/Worker {worker_id}] Encountered EOFError, possibly truncated file: {url}. Continuing...")
            except Exception as e:
                # Log error but continue to next file if possible
                logger.error(f"[Rank {rank}/Worker {worker_id}] Error processing URL {url}: {e}", exc_info=True)
                # Depending on severity, you might want to break or raise here

        logger.info(f"[Rank {rank}/Worker {worker_id}] Finished iteration. Processed approx {files_processed_this_worker} files. Yielded {records_yielded_this_worker} records.")

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
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
    def __init__(self, tokenizer, dataset_path, seq_length, dataset_name=None, split="train"):
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

        print(Colors.header(f"{'=' * 50}"))
        print(Colors.header(f" Loading {split.upper()} Dataset: {dataset_path}/{dataset_name or ''}"))
        print(Colors.header(f"{'=' * 50}"))

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
        print(Colors.success(f"» Loaded {split} dataset:"))
        print(Colors.info(f"» Dataset: {dataset_path}/{dataset_name or ''}"))
        print(Colors.info(f"» Text items processed: {text_items} (skipped {skipped_short} short items)"))
        print(Colors.info(f"» Training examples: {Colors.highlight(f'{len(self.examples):,}')}"))
        print(Colors.info(f"» Total tokens: {Colors.highlight(f'{total_tokens:,}')}"))
        print(Colors.info(f"» Avg tokens per example: {total_tokens / max(1, len(self.examples)):.1f}"))

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


class CachedHFDataset:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            dataset_path,
            dataset_name=None,
            seq_length=2048,
            split="train",
            cache_dir="dataset_cache"
    ):
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
        self.cache_dir = hydra.utils.to_absolute_path(cache_dir)
        self.console = TPConsole()
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
        self.cache_file = os.path.join(self.cache_dir, f"{cache_hash}.pkl")

        self.console.subrule(f"Dataset: {dataset_path}/{dataset_name or ''} ({split})")

        # Try to load from cache first
        self.examples = self._load_from_cache()

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

            self.console.create_progress_task("dataset", task_desc=f"Tokenizing {split} dataset", total=len(texts))
            for text in texts:
                self.console.update_progress_task("dataset", advance=1)
                if not text.strip():
                    continue

                text_items += 1
                # Tokenize the current text item
                if tokenizer_name == "PreTrainedTokenizerFast":
                    encodings = tokenizer(text, return_tensors="pt", truncation=False, max_length=None)
                    input_ids = encodings.input_ids[0]
                else:
                    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
                self.total_tokens += len(input_ids)

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

            # Ensure we have at least one example
            if len(self.examples) == 0:
                self.console.print(f"Warning: No examples found in {split} set, creating a dummy example")
                self.examples.append(torch.tensor([tokenizer.bos_token_id, tokenizer.eos_token_id]))

            self.console.remove_progress_task("dataset")

            # Save processed examples to cache
            self._save_to_cache()

        # Print summary information
        summary = [
            {"title": "Dataset", "content": dataset_path+"/"+dataset_name},
            {"title": "Split", "content": split},
            {"title": "Examples", "content": f"{len(self.examples):,}"},
            {"title": "Sequence length", "content": seq_length}
        ]
        if hasattr(self, "total_tokens"):
            summary.append({"title": "Total raw tokens", "content": f"{self.total_tokens:,}"})
        if hasattr(self, 'effective_total_tokens'):
            summary.append({"title": "Total effective tokens", "content": f"{self.effective_total_tokens:,}"})
            summary.append({"title": "Avg tokens per example", "content": f"{self.effective_total_tokens / max(1, len(self.examples)):.2f}"})

        self.console.print_list(summary)

    def _load_from_cache(self):
        """Try to load the dataset from cache file"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            return None

        if os.path.exists(self.cache_file):
            try:
                self.console.print_notification(f"Loading cached dataset from {Colors.header(self.cache_file)}")
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.total_tokens = cache_data['total_tokens']
                    self.effective_total_tokens = cache_data['effective_total_tokens']
                    return cache_data['examples']
            except Exception as e:
                self.console.print_error(f"Error loading cache: {e}")
                return None
        return None

    def _save_to_cache(self):
        """Save the processed dataset to cache file"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        self.console.print_notification(f"Saving dataset to cache at {Colors.header(self.cache_file)}")
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