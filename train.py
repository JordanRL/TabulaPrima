import math
import sys
import datetime

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

# Constants for our 250M parameter model
HIDDEN_DIM = 1152
NUM_LAYERS = 12
NUM_HEADS = 9
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
FF_DIM = 4608
MLA_LATENT_DIM = 288
DROPOUT = 0.1
MAX_SEQ_LENGTH = 2048
VOCAB_SIZE = 50257  # GPT-2 tokenizer vocab size


# Memory-efficient Rotary Position Embedding (RoPE)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=MAX_SEQ_LENGTH):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum length ({self.max_seq_len})")

        # Cached RoPE to save memory and computation
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Multi-head Latent Attention implementation
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, latent_dim, r_dim=None, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.latent_dim = latent_dim
        
        # Set a safe default for rope_dim to avoid dimension issues
        if r_dim is not None:
            self.rope_dim = r_dim
        else:
            # Calculate a safe rope_dim that won't cause negative dimensions
            # Ensure rope_dim * num_heads < hidden_dim / 2 to leave space for other dimensions
            self.rope_dim = self.head_dim // 2
            
        # Query compression
        self.q_down_proj = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.q_up_proj = nn.Linear(latent_dim, hidden_dim - self.rope_dim * num_heads, bias=False)

        # Special RoPE projections
        self.q_rope_proj = nn.Linear(latent_dim, self.rope_dim * num_heads, bias=False)
        self.k_rope_proj = nn.Linear(hidden_dim, self.rope_dim, bias=False)

        # Key-Value compression to latent space
        self.kv_down_proj = nn.Linear(hidden_dim, latent_dim, bias=False)

        # Latent to Key/Value expansions
        self.k_up_proj = nn.Linear(latent_dim, hidden_dim - self.rope_dim * num_heads, bias=False)
        self.v_up_proj = nn.Linear(latent_dim, hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.rope_dim)

        # Flag for optimized generation
        self.generation_ready = False
        self.use_absorption = False
        self.register_buffer('q_fused_weight', None)
        self.register_buffer('q_rope_fused_weight', None)

    def prepare_for_generation(self):
        """Optimize matrices for faster inference during generation"""
        if self.generation_ready:
            return

        self.generation_ready = True

        # Precompute fused matrices
        q_fused_weight = torch.matmul(
            self.q_up_proj.weight,
            self.q_down_proj.weight
        )
        self.register_buffer('q_fused_weight', q_fused_weight)

        # Fuse RoPE query projections
        q_rope_fused_weight = torch.matmul(
            self.q_rope_proj.weight,
            self.q_down_proj.weight
        )
        self.register_buffer('q_rope_fused_weight', q_rope_fused_weight)

    def set_use_absorption(self, use_absorption):
        """Toggle whether to use matrix absorption during inference"""
        self.use_absorption = use_absorption

    def forward(self, x, past_key_value=None, attention_mask=None, is_causal=True, use_cache=False):
        batch_size, seq_len, _ = x.shape

        # Check if we should use optimized path
        use_optimized = self.generation_ready and self.use_absorption and past_key_value is not None

        # === QUERY PATH ===
        if use_optimized:
            # Fast path with fused matrices for inference
            q_c = F.linear(x, self.q_fused_weight.t())
            # Use the actual output size rather than calculated dimensions
            q_c_dim = q_c.size(-1) // self.num_heads
            q_c = q_c.view(batch_size, seq_len, self.num_heads, q_c_dim)

            q_r = F.linear(x, self.q_rope_fused_weight.t())
            q_r = q_r.view(batch_size, seq_len, self.num_heads, self.rope_dim)
        else:
            # Standard path for training
            q_latent = self.q_down_proj(x)
            q_c = self.q_up_proj(q_latent)
            # Use the actual output size rather than calculated dimensions
            q_c_dim = q_c.size(-1) // self.num_heads
            q_c = q_c.view(batch_size, seq_len, self.num_heads, q_c_dim)

            q_r = self.q_rope_proj(q_latent)
            q_r = q_r.view(batch_size, seq_len, self.num_heads, self.rope_dim)

        # === POSITION EMBEDDINGS ===
        past_length = 0
        if past_key_value is not None:
            past_length = past_key_value[2]

        # Get sequence length and ensure it matches the actual tensor dimension
        actual_seq_len = q_r.size(1)
        cos, sin = self.rotary_emb(q_r, actual_seq_len + past_length)

        # Ensure proper slicing
        q_r_pos_slice = slice(-actual_seq_len, None) if past_length > 0 else slice(None)

        # Apply RoPE - FIX: Properly align dimensions for broadcasting
        cos_slice = cos[:, :, q_r_pos_slice]
        sin_slice = sin[:, :, q_r_pos_slice]

        # Reshape for proper broadcasting with q_r [batch, seq_len, num_heads, rope_dim]
        cos_slice = cos_slice.unsqueeze(0)  # [1, 1, seq_len, dim]
        sin_slice = sin_slice.unsqueeze(0)
        cos_slice = cos_slice.permute(0, 2, 1, 3)  # [1, seq_len, 1, dim]
        sin_slice = sin_slice.permute(0, 2, 1, 3)

        # Expand to match batch size and num_heads dimensions
        cos_slice = cos_slice.expand(batch_size, -1, self.num_heads, -1)
        sin_slice = sin_slice.expand(batch_size, -1, self.num_heads, -1)

        # Now the shapes align for the element-wise operations
        q_r = (q_r * cos_slice) + (rotate_half(q_r) * sin_slice)

        # === KEY-VALUE PATH ===
        kv_latent_cur = self.kv_down_proj(x)

        k_r_cur = self.k_rope_proj(x)
        k_r_cur = k_r_cur.view(batch_size, seq_len, 1, self.rope_dim)

        # Apply the same reshaping for key rotary embeddings
        k_r_cur = (k_r_cur * cos_slice.expand(-1, -1, 1, -1)) + (rotate_half(k_r_cur) * sin_slice.expand(-1, -1, 1, -1))

        # === CACHE HANDLING ===
        if past_key_value is not None:
            kv_latent_past, k_r_past, _ = past_key_value
            kv_latent = torch.cat([kv_latent_past, kv_latent_cur], dim=1)
            k_r = torch.cat([k_r_past, k_r_cur], dim=1)
            curr_seq_len = past_length + seq_len
        else:
            kv_latent = kv_latent_cur
            k_r = k_r_cur
            curr_seq_len = seq_len

        # === EXPAND FROM LATENT ===
        k_r = k_r.expand(batch_size, curr_seq_len, self.num_heads, self.rope_dim)

        k_c = self.k_up_proj(kv_latent)
        v = self.v_up_proj(kv_latent)

        # Use the actual output size rather than calculated dimensions
        k_c_dim = k_c.size(-1) // self.num_heads
        k_c = k_c.view(batch_size, curr_seq_len, self.num_heads, k_c_dim)
        v = v.view(batch_size, curr_seq_len, self.num_heads, self.head_dim)

        # === ATTENTION CALCULATION ===
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)  # [B, H, S, D]
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2)  # [B, H, C, D]
        v = v.transpose(1, 2)  # [B, H, C, D]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(torch.ones(curr_seq_len, curr_seq_len,
                                                dtype=torch.bool, device=x.device), diagonal=1)

            if past_length > 0:
                causal_mask = causal_mask[-seq_len:, :]

            # Reshape mask for broadcasting with scores [B, H, S, C]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores.masked_fill_(causal_mask, -10000.0)

        # Apply attention mask with proper shape
        if attention_mask is not None:
            # Attention mask typically has shape [B, S] or [B, 1, S] or [B, 1, 1, S]
            # We need to reshape it to [B, 1, S, 1] or [B, 1, S, C] to match scores [B, H, S, C]
            if attention_mask.dim() == 2:
                # [B, S] -> [B, 1, S, 1]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            elif attention_mask.dim() == 3:
                # [B, 1, S] -> [B, 1, S, 1]
                attention_mask = attention_mask.unsqueeze(-1)

            # Expand attention mask to match scores dimensions
            attention_mask = attention_mask.expand(-1, -1, -1, curr_seq_len)

            # If necessary, expand the head dimension
            if attention_mask.size(1) == 1 and scores.size(1) > 1:
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)

            scores = scores + attention_mask

        # Apply softmax and compute weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(context)

        # Return with cache if requested
        if use_cache:
            return output, (kv_latent, k_r, curr_seq_len)
        else:
            return output


# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Transformer Layer with MLA
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadLatentAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        # First sublayer: MLA with residual connection
        # Pass the normalized input and attention mask to MLA
        normalized_x = self.norm1(x)
        attn_output = self.attention(normalized_x, attention_mask=attention_mask)
        x = x + self.dropout(attn_output)

        # Second sublayer: FFN with residual connection
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


# Full Transformer Model
class MLATransformer(nn.Module):
    def __init__(
            self,
            vocab_size=VOCAB_SIZE,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            latent_dim=MLA_LATENT_DIM,
            dropout=DROPOUT,
            max_seq_len=MAX_SEQ_LENGTH
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # Use torch.nn.ModuleList for better memory efficiency with checkpoint
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ff_dim, latent_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        # Flag for generation optimization
        self.generation_ready = False

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.shape

        # Embed tokens and positions
        x = self.embedding(input_ids)

        # Add positional embeddings - adjust for partial positions in generation
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        if past_key_values is not None:
            # Adjust position IDs for generation
            position_ids = position_ids + past_key_values[0][2]  # Add past length

        # Add positional embeddings
        x = x + self.pos_embedding[:, position_ids, :]

        # Process with attention and KV cache
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if use_cache:
                x, present_key_value = layer(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=True
                )
                present_key_values.append(present_key_value)
            else:
                x = layer(
                    x,
                    attention_mask=attention_mask
                )

        x = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, present_key_values
        else:
            return logits

    def prepare_for_generation(self):
        """Precompute optimized matrices for faster inference"""
        if self.generation_ready:
            return

        self.generation_ready = True

        # Prepare each layer for optimized generation
        for layer in self.layers:
            # Assuming each layer has an attention module that can be optimized
            if hasattr(layer.attention, 'prepare_for_generation'):
                layer.attention.prepare_for_generation()
            if hasattr(layer.attention, 'set_use_absorption'):
                layer.attention.set_use_absorption(True)

    def generate(self, input_ids, max_length, temperature=1.0, top_k=0, top_p=0.9, device=None):
        """Generate text using the model with KV caching for efficiency"""
        # Prepare for optimized generation if not already done
        if not self.generation_ready:
            self.prepare_for_generation()

        self.eval()  # Set to evaluation mode

        if device is None:
            device = input_ids.device

        # Initial forward pass with the input sequence
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        # Process the input sequence with caching
        outputs = self(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )

        logits, past_key_values = outputs
        generated_ids = input_ids.clone()

        # Generate new tokens autoregressively
        for _ in range(max_length - seq_len):
            # Get next token probabilities
            next_token_logits = logits[:, -1, :] / temperature

            # Apply sampling (top-k, top-p, etc.)
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # Keep at least one token

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Forward pass with the last token and cached KV
            new_attention_mask = torch.ones(batch_size, 1, device=device)

            outputs = self(
                next_token,
                attention_mask=new_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits, past_key_values = outputs

            # Optional: stop generation if all sequences have generated an EOS token
            if all(generated_ids[:, -1] == self.eos_token_id):
                break

        return generated_ids


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

# Memory-efficient training function with additional optimizations for 2070 Super
def train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        device,
        total_parameters,
        gradient_accumulation_steps=8,
        max_grad_norm=1.0,
        checkpoint_dir="checkpoints",
        model_dir="models",
        use_amp=True,
        log_interval=10,
        eval_interval=1000,
        global_steps=None
):
    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Additional memory optimization: periodically empty CUDA cache
    empty_cache_interval = 100  # Empty cache every 100 optimization steps

    # Tracking training dynamics
    total_tokens = 0
    target_tokens = total_parameters * 10
    log_loss = 0
    token_window = []
    token_times = []
    loss_window = []
    global_steps = global_steps or len(train_dataloader)
    optimizer_steps = 0
    inference_steps = 0
    steps_since_instrument = 0
    start_time = time.time()
    checkpoint_interval = 60 * 60 * 4
    last_checkpoint_time = start_time

    while total_tokens < target_tokens:
        progress_bar = tqdm(total=target_tokens, desc=f"Pretrain")

        eval_interval_steps = max(1, eval_interval)

        for step, batch in train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Initialize logging dicts
            training_dict = {}
            eval_dict = {}

            # Increment
            inference_steps += 1

            # Forward pass with mixed precision
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask=attention_mask, use_cache=True)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    loss = loss / gradient_accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids, attention_mask=attention_mask, use_cache=True)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

            # Update weights after accumulating gradients
            if inference_steps % gradient_accumulation_steps == 0:
                optimizer_steps += 1
                steps_since_instrument += 1
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                # Clear CUDA cache periodically to prevent fragmentation
                #if inference_steps // gradient_accumulation_steps % empty_cache_interval == 0:
                #    torch.cuda.empty_cache()

            # Update tracking metrics
            total_tokens += input_ids.numel()
            log_loss += loss.item() * gradient_accumulation_steps

            # Update moving window metrics
            token_window.append(input_ids.numel())
            token_times.append(time.time())
            loss_window.append(loss.item() * gradient_accumulation_steps)
            if len(token_window) > 10:
                token_window = token_window[-10:]
            if len(token_times) > 10:
                token_times = token_times[-10:]
            if len(loss_window) > 15:
                loss_window = loss_window[-15:]

            progress_bar.update(input_ids.numel())

            # Calculate tokens per second
            tokens_per_sec = sum(token_window) / token_times[-1:] - token_times[0]

            # Calculate current perplexity (exp of loss)
            current_loss = sum(loss_window) / len(loss_window)
            current_perplexity = math.exp(current_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "ppl": f"{current_perplexity:.2f}",
                "tokens/s": f"{tokens_per_sec:.2f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })

            if steps_since_instrument == log_interval or optimizer_steps % eval_interval_steps == 0:
                # Calculate gradient norm
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                momentary_loss = log_loss / steps_since_instrument
                momentary_perplexity = math.exp(momentary_loss)
                training_dict = {
                    "training/loss": momentary_loss,
                    "training/perplexity": momentary_perplexity,
                    "training/learning_rate": scheduler.get_last_lr()[0],
                    "training/tokens_per_second": tokens_per_sec,
                    "training/grad_norm": grad_norm,
                    "metric/progress": optimizer_steps / global_steps
                }
                log_loss = 0
                steps_since_instrument = 0

            if optimizer_steps % eval_interval_steps == 0:
                print(Colors.header(f"\n{'-'*40}"))
                print(Colors.header(f" Evaluating model performance on test dataset"))
                print(Colors.header(f"{'-'*40}"))

                # Run evaluation
                eval_results = evaluate(model, test_dataloader, device, use_amp)
                test_loss = eval_results['loss']
                test_accuracy = eval_results['accuracy']
                test_perplexity = eval_results['perplexity']

                # Print intermediate results
                print(Colors.info(f"  • Eval loss: {Colors.highlight(f'{test_loss:.4f}')}"))
                print(Colors.info(f"  • Eval perplexity: {Colors.highlight(f'{test_perplexity:.2f}')}"))
                print(Colors.info(f"  • Eval accuracy: {Colors.highlight(f'{test_accuracy:.2%}')}"))

                eval_dict = {
                    "eval/test_loss": test_loss,
                    "eval/test_perplexity": test_perplexity,
                    "eval/test_accuracy": test_accuracy,
                    "metric/progress": optimizer_steps / global_steps,
                }
                model.train()

            log_dict = {}
            if training_dict is not None:
                log_dict.update(training_dict)
            if eval_dict is not None:
                log_dict.update(eval_dict)

            if len(log_dict) > 0:
                wandb.log(log_dict)

            if total_tokens >= target_tokens:
                break

            if (time.time() - last_checkpoint_time) >= checkpoint_interval:
                save_checkpoint(model, optimizer, scheduler, checkpoint_dir)

        if total_tokens >= target_tokens:
            progress_bar.close()
            break

    return total_tokens

def save_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    print(Colors.header(f"\n{'-' * 40}"))
    print(Colors.header(f" Saving Checkpoint"))
    print(Colors.header(f"{'-' * 40}"))

    checkpoint_path = os.path.join(checkpoint_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, checkpoint_path)
    print(Colors.success(f"✓ Saved checkpoint: {checkpoint_path}"))


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

def evaluate(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    # Use tqdm for progress bar during evaluation
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask=attention_mask)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
            else:
                logits = model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

            # Calculate accuracy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            
            # Find indices where labels are not -100 (padding)
            valid_indices = flat_labels != -100
            
            if valid_indices.sum() > 0:
                predictions = flat_logits[valid_indices].argmax(dim=-1)
                correct = predictions.eq(flat_labels[valid_indices])
                total_correct += correct.sum().item()
                total_tokens += valid_indices.sum().item()
            
            total_loss += loss.item()

    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    # Return to training mode
    model.train()
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity
    }

# Main training script with improvements for 2070 Super
def main():

    # Hyperparameters optimized for 250M model on 2070 Super
    batch_size = 4  # Can use batch size 2 with the smaller model
    learning_rate = 3e-4  # Lower learning rate for stability
    gradient_accumulation_steps = 4  # Effective batch size = 16
    max_seq_length = 2048  # Reduced for better fitting in VRAM

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=max_seq_length)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_title = "WikiText103"
    trained_tokens = None

    # Create dataset and dataloader
    train_dataset = HFDataset(
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-103-raw-v1",
        seq_length=max_seq_length,
        split="train"
    )

    test_dataset = HFDataset(
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-103-raw-v1",
        seq_length=max_seq_length,
        split="test"
    )

    # Define a collate function to handle variable length sequences
    def collate_fn(batch):
        # Sort batch by sequence length (descending) for more efficient processing
        batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
        
        # Get max lengths for this batch
        max_input_len = max([len(x["input_ids"]) for x in batch])
        max_label_len = max([len(x["labels"]) for x in batch])
        
        # Prepare padding token (usually the EOS token in GPT-2)
        pad_token_id = tokenizer.eos_token_id
        
        # Pad all sequences to max length in batch
        input_ids_padded = []
        labels_padded = []
        attention_mask = []
        
        for item in batch:
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
            attention_mask.append(mask)
            
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
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels_padded)
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

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
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        latent_dim=MLA_LATENT_DIM,
        dropout=DROPOUT,
        max_seq_len=max_seq_length
    )
    model.to(device)

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

    # Initialize optimizer with weight decay and 8-bit precision
    #try:
        # Try to use 8-bit Adam if available (reduces optimizer memory by 75%)
        #from bitsandbytes.optim import Adam8bit
        #optimizer = Adam8bit(model.parameters(), lr=learning_rate, weight_decay=0.01)
        #print(Colors.success(f"✓ Using 8-bit Adam optimizer for memory efficiency"))
    #except ImportError:
        # Fall back to regular AdamW
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    print(Colors.warning(f"⚠ Using regular AdamW optimizer (8-bit not available)"))

    # Learning rate scheduler
    target_tokens = total_params * 10
    total_steps = target_tokens // (train_dataset.total_tokens / (len(train_dataloader) // gradient_accumulation_steps))
    eval_interval = total_steps / 100
    
    # Training configuration info
    print(Colors.info(f"  • Batch size: {Colors.highlight(str(batch_size))} (effective: {Colors.highlight(str(batch_size * gradient_accumulation_steps))})"))
    print(Colors.info(f"  • Learning rate: {Colors.highlight(str(learning_rate))}"))
    print(Colors.info(f"  • Gradient accumulation steps: {Colors.highlight(str(gradient_accumulation_steps))}"))
    print(Colors.info(f"  • Estimated total optimizer steps: {Colors.highlight(f'{total_steps:,}')}"))
    print(Colors.info(f"  • Target training tokens: {Colors.highlight(f'{target_tokens:,}')}"))
    
    # Measure the actual tokens per second using a warm-up phase
    print(Colors.header(f"\n{'='*50}"))
    print(Colors.header(f" Measuring Performance"))
    print(Colors.header(f"{'='*50}"))
    
    print(Colors.info("  • Running warm-up iterations to measure tokens per second..."))
    
    # Do a few warm-up steps to measure performance
    model.train()
    
    # Create small dataloader with a few batches for measurement
    measure_batch_size = batch_size 
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
    
    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3) 
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(Colors.info(f"  • GPU memory allocated: {Colors.highlight(f'{allocated_gb:.2f} GB')}"))
    print(Colors.info(f"  • GPU memory reserved: {Colors.highlight(f'{reserved_gb:.2f} GB')}"))

    # Check gradient enabled for params
    check_grad_enabled(model)

    # Initialize wandb (optional)
    wandb.init(
        project="TabulaPrima",
        entity="jordan-ledoux-none",
        job_type="training",
        tags=["experiment","generic-dataset","pretraining"],
        config={
            "parameters": total_params,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": "Adam8Bit" if "8bit" in optimizer.__class__.__name__ else "AdamW",
        },
        name=dataset_title+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M"),
    )
    """
    wandb.watch(
        model,
        log="all",
        log_freq=100
    )
    """
    train_start_time = time.time()

    # Start training
    try:
        trained_tokens = train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            total_parameters=total_params,
            gradient_accumulation_steps=gradient_accumulation_steps,
            checkpoint_dir="checkpoints",
            eval_interval=eval_interval,
            global_steps=total_steps,
        )
    except KeyboardInterrupt:
        print(Colors.warning("\n⚠ Training interrupted by user"))
        wandb.alert(title="Training interrupted", text="Training interrupted by user")
        wandb.run.status = "stopped"
    except Exception as e:
        print(Colors.error(f"\n❌ Training failed with error: {e}"))
        wandb.alert(title="Training failed", text=f"An error occurred during training: {e}")
        wandb.run.status = "failed"
    finally:
        # Calculate training statistics
        train_end_time = time.time()
        total_training_time = train_end_time - train_start_time
        total_training_hours = total_training_time / 3600
        
        # Get actual checkpoint information if we finished at least one epoch
        best_result = {}
        try:
            best_model_path = os.path.join("checkpoints", "best_model.pt")
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location="cpu")
                best_result = {
                    "best_epoch": checkpoint.get("epoch", 0),
                    "best_train_loss": checkpoint.get("train_loss", float('inf')),
                    "best_train_perplexity": checkpoint.get("train_perplexity", float('inf')),
                    "best_test_loss": checkpoint.get("test_loss", float('inf')),
                    "best_test_perplexity": checkpoint.get("test_perplexity", float('inf')),
                    "best_test_accuracy": checkpoint.get("test_accuracy", 0),
                }
        except Exception as e:
            print(Colors.warning(f"Could not load best model info: {e}"))
            
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
            
            # Best model performance
            **{f"run/{k}": v for k, v in best_result.items()},
        })
        
        # Create a final summary table
        if wandb.run is not None:
            columns = ["Metric", "Value"]
            data = [
                ["Total Training Time", f"{total_training_hours:.2f} hours"],
                ["Average Tokens/Second", f"{avg_tokens_per_second:.2f}"],
                ["Total Tokens Processed", f"{total_tokens:,}"],
                ["Model Parameters", f"{sum(p.numel() for p in model.parameters()):,}"],
                ["Highest Test Accuracy", f"{best_result.get('best_test_accuracy', 0):.2%}"],
                ["Lowest Test Perplexity", f"{best_result.get('best_test_perplexity', float('inf')):.2f}"],
                ["Best Epoch", f"{best_result.get('best_epoch', 0)}"],
                ["GPU Utilization", f"{gpu_utilization:.1f}%"],
            ]
            wandb.log({"run/summary_table": wandb.Table(columns=columns, data=data)})
        
        wandb.finish()

    print(Colors.header(f"\n{'='*50}"))
    print(Colors.header(f" Training Complete"))
    print(Colors.header(f"{'='*50}"))

    # Save final model
    final_model_path = os.path.join("models", f"tabula_prima_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pt")
    torch.save(model, final_model_path)
    print(Colors.success(f"✓ Saved final model to: {final_model_path}"))

    # Final memory usage
    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3) 
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(Colors.info(f"  • Final GPU memory allocated: {Colors.highlight(f'{allocated_gb:.2f} GB')}"))
    print(Colors.info(f"  • Final GPU memory reserved: {Colors.highlight(f'{reserved_gb:.2f} GB')}"))
    
    print(Colors.success(f"\n✨ Training process completed successfully! ✨"))


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