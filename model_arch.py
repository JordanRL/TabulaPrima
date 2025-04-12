import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- RotaryEmbedding Class (Mostly unchanged, ensure dim matches ROPE_HEAD_DIM) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Initialize frequency bands - standard RoPE implementation
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute embeddings for all possible positions up to max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but equivalent to complex number representation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Cache cos and sin embeddings
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        """
        Return cos and sin embeddings for a given sequence length.
        Args:
            seq_len: The sequence length to get embeddings for.
        Returns:
            Tuple of (cos, sin) of shape [seq_len, dim]
        """
        if seq_len > self.max_seq_len_cached:
             self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )

# --- New RoPE Application Function (Standard Implementation) ---
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin, position_ids):
    """
    Applies Rotary Position Embedding to the input tensor x.

    Args:
        x (torch.Tensor): Input tensor, shape or.
        cos (torch.Tensor): Cosine embeddings, shape.
        sin (torch.Tensor): Sine embeddings, shape.
        position_ids (torch.Tensor): Position indices, shape.

    Returns:
        torch.Tensor: Tensor with RoPE applied.
    """
    # cos/sin: ->
    cos = cos[position_ids].unsqueeze(2) #
    sin = sin[position_ids].unsqueeze(2) #

    # Handle head dimension broadcasting if x has heads
    # and cos/sin are
    # Or if x is shared key and cos/sin are
    # This unsqueeze handles both cases correctly due to broadcasting rules.

    rotated_x = (x * cos) + (rotate_half(x) * sin)
    return rotated_x


# --- Corrected MultiHeadLatentAttention Class (DeepSeek V2 Aligned) ---
class MultiHeadLatentAttention(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 kv_latent_dim,
                 q_latent_dim,
                 rope_head_dim,
                 dropout,
                 max_seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.kv_latent_dim = kv_latent_dim # d_c
        self.q_latent_dim = q_latent_dim   # d'_c
        self.rope_head_dim = rope_head_dim # d_h^R

        # Calculate compressed content head dimension
        self.head_dim = hidden_dim // num_heads # Full head dim (d_h = d_h^C + d_h^R)
        self.compressed_head_dim = self.head_dim - self.rope_head_dim # d_h^C

        if self.compressed_head_dim <= 0:
             raise ValueError("Rope head dimension must be smaller than the full head dimension.")

        # --- Projections based on DeepSeek V2 Equations ---
        # Eq (9): KV Down-projection (Shared)
        self.kv_down_proj = nn.Linear(hidden_dim, kv_latent_dim, bias=False) # W_DKV

        # Eq (12): Query Down-projection
        self.q_down_proj = nn.Linear(hidden_dim, q_latent_dim, bias=False) # W_DQ

        # Eq (10): Key Up-projection (Compressed Content Part)
        self.k_c_proj = nn.Linear(kv_latent_dim, self.compressed_head_dim * num_heads, bias=False) # W_UK (output dim d_h^C * n_h)

        # Eq (11): Value Up-projection (Compressed Content Part)
        self.v_proj = nn.Linear(kv_latent_dim, self.head_dim * num_heads, bias=False) # W_UV (output dim d_h * n_h)

        # Eq (13): Query Up-projection (Compressed Content Part)
        self.q_c_proj = nn.Linear(q_latent_dim, self.compressed_head_dim * num_heads, bias=False) # W_UQ (output dim d_h^C * n_h)

        # Eq (15): RoPE Key Projection (from hidden_state, shared across heads)
        self.k_r_proj = nn.Linear(hidden_dim, self.rope_head_dim, bias=False) # W_KR (output dim d_h^R)

        # Eq (14): RoPE Query Projection (from query latent)
        self.q_r_proj = nn.Linear(q_latent_dim, self.rope_head_dim * num_heads, bias=False) # W_QR (output dim d_h^R * n_h)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Rotary embeddings (initialized with RoPE dimension d_h^R)
        self.rotary_emb = RotaryEmbedding(dim=self.rope_head_dim, max_seq_len=max_seq_len)

        # Flags/Buffers for potential future matrix absorption implementation
        self.generation_ready = False
        self.use_absorption = False
        # Register buffers for absorbed weights (logic to compute/use them is NOT implemented here)
        self.register_buffer('q_absorbed_weight', None) # Would absorb W_UK
        self.register_buffer('o_absorbed_weight', None) # Would absorb W_UV

    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, is_causal=True, use_cache=False):
        batch_size, seq_len, _ = x.shape

        # --- Compute Latent Vectors ---
        # Eq (9): Shared KV Latent (for current step)
        c_kv_cur = self.kv_down_proj(x) # Shape:
        # Eq (12): Query Latent (for current step)
        c_q_cur = self.q_down_proj(x)   # Shape:

        # --- Compute Compressed Components ---
        # Eq (13): Compressed Query Content
        q_c = self.q_c_proj(c_q_cur) # Shape:
        q_c = q_c.view(batch_size, seq_len, self.num_heads, self.compressed_head_dim) # Shape:

        # Eq (14): RoPE Query Component (Before RoPE application)
        q_r = self.q_r_proj(c_q_cur) # Shape:
        q_r = q_r.view(batch_size, seq_len, self.num_heads, self.rope_head_dim) # Shape:

        # Eq (15): RoPE Key Component (Shared, Before RoPE application)
        k_r_cur = self.k_r_proj(x) # Shape:
        k_r_cur = k_r_cur.unsqueeze(2) # Add singleton head dim:

        # --- Apply RoPE ---
        past_length = 0
        if past_key_value is not None:
            # Cache structure: (c_kv_cache, k_r_cache, past_length)
            past_length = past_key_value[1]

        # Get cos/sin for the combined sequence length
        cos, sin = self.rotary_emb(seq_len=past_length + seq_len) # Shape:

        # Create position_ids if not provided
        if position_ids is None:
             position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device)
             position_ids = position_ids.unsqueeze(0).view(-1, seq_len) # Shape:

        # Apply RoPE to q_r and k_r_cur using the correct positions
        q_r = apply_rotary_pos_emb(q_r, cos, sin, position_ids) # Shape:
        k_r_cur_rope = apply_rotary_pos_emb(k_r_cur, cos, sin, position_ids) # Shape:

        # --- KV Cache Handling ---
        if use_cache:
            if past_key_value is not None:
                c_kv_cache, k_r_cache, _ = past_key_value
                # Concatenate current items with cache
                c_kv = torch.cat([c_kv_cache, c_kv_cur], dim=1) # Shape:
                k_r_rope = torch.cat([k_r_cache, k_r_cur_rope], dim=1) # Shape:
            else:
                c_kv = c_kv_cur
                k_r_rope = k_r_cur_rope

            # Store updated cache: (latent_kv, rope_key, current_length)
            present_key_value = (c_kv, k_r_rope, past_length + seq_len)
            curr_seq_len = past_length + seq_len
        else:
            # Not using cache, use current values only
            c_kv = c_kv_cur
            k_r_rope = k_r_cur_rope
            present_key_value = None
            curr_seq_len = seq_len

        # --- Compute Full K and V (Explicitly, for non-absorbed inference/training) ---
        # NOTE: In a fully optimized MLA inference with matrix absorption,
        # k_c and v would NOT be explicitly computed here. The attention
        # calculation would use c_kv directly with absorbed matrices.

        # Eq (10): Compressed Key Content (from potentially cached c_kv)
        k_c = self.k_c_proj(c_kv) # Shape:
        k_c = k_c.view(batch_size, curr_seq_len, self.num_heads, self.compressed_head_dim) # Shape:

        # Eq (11): Compressed Value (from potentially cached c_kv)
        v = self.v_proj(c_kv) # Shape:
        v = v.view(batch_size, curr_seq_len, self.num_heads, self.head_dim) # Shape:

        # --- Prepare for Attention ---
        # Eq (16): Concatenate final Query
        q = torch.cat([q_c, q_r], dim=-1) # Shape:

        # Eq (17): Concatenate final Key
        # Expand shared k_r_rope to all heads: ->
        k_r_rope_expanded = k_r_rope.expand(batch_size, curr_seq_len, self.num_heads, self.rope_head_dim)
        k = torch.cat([k_c, k_r_rope_expanded], dim=-1) # Shape:

        # Transpose for attention calculation:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --- Attention Calculation (Standard Scaled Dot-Product) ---
        # Note: q shape is, k shape is
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if needed
        if is_causal:
            # Create mask for the full sequence length
            causal_mask = torch.triu(torch.ones(curr_seq_len, curr_seq_len,
                                                dtype=torch.bool, device=x.device), diagonal=1)
            # If using cache, only mask the new query tokens against the full key sequence
            if past_length > 0:
                 # Select the rows corresponding to the new query tokens
                 causal_mask_slice = causal_mask[past_length:, :] # Shape
            else:
                 causal_mask_slice = causal_mask # Shape

            # Reshape mask for broadcasting: ->
            causal_mask_slice = causal_mask_slice.unsqueeze(0).unsqueeze(0)
            # Apply mask to scores (shape)
            scores = scores.masked_fill(causal_mask_slice, torch.finfo(scores.dtype).min)


        # Apply external attention mask (e.g., for padding)
        if attention_mask is not None:
            # Expected shape or
            if attention_mask.dim() == 2: # -> ->
                attention_mask = attention_mask[:, None, :, None].expand(-1, -1, -1, curr_seq_len)
            # Add mask values (0 for attend, -inf for ignore)
            scores = scores + attention_mask

        # Apply softmax and compute weighted sum
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype) # Use float32 for stability
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v) # Shape:

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        output = self.out_proj(context)

        if use_cache:
            return output, present_key_value
        else:
            return output, None # Return None for cache when not used

    def prepare_for_generation(self):
        """
        Placeholder for computing absorbed matrices for optimized inference.
        Actual computation logic needs to be implemented based on matrix
        absorption principles for the DeepSeek V2 MLA structure.
        """
        if self.generation_ready:
            return

        # --- TODO: Implement computation of absorbed matrices ---
        # Example (Conceptual - requires actual derivation):
        # q_absorbed_weight = compute_absorbed_q_weight(self.q_down_proj, self.q_c_proj, self.k_c_proj)
        # o_absorbed_weight = compute_absorbed_o_weight(self.v_proj, self.out_proj)
        # self.register_buffer('q_absorbed_weight', q_absorbed_weight)
        # self.register_buffer('o_absorbed_weight', o_absorbed_weight)

        self.generation_ready = True
        print("Warning: Matrix absorption computation in prepare_for_generation is not implemented.")


    def set_use_absorption(self, use_absorption):
        """Toggle whether to use matrix absorption during inference (if implemented)."""
        if use_absorption and not self.generation_ready:
             print("Warning: Cannot use absorption, matrices not prepared. Call prepare_for_generation first.")
             self.use_absorption = False
        elif use_absorption and self.q_absorbed_weight is None:
             print("Warning: Cannot use absorption, absorbed matrices not computed.")
             self.use_absorption = False
        else:
             self.use_absorption = use_absorption
             if use_absorption:
                  print("Note: Using matrix absorption requires the forward pass to be modified.")


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
    def __init__(self, hidden_dim, num_heads, ff_dim, latent_dim, dropout, max_seq_len, use_checkpointing=False):
        super().__init__()
        self.attention = MultiHeadLatentAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            kv_latent_dim=latent_dim,
            q_latent_dim=latent_dim,
            rope_head_dim=(hidden_dim // num_heads) // 4,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_checkpointing = use_checkpointing

    def _attention_block(self, normalized_x, attention_mask, position_ids, past_key_value, use_cache):
        return self.attention(
            normalized_x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )

    def _ff_block(self, x):
        return self.ff(self.norm2(x))

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False, position_ids=None):
        # First sublayer: MLA with residual connection
        normalized_x = self.norm1(x)

        if self.use_checkpointing and not use_cache and self.training:
            # Use checkpointing for attention if not using KV cache
            attn_output, _ = checkpoint(
                self._attention_block,
                normalized_x, attention_mask, position_ids, None, False,
                use_reentrant=False,
            )
            # No KV cache with checkpointing
            present_key_value = None
        else:
            # Regular forward pass with potential KV cache
            attn_output, present_key_value = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

        x = x + self.dropout(attn_output)

        # Second sublayer: FFN with residual connection
        if self.use_checkpointing and self.training:
            ff_output = checkpoint(self._ff_block, x, use_reentrant=False)
        else:
            ff_output = self.ff(self.norm2(x))

        x = x + self.dropout(ff_output)

        if use_cache:
            return x, present_key_value
        else:
            return x


# Full Transformer Model
class MLATransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            ff_dim,
            latent_dim,
            dropout,
            max_seq_len,
            use_checkpointing=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.use_checkpointing = use_checkpointing
        self.num_heads = num_heads

        # Use torch.nn.ModuleList for better memory efficiency with checkpoint
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim,
                num_heads,
                ff_dim,
                latent_dim,
                dropout,
                max_seq_len,
                use_checkpointing=use_checkpointing
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Apply custom initialization
        self.apply(self._init_weights)

        # Tie weights (after initialization)
        self.lm_head.weight = self.embedding.weight

        # Flag for generation optimization
        self.generation_ready = False

    def set_layer_bf16(self, i: int):
        self.layers[i].bfloat16()

    def set_layer_fp32(self, i: int):
        self.layers[i].float32()

    def set_layer_fp16(self, i: int):
        self.layers[i].float16()

    def set_layer_attention_bf16(self, i: int):
        self.layers[i].attention.bfloat16()

    def set_layer_attention_fp32(self, i: int):
        self.layers[i].attention.float32()

    def set_layer_attention_fp16(self, i: int):
        self.layers[i].attention.float16()

    def set_layer_ff_bf16(self, i: int):
        self.layers[i].ff.bfloat16()

    def set_layer_ff_fp32(self, i: int):
        self.layers[i].ff.float32()

    def set_layer_ff_fp16(self, i: int):
        self.layers[i].ff.float16()

    def _init_weights(self, module):
        """Initialize the weights - critical for stable training"""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with small random values
            # Using a Kaiming normal distribution suitable for GELU activation
            torch.nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with small normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize LayerNorm with ones and zeros
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MultiHeadLatentAttention):
            # Special initialization for attention matrices
            for name, param in module.named_parameters():
                if 'weight' in name:
                    # Attention weights need careful initialization
                    # Scale by head dimension
                    torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(self.hidden_dim // self.num_heads))
                elif 'bias' in name and param is not None:
                    torch.nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.shape

        # --- FIX: Generate position_ids ---
        past_length = 0
        if past_key_values is not None and past_key_values is not None:
            # Assuming cache structure is (c_kv_cache, k_r_cache, past_len)
            past_length = past_key_values[2]

        position_ids = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device)
        # Expand position_ids to match batch size
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # --- End position_ids generation ---

        # Embed tokens and positions
        x = self.embedding(input_ids)

        # Process with attention and KV cache
        present_key_values_list = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past_key_value = past_key_values[i] if past_key_values is not None else None

            # --- FIX: Pass position_ids to the layer ---
            layer_output = layer(
                x,
                attention_mask=attention_mask,
                past_key_value=layer_past_key_value,
                use_cache=use_cache,
                position_ids=position_ids  # Pass generated position_ids
            )

            # Unpack output based on use_cache
            if use_cache:
                x = layer_output
                present_key_values_list.append(layer_output[1])
            else:
                # layer() returns only x when use_cache=False now
                x = layer_output

        x = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, present_key_values_list
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
