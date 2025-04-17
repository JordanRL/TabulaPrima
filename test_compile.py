import json
import os

import torch, time

from model_arch import MLATransformer

model_def_path = os.path.join('configs', 'model_defs', '255m_params.json')
train_def_path = os.path.join('configs', 'training_defs', '255m_params_2070.json')
with open(model_def_path, 'r') as f:
    model_def = json.load(f)

# Update model parameters from the loaded definition
HIDDEN_DIM = model_def.get('HIDDEN_DIM', 0)
NUM_LAYERS = model_def.get('NUM_LAYERS', 0)
NUM_HEADS = model_def.get('NUM_HEADS', 0)
HEAD_DIM = model_def.get('HEAD_DIM', 0)
FF_DIM = model_def.get('FF_DIM', 0)
MLA_LATENT_DIM = model_def.get('MLA_LATENT_DIM', 0)
MAX_SEQ_LENGTH = model_def.get('MAX_SEQ_LENGTH', 0)
ROPE_HEAD_DIM = model_def.get('ROPE_HEAD_DIM', 0)
COMPRESSED_HEAD_DIM = model_def.get('COMPRESSED_HEAD_DIM', 0)
KV_LATENT_DIM = model_def.get('KV_LATENT_DIM', 0)
Q_LATENT_DIM = model_def.get('Q_LATENT_DIM', 0)

with open(train_def_path, 'r') as f:
    train_def = json.load(f)

# Update training parameters
DROPOUT = train_def.get('DROPOUT', 0)
BATCH_SIZE = train_def.get('BATCH_SIZE', 0)
GRAD_STEPS = train_def.get('GRAD_STEPS', 0)
LEARNING_RATE = train_def.get('LEARNING_RATE', 0)
TOK_PER_PARAM = train_def.get('TOK_PER_PARAM', 0)
WEIGHT_DECAY = train_def.get('WEIGHT_DECAY', 0)

model = MLATransformer(
    vocab_size=100257,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    kv_latent_dim=KV_LATENT_DIM,
    q_latent_dim=Q_LATENT_DIM,
    dropout=DROPOUT,
    max_seq_len=MAX_SEQ_LENGTH,
    rope_head_dim=ROPE_HEAD_DIM,
    use_checkpointing=False,
    use_fusion=True
).cuda().eval()
compiled = torch.compile(model, backend="inductor", mode="default")
print("compiled")
# Prepare dummy input_ids: integers in [0, vocab_size), shape [batch_size, seq_len]
batch_size = BATCH_SIZE or 1
seq_len = MAX_SEQ_LENGTH or 1
vocab_size = model.embedding.num_embeddings
dummy = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long)
t0 = time.time()
_ = compiled(dummy)  # <-- expensive compile + run
print("first call:", time.time() - t0)
t1 = time.time()
_ = compiled(dummy)  # <-- pure run
print("second call:", time.time() - t1)