"""
Distributed nanogpt training with the Schatten-Muon (SMuon) optimizer,
reading the preprocessed FineWeb .bin shards (nanogpt format, magic 20240520).

Launch under torchrun:

    export FINEWEB10B_DIR=/path/to/fineweb10B
    torchrun --standalone --nproc_per_node=8 train_smuon_ddp.py

It also runs as `python train_smuon_ddp.py` for single-GPU debugging.

Distributed design:

* DDP averages gradients across ranks during backward; gradient sync is
  skipped on every microbatch except the last one with ddp_model.no_sync(),
  so DDP triggers exactly one all_reduce per optimizer step.
* SMuonWithAuxAdam (multi-GPU class) shards the heavy fractional-power update
  across ranks: rank r computes the Schatten update for params[base_i + r]
  only, then all_gather syncs the updated weights so every rank ends up with
  identical parameters. Auto-detects dist.is_initialized().
* ActivationRecorder uses Gram matrices (use_gram=True). Each rank captures
  A^T A locally on its data shard; update_p_state internally all_reduces those
  Grams across ranks before computing p*, so the per-layer Schatten exponent
  reflects activation curvature on the global batch.
"""

import os
from pathlib import Path

_lr = os.environ.get("LOCAL_RANK", "0")
_home = os.environ.get("HOME", "/tmp")
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR", f"{_home}/.cache/torchinductor_lr{_lr}"
)
os.environ.setdefault("TRITON_CACHE_DIR", f"{_home}/.cache/triton_lr{_lr}")
os.environ.setdefault("WANDB_MODE", "offline")

import sys

import copy
from functools import lru_cache
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
import glob
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from smuon.optimizers.adaptive import SMuonWithAuxAdam
from smuon.optimizers.baseline import MuonWithAuxAdam
from smuon.optimizers.adamuon import AdaMuon
from smuon.optimizers.muadam import MuAdam
from smuon.wrap_model import ActivationRecorder

# -----------------------------------------------------------------------------
# Model definitions.


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # Half-truncated RoPE by @YouJiacheng
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x):  # x: (B, T, n_head, head_dim)
        cos = self.cos[None, :x.size(1), None, :]
        sin = self.sin[None, :x.size(1), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1).type_as(x)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


def next_multiple_of_n(v: int, n: int):
    return (v + n - 1) // n * n

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    _GATE_DIM = 12

    def __init__(self, config, max_seq_len: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, max_seq_len)
        # Sparse gated attention: gates each head's output, zero-init = identity at start
        self.attn_gate = nn.Linear(self._GATE_DIM, self.n_head, bias=False)
        nn.init.zeros_(self.attn_gate.weight)

    def forward(self, x, ve, sa_lambdas, block_mask):
        B, T, C = x.size()
        assert B == 1, "FlexAttention requires batch size 1"
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        # QK norm
        q = rmsnorm(q)
        k = rmsnorm(k)
        # RoPE
        q = self.rotary(q)
        k = self.rotary(k)
        # Value residual mixing with learnable sa_lambdas
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v
        # FlexAttention
        y = flex_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        block_mask=block_mask, scale=0.12,
        ).transpose(1, 2)
        # Sparse gate per head
        gate = torch.sigmoid(self.attn_gate(x[..., :self._GATE_DIM]))  # (B, T, n_head)
        y = (y * gate.unsqueeze(-1)).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        std   = 0.5 * (config.n_embd ** -0.5)
        bound = (3 ** 0.5) * std
        nn.init.uniform_(self.c_fc.weight, -bound, bound)
        nn.init.zeros_(self.c_proj.weight)   # zero-init output projection

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self, config, max_seq_len: int, layer_idx: int):
        super().__init__()
        # Skip attention in layer 7 (empirically beneficial for 12-layer models)
        self.attn = CausalSelfAttention(config, max_seq_len) if layer_idx != 7 else None
        self.mlp  = MLP(config)

    def forward(self, x, ve, x0, lambdas, sa_lambdas, block_mask):
        # Blend hidden state with initial embedding residual
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(rmsnorm(x), ve, sa_lambdas, block_mask)
        x = x + self.mlp(rmsnorm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config, max_seq_len: int):
        super().__init__()
        self.config = config
        padded = next_multiple_of_n(config.vocab_size, 128)
        self.transformer = nn.ModuleDict(dict(
        wte=nn.Embedding(padded, config.n_embd),
            value_embeds=nn.ModuleList([nn.Embedding(padded, config.n_embd) for _ in range(3)]),
            h=nn.ModuleList([Block(config, max_seq_len, i) for i in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, padded, bias=False)
        nn.init.zeros_(self.lm_head.weight)   # untied, zero-init
        # NOTE: wte is no longer tied to lm_head

        assert config.n_layer % 2 == 0
        n = config.n_layer
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(n // 2),                        # skip_weights
            torch.tensor([1.0, 0.0]).repeat(n),        # block lambdas  (x, x0)
            torch.tensor([0.5, 0.5]).repeat(n),        # SA lambdas     (v, ve)
        ]))

    def forward(self, idx, targets=None, block_masks=None, return_logits=True):
        assert idx.ndim == 1   # FlexAttention: 1-D token sequence
        T = idx.shape[0]
        n = self.config.n_layer

        x = x0 = rmsnorm(self.transformer.wte(idx)[None])   # (1, T, C)

        ve_base = [ve(idx) for ve in self.transformer.value_embeds]
        ve_pattern = ve_base + [None] * (n - 2 * len(ve_base)) + ve_base

        skip_w    = self.scalars[:n // 2]
        blk_lam   = self.scalars[n // 2 : n // 2 + n * 2].view(n, 2)
        sa_lam    = self.scalars[n // 2 + n * 2 : n // 2 + n * 4].view(n, 2)

        # Alternate long/short pattern (matches modded-nanogpt for 12 layers)
        assert block_masks is not None
        bm_seq = block_masks  # just use directly

        skip_stack = []
        n_half = n // 2
        for i, block in enumerate(self.transformer.h):
            if i >= n_half:
                x = x + skip_w[i - n_half] * skip_stack.pop()
            x = block(x, ve_pattern[i], x0, blk_lam[i], sa_lam[i], bm_seq[i])
            if i < n_half:
                skip_stack.append(x)

        x = rmsnorm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / 7.5)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum" if self.training else "mean",
            )
        return (logits if return_logits else None), loss

def get_block_masks(x: torch.Tensor, step: int, device: str):
    """Document-causal sliding-window block masks, one per layer."""
    BLOCK_SIZE = 128
    T = x.shape[0]
    assert T % BLOCK_SIZE == 0
    NUM_BLOCKS = T // BLOCK_SIZE

    docs = (x == 50256).cumsum(0)  # which document each token belongs to

    def document_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx])

    def dense_to_ordered(dense_blockmask):
        num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
        indices = (dense_blockmask.argsort(dim=-1, descending=False, stable=True)
                   .flip(-1).to(torch.int32))
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=device)
    causal_any = block_idx[:, None] >= block_idx
    causal_all = block_idx[:, None] >  block_idx

    docs_low  = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
    docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
    doc_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
    doc_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)

    bm_any = causal_any & doc_any
    bm_all = causal_all & doc_all

    partial_kv_num, partial_kv_idx = dense_to_ordered(bm_any & ~bm_all)
    full_kv_num,    full_kv_idx    = dense_to_ordered(bm_all)

    def build_bm(w: int):
        wsb = torch.tensor(max(w, 1), dtype=torch.int32, device=device)
        return BlockMask.from_kv_blocks(
            torch.clamp_max(partial_kv_num,
                            torch.clamp_min(wsb - full_kv_num, 1)),
            partial_kv_idx,
            torch.clamp_max(full_kv_num, wsb - 1),
            full_kv_idx,
            BLOCK_SIZE=BLOCK_SIZE,
            mask_mod=document_causal,
        )

    w = int(get_window_size_blocks(step).item())
    long_bm  = build_bm(w)
    short_bm = build_bm(w // 2)
    return [long_bm,  short_bm, short_bm, short_bm,
            long_bm,  short_bm, short_bm, long_bm,
            short_bm, short_bm, short_bm, long_bm]


# -----------------------------------------------------------------------------
# Distributed Data Loader (preprocessed nanogpt-style .bin shards).

def _load_shard_pinned(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520
    assert header[1] == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    return tokens

def find_batch_starts(tokens, pos, seq_len, token_window):
    boundary_mask = tokens[pos: pos + token_window] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item()
        if end - start >= seq_len:
            starts.append(start)
            if len(starts) == ddp_world_size:
                return starts, end - pos
            start = end
    raise RuntimeError("increase token_window")

def distributed_data_generator(filename_pattern, seq_len, grad_accum_steps, align_to_bos):
    files = [Path(f) for f in sorted(glob.glob(filename_pattern))]
    file_iter = iter(files)
    tokens, pos = _load_shard_pinned(next(file_iter)), 0
    batch_size = seq_len * ddp_world_size
    while True:
        token_window = grad_accum_steps * (2 * batch_size if align_to_bos else batch_size)
        if pos + token_window + 1 >= len(tokens):
            tokens = _load_shard_pinned(next(file_iter))
            pos = 0
        for _ in range(grad_accum_steps):
            if align_to_bos:
                batch_starts, tokens_consumed = find_batch_starts(tokens, pos, seq_len, token_window)
                start_idx = batch_starts[ddp_rank]
            else:
                tokens_consumed = batch_size
                start_idx = pos + ddp_rank * seq_len
            buf = tokens[start_idx:][:seq_len + 1]
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            pos += tokens_consumed
            token_window -= tokens_consumed
            yield inputs, targets

def make_val_loader():
    return distributed_data_generator(
        args.input_val_bin, args.val_seq_len, 1, align_to_bos=False
    )

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print(f"ERROR: magic number mismatch in {filename}")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return int(header[2])


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


# -----------------------------------------------------------------------------
# Hyperparameters.

# Default to $FINEWEB10B_DIR if set; otherwise the explicit Jean Zay path the
# user has on $WORK. Override either with the env var or by editing args.
_FW_DIR = os.environ.get(
    "FINEWEB10B_DIR",
    "/path/to/fineweb10B",
)
optimizer_kind, moment_type = os.environ.get("OPTIMIZER", "smuon-none").lower().split("-")

@dataclass
class Hyperparameters:
    # 1. CLI Environment Variable for Optimizer Selection (Defaults to 'smuon')
    # Run via: OPTIMIZER=muon torchrun ... OR OPTIMIZER=smuon torchrun ...
    optimizer_kind: str = optimizer_kind
    
    # Data
    input_bin: str = f"{_FW_DIR}/fineweb_train_*.bin"
    input_val_bin: str = f"{_FW_DIR}/fineweb_val_*.bin"
    
    # Optimization
    # Modded script processes 393,216 tokens per global step (48*1024 * 8 GPUs).
    # To match this with sequence_length=1024, global batch_size = 384.
    batch_size: int = 8          # global sequences per step (each seq = 48*1024 tokens)
    device_batch_size: int = 1   # FlexAttention requires B=1
    sequence_length: int = 48 * 1024
    max_seq_len: int = 4 * 64 * 1024

    num_iterations: int = 1695 # Matched from Modded-NanoGPT
    learning_rate: float = 0.008 # Matched from Modded-NanoGPT DistAdam
    muon_lr = 0.05 * float(os.environ.get("LR_MULT", 1.0))
    
    warmup_iters: int = 0 
    # Modded-NanoGPT uses a 0.45 cooldown fraction (45% of 1695 steps)
    warmdown_iters: int = int(1695 * 0.45) # 762 steps
    weight_decay: float = 0.0
    
    # Eval / logging
    val_loss_every: int = 50 # Matched from Modded-NanoGPT
    val_seq_len: int = 2 * 64 * 1024   # 262144 — matches modded-nanogpt
    val_tokens: int = 20 * (4 * 64 * 1024) * 2  # 20 steps × val_seq_len × 2 GPUs = 10,485,760
    save_every: int = 0
    
    # SMuon-specific
    smuon_interval: int = 100
    pmin: float = 1.02
    pmax: float = 50.0
    init_p: str = "pmax"
    p_method: str = "exact_tightness"
    sv_momentum: float = 0.95
    moment_type: str= moment_type
    
    # Randomness
    seed: int = int(os.environ.get("SEED", 42))
    
    # wandb
    wandb_project: str = "smuon-nanogpt"
    # Dynamically update the run name based on the chosen optimizer
    wandb_run_name: str = f"{os.environ.get('OPTIMIZER', 'smuon').lower()}-seed-{os.environ.get('SEED', '42')}-lr{os.environ.get('LR_MULT', 1.0)}-sweep"

args = Hyperparameters()

@lru_cache(1)
def _window_blocks_tensor(w: int):
    return torch.tensor(w, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_window_size_blocks(step: int):
    x = step / args.num_iterations
    raw = int(1728 * x)
    # Round up to nearest 128, minimum 1 block
    w = max(next_multiple_of_n(raw, 128), 128)
    return _window_blocks_tensor(w)

# -----------------------------------------------------------------------------
# Distributed init.

assert torch.cuda.is_available()
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
else:
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda:0"

torch.cuda.set_device(device)
master_process = ddp_rank == 0

torch.manual_seed(args.seed + ddp_rank)
torch.cuda.manual_seed_all(args.seed + ddp_rank)

if master_process:
    print(f"using device: {device}, ddp_world_size: {ddp_world_size}")
    print(f"train shards: {args.input_bin}")
    print(f"val   shards: {args.input_val_bin}")

B, T = args.device_batch_size, args.sequence_length
val_steps = args.val_tokens // (args.val_seq_len * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0, (
    f"batch_size ({args.batch_size}) must be divisible by B*world_size ({B * ddp_world_size})"
)
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

train_accumulation_steps = 8 // ddp_world_size  # matches modded at 8-GPU equivalent
train_loader = distributed_data_generator(args.input_bin, T, train_accumulation_steps, align_to_bos=True)

if master_process:
    print(f"train_accumulation_steps={train_accumulation_steps}, val_steps={val_steps}")

# -----------------------------------------------------------------------------
# Model + activation recorder + DDP + compile.

num_vocab = 50257
raw_model = GPT(
    GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768),
    max_seq_len=args.max_seq_len,
).cuda()

# Hooks attach to the underlying nn.Linear modules; DDP and torch.compile both
# wrap the module hierarchy without removing them.
recorder = (
    ActivationRecorder(raw_model, use_gram=True)
    if args.optimizer_kind == "smuon"
    else None
)

if ddp:
    # DDP's constructor broadcasts rank-0 weights to every rank, so different
    # per-rank seeds above only diversify dropout / data noise.
    ddp_model = DDP(raw_model, device_ids=[ddp_local_rank])
else:
    ddp_model = raw_model

model = torch.compile(ddp_model)
ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

param_names = {p: n for n, p in raw_model.named_parameters()}

# -----------------------------------------------------------------------------
# SMuon optimizer.

muon_params = [
    p for n, p in raw_model.transformer.h.named_parameters()
    if p.ndim >= 2 and "attn_gate" not in n
]
adam_params = (
    list(raw_model.lm_head.parameters()) +
    list(raw_model.transformer.wte.parameters()) +        # wte now untied
    list(raw_model.transformer.value_embeds.parameters()) +
    [raw_model.scalars] +                                 # replaces skip_weights
    [p for n, p in raw_model.transformer.h.named_parameters() if "attn_gate" in n]
)

_groups = [
    dict(
        params=muon_params,
        use_muon=True,
        lr=args.muon_lr, # Modded-NanoGPT Muon LR
        momentum=0.95,
        beta2=0.95,
        weight_decay=args.weight_decay,
    ),
    dict(
        params=adam_params,
        use_muon=False,
        lr=args.learning_rate,
        betas=(0.8, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-10,
    ),
]

if args.optimizer_kind == "smuon":
    optimizer = SMuonWithAuxAdam(
        _groups,
        param_names=param_names,
        pmin=args.pmin,
        pmax=args.pmax,
        init_p=args.init_p,
        p_method=args.p_method,
        moment_type=args.moment_type,
        use_bias_correction=True,
    )
elif args.optimizer_kind == "muon":
    optimizer = MuonWithAuxAdam(_groups)
elif args.optimizer_kind == "adam":
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate)
elif args.optimizer_kind == "adamuon":
    optimizer = AdaMuon(_groups)
elif args.optimizer_kind == "muadam":
    optimizer = MuAdam(_groups)
else:
    raise ValueError(f"unknown optimizer_kind: {args.optimizer_kind!r}")

if master_process:
    print(f"optimizer: {args.optimizer_kind}")


def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / max(1, args.warmup_iters)
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.warmdown_iters


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
_warmup_steps = 17
_init_state = dict(
    model=copy.deepcopy(raw_model.state_dict()),
    optimizer=copy.deepcopy(optimizer.state_dict()),
)

for _ in range(_warmup_steps):
    xw, yw = next(train_loader)
    with ctx:
        _, loss = model(xw, yw,
                        block_masks=get_block_masks(xw, 1, device),
                        return_logits=False)
    (loss / train_accumulation_steps).backward()
    optimizer.step()
    model.zero_grad(set_to_none=True)
raw_model.load_state_dict(_init_state["model"])

optimizer.load_state_dict(_init_state["optimizer"])
del _init_state
train_loader = distributed_data_generator(args.input_bin, T, train_accumulation_steps, align_to_bos=True)

# -----------------------------------------------------------------------------
# Run / log setup.

run_id = str(uuid.uuid4()) if master_process else ""
if ddp:
    # Make all ranks agree on run_id so checkpoint paths stay consistent.
    run_id_list = [run_id]
    dist.broadcast_object_list(run_id_list, src=0)
    run_id = run_id_list[0]

if master_process:
    os.makedirs(f"logs/{run_id}/", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    with open(logfile, "w") as f:
        f.write(f"LR_MULT={os.environ.get('LR_MULT')}\n")
        f.write(f"OPTIMIZER={args.optimizer_kind}\n")
        f.write("=" * 100 + "\n")
        f.write(code)
        f.write("=" * 100 + "\n")
        f.write(
            f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\n"
        )
        f.write(f"ddp_world_size={ddp_world_size}\n")
        f.write(f"input_bin={args.input_bin}\n")
        f.write("nvidia-smi:\n")
        import subprocess

        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        f.write(f"{result.stdout}\n")
        f.write("=" * 100 + "\n")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or run_id,
        config={**vars(args), "run_id": run_id, "ddp_world_size": ddp_world_size},
    )

# -----------------------------------------------------------------------------
# Training loop.

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

train_loader = distributed_data_generator(args.input_bin, T, train_accumulation_steps, align_to_bos=True)
p_star_min = None
for step in range(args.num_iterations + 1):
    last_step = step == args.num_iterations
    if step == 32:
        torch.cuda.synchronize()
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float("nan") if step <= 33 else (step - 32) + 1

    # ---------------- Validation ----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        model.eval()
        val_gen = make_val_loader()
        val_loss = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(val_steps):
            x_val, y_val = next(val_gen)           # already 1D, no squeeze
            with torch.no_grad():
                bm_val = get_block_masks(x_val, step, str(device))  # actual step, not 0
                _, loss = model(x_val, y_val,
                                block_masks=bm_val,
                                return_logits=False)
                val_loss += loss.float()
        val_loss /= val_steps


        # Each rank validated on its own slice; average across ranks.
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss_f = float(val_loss.item())

        if master_process:
            print(
                f"step:{step}/{args.num_iterations} val_loss:{val_loss_f:.4f} "
                f"train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(timed_steps - 1, 1):.2f}ms"
            )
            with open(logfile, "a") as f:
                f.write(
                    f"step:{step}/{args.num_iterations} val_loss:{val_loss_f:.4f} "
                    f"train_time:{training_time_ms:.0f}ms "
                    f"step_avg:{training_time_ms / max(timed_steps - 1, 1):.2f}ms\n"
                )
            wandb.log(
                {"val/loss": val_loss_f, "train_time_ms": training_time_ms}, step=step
            )

        torch.cuda.synchronize()
        t0 = time.time()

    # ---------------- Checkpointing ----------------
    if master_process and (
        last_step or (args.save_every > 0 and step % args.save_every == 0)
    ):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        try:
            log = dict(
                step=step,
                code=code,
                model=raw_model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        except Exception as e:
            print(f"[warn] checkpoint save failed: {e}")
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # ---------------- Training step ----------------
    model.train()

    # Record activations on the first microbatch on every rank every
    # smuon_interval steps. With use_gram=True each rank stores its local
    # A^T A; update_p_state below all_reduces them into the global Gram.
    should_record = args.optimizer_kind == "smuon" and (
        ((step > 0) and (step % args.smuon_interval == 0)) or (step == 10)
    )
    for accum_step in range(train_accumulation_steps):
        x, y = next(train_loader)

        if accum_step == 0:
            bm = get_block_masks(x, 1, str(device))

        is_last_micro = accum_step == train_accumulation_steps - 1
        # Skip DDP grad all_reduce on every microbatch except the last.
        sync_ctx = ddp_model.no_sync() if (ddp and not is_last_micro) else nullcontext()

        with sync_ctx:
            with ctx:
                if should_record and accum_step == 0:
                    with recorder.recording():
                        _, loss = model(x, y,
                                        block_masks=bm,
                                        return_logits=False)
                else:
                    _, loss = model(x, y,
                                    block_masks=bm,
                                    return_logits=False)
                train_loss = loss.detach()
                loss = loss / train_accumulation_steps
            loss.backward()

    # Update p* (and alpha_star) using DDP-averaged grads and the all_reduced
    # global activation Grams. Every rank must call update_p_state so the
    # collective ops match.
    if should_record:
        optimizer.update_p_state(
            activations=recorder.get_activations(),
            use_gram=True,
        )
        recorder.clear()

        if master_process:
            p_state = optimizer.get_p_state_for_logging()
            log_dict, p_values, alpha_values = {}, [], []
            for name, st in p_state.items():
                safe = name.replace(".", "/")
                if st.get("p_star") is not None:
                    log_dict[f"p_star/{safe}"] = float(st["p_star"])
                    p_values.append(float(st["p_star"]))
            if p_values:
                log_dict["p_star/_mean"] = float(np.mean(p_values))
                log_dict["p_star/_min"] = float(np.min(p_values))
                log_dict["p_star/_max"] = float(np.max(p_values))
            p_star_max, p_star_min = max(p_values), min(p_values)
            wandb.log(log_dict, step=step)

    # SMuon's step(): each rank computes the Schatten update for its assigned
    # subset of params, then all_gather syncs every parameter back across ranks.
    muon_frac = min(step / 300, 1.0)
    for group in optimizer.param_groups:
        if group.get("use_muon", False):
            group["momentum"] = (1 - muon_frac) * 0.85 + muon_frac * 0.95

    optimizer.step()
    scheduler.step()
    model.zero_grad(set_to_none=True)

    # ---------------- Per-step train logging ----------------
    if ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        train_loss_f = float(train_loss.item()) / T
        if p_star_min is not None:
            print(
                f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss_f:.4f} "
                f"train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms "
                f"p_star (max):{p_star_max:.1f} p_star (min):{p_star_min:.1f}"
            )
            with open(logfile, "a") as f:
                f.write(
                    f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss_f:.4f} "
                    f"train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms "
                    f"p_star (max):{p_star_max:.1f} p_star (min):{p_star_min:.1f}\n"
                )
        else:
            print(
                f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss_f:.4f} "
                f"train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms"
            )
            with open(logfile, "a") as f:
                f.write(
                    f"step:{step + 1}/{args.num_iterations} train_loss:{train_loss_f:.4f} "
                    f"train_time:{approx_time:.0f}ms step_avg:{approx_time / timed_steps:.2f}ms\n"
                )
        last_lrs = scheduler.get_last_lr()
        wandb.log(
            {
                "train/loss": train_loss_f,
                "train/lr_muon": last_lrs[0] if len(last_lrs) > 0 else 0.0,
                "train/lr_adam": last_lrs[1] if len(last_lrs) > 1 else 0.0,
                "train_time_ms": approx_time,
            },
            step=step,
        )

# -----------------------------------------------------------------------------
# Cleanup.
if master_process:
    print(
        f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
    )
    wandb.finish()

if ddp:
    destroy_process_group()
