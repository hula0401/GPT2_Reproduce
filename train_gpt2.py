from dataclasses import dataclass
import math
import os
import time
import inspect
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------------------------------------

batch_size = 64
block_size = 256
max_iters =500
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 12
droupout = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length and embd_dimensions
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # flash attention
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1)
        #y = att @ v

        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # type: ignore
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T), T-long sequences of tokens and B-batch of independent samples
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)  # type: ignore
        tok_embd = self.transformer.wte(idx)  # type: ignore
        x = tok_embd + pos_embd
        for block in self.transformer.h:  # type: ignore
            x = block(x)
        x = self.transformer.ln_f(x)  # type: ignore
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    ## copy directly from github repo
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2] # >=2 dimension in decay
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2] # 1 dimension in no decay
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype = torch.long)
    return ptt

class DataloaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val']

        #with open("data/tinyshakespeare.txt", "r") as f:
        #    text = f.read()
        #    enc = tiktoken.get_encoding("gpt2")
        #    tokens = enc.encode(text)
        #    self.tokens = torch.tensor(tokens)
        #    print(f"loaded tokens: {len(self.tokens)}")
        #    print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        #    self.current_postion = 0

        #enc = tiktoken.get_encoding("gpt2")
        #tokens = enc.encode(text)
        #self.tokens = torch.tensor(tokens)
        #print(f"loaded tokens: {len(self.tokens)}")
        #print(f"1 epoch = {len(self.tokens) // (B*T)} batches")
        #self.current_postion = self.B * self.T * self.process_rank

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split: {split}"
        if master_process:
            print(f"found {len(shards)} shards for split: {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_postion = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_postion:self.current_postion+B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_postion += B*T * self.num_processes
        # met problem that the last batch is not complete, so add another B*T
        if self.current_postion+ B*T*self.num_processes+1 > len(self.tokens):
            self.current_postion = self.B * self.T * self.process_rank
        return x,y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
#model = GPT.from_pretrained('gpt2')
#print('it works!')
# -----------------------------------------------------------------------------------------------------------
# Set up DDP and device
# prepare for dataloader, model, optimizer, etc.
# parameters
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "cuda is not available for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK", 0))
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# add batch size automloader
total_batch_size = 524288
B =8
T = 1024
assert total_batch_size % (B*T* ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f">= calculated grad accum steps: {grad_accum_steps}")

#print("I am GPU", ddp_rank) # I am GPU 0
#import sys; sys.exit(0)

train_loader = DataloaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'train')
val_loader = DataloaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split = 'val')

torch.set_float32_matmul_precision('high') # this is optional for A100

model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
use_compile = False
if use_compile:
    model = torch.compile(model)
# DDP will synchronize the model parameters / gradients across all the processes
# forward and backward are almost identical /unchanged for each process
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------------------------------------
# parameters
max_lr = 3e-4
min_lr = max_lr * 0.1
# scale down to 20% of the whole datasets
warmup_steps = 143 # 375e6 / 2**9 = 715
max_steps = 3814 #3814 # log2(524288) = 19, 10e9 / 2**19 = 19073
eval_step = 75

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_.txt")
with open(log_file, "w") as f:
    pass

# -----------------------------------------------------------------------------------------------------------
# Start training
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # evaluate loss
    if step % eval_step == 0 or last_step: # 100
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Step{step:4d} | val_loss: {val_loss_accum.item():.4f} | time: {time.strftime("%H:%M:%S", time.localtime())}")
            with open(log_file, "a") as f:
                f.write(f"step {step} | val: {val_loss_accum.item():.4f} | time: {time.strftime("%H:%M:%S", time.localtime())}\n")
            with open("log/val_loss_history.txt", "a") as f:
                f.write(f"{val_loss_accum:.6f}\n")
            if step > 0 and (step % 1000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
    
    # evaluate hellaswag -- benchmark
    if (step % eval_step == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step} | hella: {acc_norm:.4f}\n")
            with open("log/acc_history.txt", "a") as f:
                f.write(f"{acc_norm:.6f}\n")
    
    # generate samples
    if (step % eval_step == 0 or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to('cuda')
        sample_rng = torch.Generator(device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad(): # not count backward pass
                logits, loss = model(xgen)
                logits = logits[:,-1,:] # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                # top k sampling of 50, sampling only happen in top 50 to avoid werid tokens
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_indices, dim=-1, index=ix)
                xgen = torch.cat((xgen, xcol), dim=1)
                # this will generate 5x30
        if master_process:
            with open("log/generated_samples.txt", "a") as f:
                f.write(f"\n--- Step {step} ---\n")
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    sample_text = f"Sample {i} : {decoded}"
                    #print(sample_text)
                    f.write(sample_text + "\n")
    
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # softmax, layernorm, adam... are set to BF16 while others are still TF32
            logits, loss = model(x, y)
            #import code; code.interact(local=locals()) # print logits.dtype --> get 'torch.float32'
        # accumulate loss
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # the length of the gradient norm is no more than 0.1
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)
    tpkens_processed= train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tpkens_processed / dt 
    if master_process:
        print(f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}") 
        with open(log_file, "a") as f:
            f.write(f"Step {step:4d} | loss: {loss_accum.item():.6f}\n")
        with open("log/loss_history.txt", "a") as f:
            f.write(f"{loss_accum.item():.6f}\n")

# Save final model
if master_process:
    final_checkpoint = {
        'step': max_steps,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_accum.item(),
        'config': raw_model.config
    }
    final_path = os.path.join(log_dir, 'final_model.pt')
    torch.save(final_checkpoint, final_path)
    print(f"Saved final model to {final_path}")



import sys; sys.exit(0)

# -----------------------------------------------------------------------------------------------------------
# for model testing
#model = GPT.from_pretrained('gpt2')
model.eval() 
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')

# generate what's next after the last token
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad(): # not count backward pass
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        # top k sampling of 50, sampling only happen in top 50 to avoid werid tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        xcol = torch.gather(topk_indices, dim=-1, index=ix)
        x = torch.cat((x, xcol), dim=1)
        # this will generate 5x30
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
# -----------------------------------------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved config
    model = GPT(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from step {checkpoint['step']} with loss {checkpoint['loss']:.6f}")
    return model

# Example usage:
# model = load_model_from_checkpoint('log/final_model.pt', device='cuda')
# model = load_model_from_checkpoint('log/checkpoint_step_1000.pt', device='cuda')
