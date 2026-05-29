import inspect
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from IPython.core.pylabtools import figsize
from numpy import dtype
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
import os
from pathlib import Path
from transformers import GPT2Tokenizer

import torch._inductor.config as inductor_config

# Fix Windows filesystem race conditions
inductor_config.coordinate_descent_tuning = True
inductor_config.fx_graph_cache = True
inductor_config.max_autotune = False
inductor_config.shape_padding = False

# Performance improvements
inductor_config.triton.cudagraphs = True  # Enable CUDA graphs for kernel fusion
inductor_config.epilogue_fusion = True  # Fuse epilogue operations
inductor_config.pattern_matcher = True  # Enable pattern matching optimizations

def get_local_model_path(model_name):
    hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
    model_cache = hf_home / 'hub' / f'models--{model_name.replace("/", "--")}'

    if not model_cache.exists():
        raise FileNotFoundError(f"Model {model_name} not found in cache at {model_cache}")


    snapshots = list((model_cache / 'snapshots').glob('*'))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found for {model_name}")

    return str(snapshots[0])


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CasualSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query and value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANO_SCALE_INIT = 1# flag
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # it's not really a bias, more of a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)

        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2,-1)) * (1 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim = -1)
        # y = att @ v
        # -- flash attention : this method makes it so there's no reads/writes to HBM and everything is happening i shared memory because of online softmax calculation
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C)# reassemble all the head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_SCALE_INIT = 1# flag

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # initilize params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module , 'NANO_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # scaling down the activations to control the growth in residual path
            torch.nn.init.normal_(module.weight, mean = 0.0 , std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0, std = 0.02)



    def forward(self, idx, targets = None):
        B,T = idx.size()
        assert T <= self.config.block_size, f"cannot forward sequance length of {T}"
        pos = torch.arange(0, T, dtype= torch.long, device= idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the block of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)# (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in ['gpt2', 'gpt2_xl', 'gpt2_medium', 'gpt2_large']
        from transformers import GPT2LMHeadModel
        print(f'Loading {model_type} model')

        #n_layer, n_head and n_embd are determined from model_type
        config_args={
            'gpt2':     dict(n_layer = 12, n_head = 12, n_embd = 768), # 124M params
            'gpt2_medium': dict(n_layer = 24, n_head = 16, n_embd = 1024), #350M params
            'gpt2_large': dict(n_layer = 36, n_head = 20, n_embd = 1280), #774M params
            'gpt2_xl': dict(n_layer = 48, n_head = 25, n_embd = 1600), #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoint
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoint
        # create a from scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer

        model_path = get_local_model_path(model_type) #To use the local GPT2 version i have to specify the path
        model_hf = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all the candidate parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups any parameters that are 2D or more will be weight decayed
        # example: all the weights in matmul + Embeddings will decay, no bias nor Layer norms
        # this decay will create something like a gravity and makes the optimization (like a regularization) to use more of the weights and won't let a particular weight/ weights to be too large and this will distribute the work across more channels
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num of decayed parameter tensors : {len(decay_params)}, with {num_decay_params} parameters')
        print(f'num of non-decayed parameter tensors : {len(nodecay_params)}, with {num_nodecay_params} parameters')
        # Create AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95),
                                      eps=1e-8, fused=use_fused)
        return optimizer
def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}


        # at init load tokens from disk and store them in memory
        # data_root = 'edu_fineweb10B'
        data_root = 'C:/training_data/finsight/tokens/phase3_persian'
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root,s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f'no shards found for split {split}'
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state , init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T *  self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]# +1 is to get the target token for the last token in the batch
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B,T)
        self.current_position += B * T * self.num_processes
        #if loading the next batch goes out of bound, advance to next shard
        if self.current_position + (B * T * self.num_processes +1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            # shuffle tokens at epoch boundary
            perm = torch.randperm(len(self.tokens))
            self.tokens = self.tokens[perm]
        return x, y

def get_lr (step, max_steps, max_lr,min_lr,warmup_steps):
    if step <  warmup_steps:
        return max_lr * ((step + 1) / warmup_steps)
    elif step >= max_steps:
        return min_lr
    else:
        decay_ratio =  (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0<= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


num_return_sequences = 5
max_length = 30
# number of steps to validate the performance on laptop
max_steps = 5
# # steps for the final training
# max_steps = 19073
max_lr = 6e-4
min_lr = max_lr * 0.1
## ideal warmup steps is 10% of the max steps
# warmup_steps = max(1, max_steps // 10)
# warm up schedule GPT-3 used
warmup_steps = 715
norm_history = []

import time

#-------------------------------------------------------------------------------------------------------------------------
# torchrun --standalone --nproc-per-node=8 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
#set up  ddp( distributed data parallel)
ddp = int(os.environ.get("RANK", -1)) != -1 # to verify if it's a ddp run
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to RANK
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do the logging, checking, printing etc.
else:
    # vanilla , no ddp run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to auto detect device
    device = "cpu"
    # get a data batch
    if torch.cuda.is_available():
        device = "cuda"
#-------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# accumulation steps to be able to train on 0.5M batches so we can train the model faithful to the original parameters
total_batch_size = 524288
B = 8
T = 1024
assert total_batch_size % (B * T) == 0
accumulation_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderLite(B= B,T= T , process_rank = ddp_rank, num_processes = ddp_world_size, split='train')
val_loader = DataLoaderLite(B= B, T= T, process_rank = ddp_rank, num_processes = ddp_world_size, split='val')
torch.set_float32_matmul_precision('high')
# model =GPT.from_pretrained('gpt2')
model = GPT(GPT2Config(vocab_size=50304))
print(device)
model.to(device)

# for evaluation and generating samples you might need to disable the compilation
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contain the "raw" unwrapped model

#optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr = max_lr, betas=(0.9,0.95), eps = 1e-8,fused=True)
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = max_lr, device= device)
for i in range(max_steps):
    lr = get_lr(i, max_steps, max_lr, min_lr, warmup_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    t0 = time.time()
    if i % 100 == 0:
        raw_model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device_type= device, dtype= torch.bfloat16):
                    logits, loss = raw_model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f'validation loss: {val_loss_accum.item():.4f}')

    if master_process and i % 500 == 0:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': i,
            'val_loss': val_loss_accum.item() if 'val_loss_accum' in dir() else None,
        }
        torch.save(checkpoint, f'checkpoint_step{i}.pt')
    torch._dynamo.disable(raw_model)
    if i > 0 and i % 100 == 0:
        raw_model.eval()
        enc = GPT2Tokenizer.from_pretrained(get_local_model_path('gpt2'))
        tokens = enc.encode("Hello I am a Language model and i am capable of")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device= device)
        sample_rng.manual_seed(42 + ddp_rank)
        with torch.no_grad():
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                logits , loss = raw_model(xgen)

                logits = logits [:, -1, :]

                probs = F.softmax(logits, dim=-1)

                topk_probs, topk_indicies = torch.topk(probs, 50, dim=-1)

                ix = torch.multinomial(topk_probs, 1 , generator=sample_rng)

                x_col = torch.gather(topk_indicies, -1, ix)

                xgen = torch.cat((xgen, x_col ), dim = 1)

        for seq_idx in range (num_return_sequences):
            tokens = xgen[seq_idx, :max_length].tolist()
            decode = enc.decode(tokens)
            print(f'rank{ddp_rank} sample {seq_idx}:  {decode}')

    torch._dynamo.enable(raw_model)


    # training loop
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro_step in range(accumulation_steps):
        x , y = train_loader.next_batch()
        x , y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type= device, dtype= torch.bfloat16):
            logits, loss = model(x, y)
            # scaling the loss and in the loss_accum it will be added "accumulation_steps" times
        loss = loss / accumulation_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync(micro_step == accumulation_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    norm_history.append(norm.item())
    optimizer.step()
    torch.cuda.synchronize()# wait for GPU to finish
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (accumulation_steps * train_loader.B * train_loader.T * ddp_world_size) / (dt/1000)
    if master_process:
        print(f"Step{i} | the Loss is : {loss_accum.item():.6f} | norm:{norm:.4f} | dt : {dt:.2f} ms | tokens / sec : {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()

import sys; sys.exit(0)


model.eval()


from transformers import  GPT2Tokenizer
enc = GPT2Tokenizer.from_pretrained(get_local_model_path('gpt2'))
tokens = enc.encode("I play electric guitar")
tokens = torch.tensor(tokens, dtype= torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# x is (B,T) where B = 5 and T = 8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get logits
    with torch.no_grad():
        logits,_ = model(x)
        # take the logits at the last position
        logits = logits[:, -1, :]
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default) we keep top 50 this way we insure the model doesn't go off rails so easily
        topk_probs , topk_indices = torch.topk(probs, 50, dim= -1)
        # select a token from top-k probabilities
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)
        # append to the sequence
        x = torch.cat((x, xcol), dim= 1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decode = enc.decode(tokens)
    print(">",decode)