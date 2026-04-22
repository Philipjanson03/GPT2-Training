import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from numpy import dtype
from torch.nn import functional as F
import os
from pathlib import Path


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

        k = k.view(B,T, self.n_head, C//self.n_head)
        q = q.view(B,T, self.n_head, C//self.n_head)
        v = v.view(B,T, self.n_head, C//self.n_head)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2,-1)) * (1 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
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

    def forward(self, idx):
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
        return logits

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

num_return_sequense = 5
max_length = 30

model =GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')
print("worked")

from transformers import  GPT2Tokenizer
enc = GPT2Tokenizer.from_pretrained(get_local_model_path('gpt2'))
tokens = enc.encode("I play electric guitar")
tokens = torch.tensor(tokens, dtype= torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequense,1)
x = tokens.to('cuda')
print(x)