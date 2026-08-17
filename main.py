"""
Sehnsucht — FastAPI backend for Render.com's free web service tier.

No Gradio, no Hugging Face hardware gating. Serves:
  GET  /            -> a minimal chat UI (static/index.html)
  POST /generate    -> { "prompt": "...", "max_new_tokens": 200, "temperature": 0.7 }
                        -> { "response": "..." }

Render spins this service down after ~15 min idle and wakes it on the
next request — same on-demand behavior as the Gradio plan, just on a
host that doesn't require a paid plan to run Python.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# Weights live on Hugging Face Hub (plain Model repo — free, no size limit
# problem like GitHub, and unrelated to the Spaces hardware paywall).
# Replace with your actual repo id after uploading the two files there.
HF_REPO_ID = "Durveshjadhav/sehnsucht-model_v1"

MODEL_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pth")
TOKENIZER_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename="tokenizer.model")
DEVICE = "cpu"

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

checkpoint = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))

vocab_size = sp.get_piece_size()

block_size = 256
embed_size = 512
heads = 8
layers = 8


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        T = x.shape[0]
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x)
        return weights @ v


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = embed_size // heads
        self.attn = MultiHead(heads, head_size)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.Sequential(*[Block() for _ in range(layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.ln(x)
        return self.head(x)


model = MiniGPT()
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Optional: shrinks RAM/CPU load, worth trying given Render free tier's
# 512MB RAM ceiling. All this model's weight is in nn.Linear layers.
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )


def encode(s):
    return sp.encode(s)


def decode(t):
    return sp.decode(t)


@torch.no_grad()
def run_generation(prompt: str, max_new_tokens: int, temperature: float) -> str:
    idx = torch.tensor(encode(prompt), dtype=torch.long)
    for _ in range(max_new_tokens):
        idx_cond = idx[-block_size:]
        logits = model(idx_cond)
        logits = logits[-1] / temperature
        probs = torch.softmax(logits, dim=0)
        next_id = torch.multinomial(probs, 1).item()
        idx = torch.cat([idx, torch.tensor([next_id])])
    return decode(idx.tolist())


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    response: str


app = FastAPI(title="Sehnsucht")


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.prompt.strip():
        return GenerateResponse(response="Please enter a prompt.")
    text = run_generation(req.prompt, req.max_new_tokens, req.temperature)
    return GenerateResponse(response=text)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serves static/index.html at "/" so this same service is your web app too
app.mount("/", StaticFiles(directory="static", html=True), name="static")
