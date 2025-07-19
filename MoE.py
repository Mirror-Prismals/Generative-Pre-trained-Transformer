#!/usr/bin/env python3
# gpt_moe_trainer_v2.py — Refactored GPT‑MoE trainer
# 2025‑07‑19 • MIT‑0

from __future__ import annotations
import os, argparse, glob, shutil, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

import sentencepiece as spm

# ─── 1. CONFIG ────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    text_dir: Path
    vocab_dir: Path
    output_dir: Path = Path("./checkpoints")
    seq_len: int = 64
    batch_size: int = 8
    vocab_size: int = 32_000
    emb_dim: int = 512
    depth: int = 12
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    n_experts: int = 16
    k: int = 2
    aux_coef: float = 1e-2
    capacity_factor: float = 1.0
    gate_noise_init: float = 1e-1
    gate_noise_final: float = 0.0
    noise_anneal_steps: int = 10_000
    lr: float = 3e-4
    warmup_steps: int = 500
    clip_norm: float = 1.0
    epochs: int = 3
    val_split: float = 0.01
    checkpoint_keep: int = 3
    log_dir: Path = Path("./logs")

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("GPT‑MoE v2 trainer")
    for f in TrainConfig.__dataclass_fields__.values():
        name = f.name
        t = f.type
        default = f.default
        argname = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            p.add_argument(argname, action="store_true" if not default else "store_false")
        else:
            p.add_argument(argname, type=t, default=default)
    args = p.parse_args()
    return TrainConfig(**vars(args))

# ─── 2. TOKENIZER & VOCAB ─────────────────────────────────────────────────
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"

def train_sentencepiece(corpus_dir: Path, model_prefix: Path, vocab_size: int):
    model_file = model_prefix.with_suffix(".model")
    if model_file.exists(): return
    txt_files = sorted(glob.glob(str(corpus_dir / "**" / "*.txt"), recursive=True))
    if not txt_files:
        raise RuntimeError("No .txt files found for SentencePiece training.")
    spm.SentencePieceTrainer.Train(
        input=",".join(txt_files),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_piece=PAD_TOKEN, pad_id=0,
        unk_piece=UNK_TOKEN, unk_id=1
    )

class BPETokenizer:
    def __init__(self, model_file: Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.pad_id = self.sp.pad_id()
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode([i for i in ids if i != self.pad_id])

# ─── 3. DATASET WITH PER‑FILE SHARDS ────────────────────────────────────────
class FileShardDataset(Dataset):
    def __init__(self, txt_path: Path, tok: BPETokenizer, seq_len: int):
        self.seq_len = seq_len
        npy_path = txt_path.with_suffix(".npy")
        if not npy_path.exists():
            tokens = tok.encode(txt_path.read_text(encoding="utf8", errors="ignore"))
            np.save(npy_path, np.array(tokens, dtype=np.int32))
        self.arr = np.load(npy_path)
        # number of sequences
        self.count = max(0, len(self.arr) - seq_len)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        span = self.arr[idx : idx + self.seq_len + 1]
        return torch.tensor(span, dtype=torch.long)

def build_dataloaders(cfg: TrainConfig, tok: BPETokenizer):
    all_txt = sorted(glob.glob(str(cfg.text_dir / "**" / "*.txt"), recursive=True))
    datasets = [FileShardDataset(Path(p), tok, cfg.seq_len) for p in all_txt]
    full = torch.utils.data.ConcatDataset(datasets)
    val_size = int(len(full) * cfg.val_split)
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, drop_last=False)
    return train_dl, val_dl

# ─── 4. MoE MODULE WITH NOISE ANNEALING ────────────────────────────────────
class MoE(nn.Module):
    def __init__(self, d_model, d_hidden, n_experts, k, aux_coef, capacity_factor):
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        self.aux_coef = aux_coef
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden), nn.GELU(),
                nn.Linear(d_hidden, d_model)
            ) for _ in range(n_experts)
        ])
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self, x, noise_init, noise_final, anneal_steps):
        B, L, D = x.size()
        N = B * L
        flat = x.view(N, D)

        # compute current noise
        cur_step = self.step.item()
        frac = min(cur_step / anneal_steps, 1.0)
        noise_scale = noise_init * (1 - frac) + noise_final * frac
        self.step += 1

        # gating
        logits = self.gate(flat)
        if noise_scale > 0:
            logits = logits + torch.randn_like(logits) * noise_scale
        prob = F.softmax(logits, dim=-1)

        # top‑k & mask
        topk_vals, topk_idx = logits.topk(self.k, dim=-1)
        mask = torch.zeros_like(prob).scatter_(1, topk_idx, 1.0)
        gated = prob * mask

        # capacity enforcement
        cap = int(self.capacity_factor * N / self.n_experts)
        scores_t = gated.transpose(0,1)
        if cap < N:
            top_caps, _ = scores_t.topk(cap, dim=-1, largest=True)
            thresh = top_caps[:, -1]
            keep = (scores_t >= thresh.unsqueeze(1))
            gated = gated * keep.transpose(0,1)

        # renormalize
        gated = gated / (gated.sum(dim=1, keepdim=True) + 1e-9)

        # dispatch
        out_flat = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            idx_e = torch.nonzero(gated[:, e], as_tuple=True)[0]
            if idx_e.numel() == 0: continue
            inp_e = flat[idx_e]
            out_e = expert(inp_e)
            w = gated[idx_e, e].unsqueeze(1)
            out_flat[idx_e] += out_e * w

        out = out_flat.view(B, L, D)

        # aux loss
        imp = gated.sum(dim=0) / N
        load = (gated > 0).float().sum(dim=0) / N
        aux = 0.5 * self.aux_coef * self.n_experts * ((imp**2).sum() + (load**2).sum())

        return out, aux

# ─── 5. TRANSFORMER + GPT_MoE ──────────────────────────────────────────────
def make_causal_mask(sz):
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

class TransformerLayerMoE(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        d, h = cfg.emb_dim, cfg.n_heads
        self.attn = nn.MultiheadAttention(d, h, dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        hidden = int(d * cfg.mlp_ratio)
        self.moe = MoE(d, hidden, cfg.n_experts, cfg.k,
                       cfg.aux_coef, cfg.capacity_factor)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", make_causal_mask(cfg.seq_len))

    def forward(self, x, pad_mask, noise_init, noise_final, anneal_steps):
        B, L, D = x.size()
        # self‑attn
        y, _ = self.attn(x, x, x,
                         attn_mask=self.mask[:L, :L].to(x.device),
                         key_padding_mask=pad_mask)
        x = self.norm1(x + self.dropout(y))
        # MoE
        y2, aux = self.moe(x, noise_init, noise_final, anneal_steps)
        x = self.norm2(x + self.dropout(y2))
        return x, aux

class GPTMoE(nn.Module):
    def __init__(self, cfg: TrainConfig, vocab_size, pad_id):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.pad_id = pad_id

        # embeddings
        self.tok_emb = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        # layers
        self.layers = nn.ModuleList([
            TransformerLayerMoE(cfg) for _ in range(cfg.depth)
        ])

        # output head tied to tok_emb
        self.head = nn.Linear(cfg.emb_dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, inp, noise_init, noise_final, anneal_steps):
        B, Lp1 = inp.size()
        L = Lp1 - 1
        x = inp[:, :-1]
        pad_mask = x == self.pad_id
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(x) + self.pos_emb(pos)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.dropout(x)

        total_aux = x.new_zeros(1)
        for layer in self.layers:
            x, aux = layer(x, pad_mask,
                           noise_init, noise_final, anneal_steps)
            total_aux = total_aux + aux

        logits = self.head(x)
        return logits, total_aux

# ─── 6. TRAIN & EVAL LOOP ─────────────────────────────────────────────────
def save_checkpoint(state, cfg: TrainConfig, epoch: int):
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"ckpt_epoch{epoch}.pt.tmp"
    torch.save(state, fname)
    final = fname.with_suffix("")  # remove .tmp
    fname.replace(final)

    # remove old
    all_ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"), key=os.path.getmtime)
    while len(all_ckpts) > cfg.checkpoint_keep:
        all_ckpts.pop(0).unlink()

def train_and_validate(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # prepare vocab & data
    sp_prefix = cfg.vocab_dir / "bpe"
    train_sentencepiece(cfg.vocab_dir, sp_prefix, cfg.vocab_size)
    tok = BPETokenizer(sp_prefix.with_suffix(".model"))
    train_dl, val_dl = build_dataloaders(cfg, tok)

    # model, optim, scheduler
    model = GPTMoE(cfg, tok.vocab_size, tok.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = len(train_dl) * cfg.epochs
    scheduler = LambdaLR(opt, lr_lambda=lambda step: (
        step / cfg.warmup_steps if step < cfg.warmup_steps
        else max((total_steps - step) / (total_steps - cfg.warmup_steps), 0)
    ))
    ce = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    scaler = GradScaler()
    writer = SummaryWriter(cfg.log_dir)

    best_val = float("inf")
    step = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        loop = tqdm(train_dl, desc=f"Train Epoch {ep}", leave=False)
        for xb in loop:
            xb = xb.to(device)
            with autocast():
                logits, aux = model(xb,
                                    cfg.gate_noise_init,
                                    cfg.gate_noise_final,
                                    cfg.noise_anneal_steps)
                loss_main = ce(logits.reshape(-1, tok.vocab_size),
                               xb[:,1:].reshape(-1))
                loss = loss_main + aux

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            scaler.step(opt); scaler.update()
            opt.zero_grad()
            scheduler.step()

            # logging
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/main_loss", loss_main.item(), step)
            writer.add_scalar("train/aux_loss", aux.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            step += 1
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        model.eval()
        val_loss = 0.0; val_steps = 0
        with torch.no_grad():
            for xb in val_dl:
                xb = xb.to(device)
                logits, aux = model(xb,
                                    cfg.gate_noise_init,
                                    cfg.gate_noise_final,
                                    cfg.noise_anneal_steps)
                lm = ce(logits.reshape(-1, tok.vocab_size),
                        xb[:,1:].reshape(-1))
                val_loss += (lm + aux).item()
                val_steps += 1
        avg_val = val_loss / max(1, val_steps)
        writer.add_scalar("val/loss", avg_val, ep)
        print(f"[Epoch {ep}] Validation loss: {avg_val:.4f}")

        # checkpoint & early stopping
        ckpt_state = {
            "model": model.state_dict(),
            "opt":   opt.state_dict(),
            "sched": scheduler.state_dict(),
            "step":  step,
            "spm":   str(sp_prefix.with_suffix(".model")),
        }
        save_checkpoint(ckpt_state, cfg, ep)

        if avg_val < best_val:
            best_val = avg_val
        else:
            print("[!] Validation did not improve; consider early stopping.")
    
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    cfg = parse_args()
    train_and_validate(cfg)
