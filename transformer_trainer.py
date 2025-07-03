#!/usr/bin/env python3
# gpt_text_trainer.py — tiny single-GPU GPT trainer for plain text
# 2025-07-03 • MIT-0
"""
Train an **autoregressive Transformer (GPT)** on a folder of .txt files.

Major refactor
--------------
* Replaced the RAM-hungry list-of-windows dataset with a **memory-mapped
  token buffer**.  Windows are sliced on-the-fly, so startup time and
  memory use are independent of corpus size.
* No other behaviour changed; the CLI is identical.

Example
-------
python gpt_text_trainer.py \
    --text_corpus  data/text \
    --vocab_src    data/text \
    --ckpt         gpt_text.pt
"""
from __future__ import annotations
import os, argparse
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import sentencepiece as spm

# ─── hyper-parameters ───────────────────────────────────────────
SEQ_LEN     = 64
BATCH_SIZE  = 8
EPOCHS      = 3
EMB_DIM     = 512
N_HEADS     = 8
DEPTH       = 12
MLP_RATIO   = 4
DROPOUT     = 0.1
LR          = 3e-4
CLIP_NORM   = 1.0
VOCAB_SIZE  = 32_000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PAD_PIECE   = "<pad>"
# ────────────────────────────────────────────────────────────────


# ─── SentencePiece helpers ─────────────────────────────────────
def train_spm(corpus_dir: str, model_prefix: str, vocab_size: int = VOCAB_SIZE):
    if Path(model_prefix + ".model").exists():
        return
    tmp = Path(model_prefix + "_tmp.txt")
    with tmp.open("w", encoding="utf8") as out:
        for p in Path(corpus_dir).rglob("*.txt"):
            out.write(p.read_text(encoding="utf8", errors="ignore") + "\n")
    spm.SentencePieceTrainer.Train(
        input=str(tmp),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_piece=PAD_PIECE,
        pad_id=0,
        unk_piece="<unk>",
    )
    tmp.unlink()


class BPETokenizer:
    def __init__(self, model_file: str):
        self.sp  = spm.SentencePieceProcessor(model_file=model_file)
        self.pad = self.sp.pad_id()
        self.vsz = self.sp.get_piece_size()

    def encode_ids(self, txt: str) -> list[int]:
        return self.sp.encode(txt, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i for i in ids if i != self.pad])


# ─── Memory-mapped dataset ─────────────────────────────────────
class TokenBufferDataset(Dataset):
    """
    Encodes the entire corpus once into a flat int32 buffer saved as
    <vocab_src>/corpus_tokens.npy.  The array is loaded with mmap, so
    RAM usage stays tiny.  Each __getitem__ returns a sliding window
    (SEQ_LEN+1) starting at 'idx'.
    """
    def __init__(self, corpus_dir: str, tok: BPETokenizer):
        buf_path = Path(corpus_dir) / "corpus_tokens.npy"
        if not buf_path.exists():
            self._build_buffer(corpus_dir, tok, buf_path)
        self.buf = np.load(buf_path, mmap_mode="r")          # read-only memmap
        self.length = len(self.buf) - (SEQ_LEN + 1)          # #windows

    def _build_buffer(self, root: str, tok: BPETokenizer, out_file: Path):
        ids: list[int] = []
        for p in Path(root).rglob("*.txt"):
            ids.extend(tok.encode_ids(
                Path(p).read_text(encoding="utf8", errors="ignore")))
        arr = np.asarray(ids, dtype=np.int32)
        np.save(out_file, arr)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        slc = self.buf[idx : idx + SEQ_LEN + 1]
        return torch.as_tensor(slc, dtype=torch.long)


def loader(ds: Dataset) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )


# ─── Causal Transformer backbone ───────────────────────────────
class CausalEncoder(nn.Module):
    def __init__(self, emb_dim: int, depth: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        return self.enc(x, mask)


# ─── GPT model ─────────────────────────────────────────────────
class GPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMB_DIM)
        self.pos_emb = nn.Parameter(torch.randn(SEQ_LEN + 1, EMB_DIM))
        self.drop    = nn.Dropout(DROPOUT)
        self.backbone= CausalEncoder(EMB_DIM, DEPTH, N_HEADS, MLP_RATIO)
        self.head    = nn.Linear(EMB_DIM, vocab_size, bias=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        tok = inp[:, :-1]
        x   = self.drop(self.tok_emb(tok) + self.pos_emb[: tok.size(1)])
        h   = self.backbone(x)
        return self.head(h)


# ─── Training loop ─────────────────────────────────────────────
def train(args):
    sp_prefix = os.path.join(args.vocab_src, "bpe")
    train_spm(args.vocab_src, sp_prefix)
    tok = BPETokenizer(sp_prefix + ".model")

    dl = loader(TokenBufferDataset(args.text_corpus, tok))

    model  = GPT(tok.vsz).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=LR)
    ce     = nn.CrossEntropyLoss(ignore_index=tok.pad)
    scaler = GradScaler()

    print(f"[i] parameters: {sum(p.numel() for p in model.parameters()):,}")

    for ep in range(1, EPOCHS + 1):
        bar = tqdm(dl, desc=f"Epoch {ep}/{EPOCHS}", ncols=0, unit="batch")
        for xb in bar:
            xb = xb.to(DEVICE)
            with autocast():
                logits = model(xb)
                loss   = ce(
                    logits.reshape(-1, tok.vsz),
                    xb[:, 1:].reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save({"state_dict": model.state_dict(),
                    "spm_model": sp_prefix + ".model"},
                   args.ckpt)
        print(f"[✓] epoch {ep} saved to {args.ckpt}")


# ─── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("tiny GPT text trainer (BPE version, streaming)")
    p.add_argument("--text_corpus", required=True,
                   help="Folder with .txt files for training examples")
    p.add_argument("--vocab_src",   required=True,
                   help="Folder (same or superset) to build the BPE vocab from")
    p.add_argument("--ckpt",        default="gpt_text.pt")
    args = p.parse_args()
    train(args)
