#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from pymongo import MongoClient, UpdateOne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# In[10]:


@dataclass
class CFG:
    # Mongo
    host: str = "10.255.68.40"
    port: int = 27017
    username: str = ""  # empty means no auth
    password: str = ""
    db_name: str = "ejoow"
    src_collection: str = "2. US_sequence_catlen_gt1_with_catid"  # note trailing space
    out_collection: str = "user_tl_day_embeddings_v1"

    # Tokens / vocab
    pad_id: int = 0
    n_items: int = 431         # category_id in [1..431]
    mask_id: int = 432         # special token
    vocab_size: int = 433      # 0..432

    # Sequence / MLM
    max_len: int = 16
    mask_ratio: float = 0.5

    # Model
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.1

    # Train
    seed: int = 42
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    num_workers: int = 4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
cfg


# 1. (셀) Reproducibility & Mongo 연결

# In[11]:


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)


# In[12]:


def get_mongo_client(cfg: CFG) -> MongoClient:
    # no auth case
    if cfg.username == "" and cfg.password == "":
        return MongoClient(cfg.host, cfg.port)
    # auth case (if you later add username/pw)
    return MongoClient(
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        password=cfg.password,
        authSource=cfg.db_name,
    )

client = get_mongo_client(cfg)
db = client[cfg.db_name]
src_col = db[cfg.src_collection]
out_col = db[cfg.out_collection]

print("src count:", src_col.estimated_document_count())


# 2. (셀) 학습 샘플 로드 (메모리 로드 버전)

# In[13]:


def load_samples_from_mongo(src_col, limit: int = 0) -> List[Dict[str, Any]]:
    q = {"has_missing_category_id": False}
    proj = {"_id": 0, "user_id": 1, "TL": 1, "date": 1, "category_id_sequence": 1}
    cur = src_col.find(q, proj)

    samples = []
    for doc in tqdm(cur, desc="Loading samples"):
        seq = doc.get("category_id_sequence", [])
        if len(seq) >= 2:
            samples.append(doc)
        if limit and len(samples) >= limit:
            break
    return samples

samples = load_samples_from_mongo(src_col, limit=0)  # limit=0 => all
len(samples), samples[0]


# 3. (셀) Train/Val Split

# In[14]:


def train_val_split(samples: List[Dict[str, Any]], val_ratio=0.02):
    random.shuffle(samples)
    n = len(samples)
    cut = int(n * (1 - val_ratio))
    return samples[:cut], samples[cut:]

train_samples, val_samples = train_val_split(samples, val_ratio=0.02)
len(train_samples), len(val_samples)


# 4. (셀) MLM 마스킹 함수 & Dataset

# In[15]:


def make_mlm_example(seq: List[int], cfg: CFG) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # truncate to last max_len (recent)
    seq = seq[-cfg.max_len:]
    L = len(seq)

    input_ids = seq.copy()
    labels = [-100] * L

    # choose mask positions
    n_mask = max(1, int(round(L * cfg.mask_ratio)))
    mask_positions = random.sample(range(L), k=min(n_mask, L))

    for pos in mask_positions:
        original = input_ids[pos]
        labels[pos] = original

        r = random.random()
        if r < 0.8:
            input_ids[pos] = cfg.mask_id
        elif r < 0.9:
            input_ids[pos] = random.randint(1, cfg.n_items)  # 1..431
        else:
            pass  # keep original

    # pad to max_len
    attn = [1] * L
    if L < cfg.max_len:
        pad_len = cfg.max_len - L
        input_ids += [cfg.pad_id] * pad_len
        labels += [-100] * pad_len
        attn += [0] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

class SeqDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], cfg: CFG, return_meta: bool = False):
        self.rows = rows
        self.cfg = cfg
        self.return_meta = return_meta

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        doc = self.rows[idx]
        seq = doc["category_id_sequence"]
        input_ids, attn, labels = make_mlm_example(seq, self.cfg)
        if not self.return_meta:
            return input_ids, attn, labels
        meta = {"user_id": doc["user_id"], "TL": doc["TL"], "date": doc["date"], "seq_len": len(seq)}
        return input_ids, attn, labels, meta

train_ds = SeqDataset(train_samples, cfg)
val_ds = SeqDataset(val_samples, cfg)


# In[16]:


train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)


# 5. (셀) BERT4Rec 인코더 모델 (PyTorch)

# In[17]:


class BERT4RecEncoder(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # MLM head: hidden -> vocab logits
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids: (B, L)
        attention_mask: (B, L), 1=valid, 0=pad
        """
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_ids) + self.pos_emb(pos)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        # Transformer key padding mask: True where PAD
        key_padding_mask = (attention_mask == 0)  # (B, L)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, D)

        logits = self.mlm_head(h)  # (B, L, V)
        return logits, h

model = BERT4RecEncoder(cfg).to(cfg.device)
model


# 6. (셀) 학습 루프 (MLM)

# In[18]:


def evaluate_mlm(model, loader, cfg: CFG):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

    with torch.no_grad():
        for input_ids, attn, labels in loader:
            input_ids = input_ids.to(cfg.device, non_blocking=True)
            attn = attn.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            logits, _ = model(input_ids, attn)  # (B, L, V)
            loss = ce(logits.view(-1, cfg.vocab_size), labels.view(-1))
            total_loss += loss.item()

            # count masked tokens
            total_tokens += (labels.view(-1) != -100).sum().item()

    return total_loss / max(1, total_tokens)

def train_mlm(model, train_loader, val_loader, cfg: CFG):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running = 0.0
        steps = 0

        for input_ids, attn, labels in pbar:
            input_ids = input_ids.to(cfg.device, non_blocking=True)
            attn = attn.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits, _ = model(input_ids, attn)
            loss = ce(logits.view(-1, cfg.vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += loss.item()
            steps += 1
            pbar.set_postfix(loss=running / steps)

        val_loss = evaluate_mlm(model, val_loader, cfg)
        print(f"[Epoch {epoch}] val_mlm_loss_per_masked_token = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "bert4rec_encoder_best.pt")
            print("  saved best checkpoint")

train_mlm(model, train_loader, val_loader, cfg)


# 7. (셀) Day 임베딩 추출용 DataLoader

# In[25]:


class EmbedDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], cfg: CFG):
        self.rows = rows
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        doc = self.rows[idx]
        seq_full = doc["category_id_sequence"]
        seq = seq_full[-self.cfg.max_len:]
        L = len(seq)

        input_ids = seq + [self.cfg.pad_id] * (self.cfg.max_len - L)
        attn = [1] * L + [0] * (self.cfg.max_len - L)

        # ✅ meta를 dict 대신 튜플로
        meta = (doc["user_id"], int(doc["TL"]), doc["date"], int(len(seq_full)))
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            meta
        )

embed_ds = EmbedDataset(samples, cfg)
# embed_loader = DataLoader(embed_ds, batch_size=2048, shuffle=False,
#                           num_workers=cfg.num_workers, pin_memory=True)


# In[26]:


def embed_collate_fn(batch):
    # batch: list of (input_ids, attn, meta)
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attn = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]   # ✅ meta를 그대로 리스트로 유지
    return input_ids, attn, metas

embed_loader = DataLoader(
    embed_ds,
    batch_size=2048,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=embed_collate_fn
)


# In[27]:


input_ids, attn, metas = next(iter(embed_loader))
print("metas type:", type(metas))
print("metas len:", len(metas))
print("metas[0]:", metas[0])
print("type(metas[0]):", type(metas[0]))


# 8. (셀) Mean Pooling 임베딩 + MongoDB 저장

# In[ ]:


import torch

def mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden: (B, L, D)
    attention_mask: (B, L)  with 1 for valid tokens, 0 for PAD
    return: (B, D)
    """
    # (B, L, 1)
    mask = attention_mask.unsqueeze(-1).float()

    # sum of valid hidden states
    summed = (hidden * mask).sum(dim=1)   # (B, D)

    # number of valid tokens per sample
    denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

    return summed / denom


@torch.no_grad()
def write_day_embeddings_to_mongo(model, loader, cfg: CFG, out_col, upsert=True, chunk_size=5000):
    model.eval()

    # 유니크 키(중복 방지)
    out_col.create_index([("user_id", 1), ("TL", 1), ("date", 1)], unique=True)

    ops = []
    total_ops = 0

    for input_ids, attn, metas in tqdm(loader, desc="Embedding & writing"):
        input_ids = input_ids.to(cfg.device, non_blocking=True)
        attn = attn.to(cfg.device, non_blocking=True)

        # forward
        _, h = model(input_ids, attn)      # (B, L, D)
        z = mean_pooling(h, attn)          # (B, D)
        z = F.normalize(z, p=2, dim=-1)    # L2 normalize

        z_np = z.detach().cpu().numpy().astype(np.float32)

        for i, meta in enumerate(metas):
            # meta == ['user_id', TL, 'date', seq_len]
            user_id = str(meta[0])
            TL = int(meta[1])
            date = str(meta[2])
            seq_len = int(meta[3])

            _id = f"{user_id}|{TL}|{date}"

            doc = {
                "_id": _id,
                "user_id": user_id,
                "TL": TL,
                "date": date,
                "seq_len": seq_len,
                "dim": int(cfg.d_model),
                "embedding": z_np[i].tolist(),
                "model": {
                    "name": "BERT4Rec-Encoder-MeanPool",
                    "max_len": cfg.max_len,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                }
            }
            ops.append(UpdateOne({"_id": _id}, {"$set": doc}, upsert=upsert))

        if len(ops) >= chunk_size:
            out_col.bulk_write(ops, ordered=False)
            total_ops += len(ops)
            ops = []

    if ops:
        out_col.bulk_write(ops, ordered=False)
        total_ops += len(ops)

    print("Done. bulk ops written:", total_ops)


# In[29]:


write_day_embeddings_to_mongo(model, embed_loader, cfg, out_col, upsert=True, chunk_size=5000)


# In[30]:


print("out count:", out_col.estimated_document_count())
print(out_col.find_one({}, {"_id": 1, "user_id": 1, "TL": 1, "date": 1, "seq_len": 1, "dim": 1, "embedding": {"$slice": 5}}))


