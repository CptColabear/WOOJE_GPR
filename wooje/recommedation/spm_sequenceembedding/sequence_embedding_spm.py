#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
from pymongo import MongoClient, UpdateOne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from prefixspan import PrefixSpan


# =========================
# 0. Config
# =========================

@dataclass
class CFG:
    # Mongo
    host: str = "10.255.68.40"
    port: int = 27017
    username: str = ""
    password: str = ""
    db_name: str = "ejoow"

    src_collection: str = "2. US_sequence_catlen_gt1_with_catid"

    # Tokens / vocab
    pad_id: int = 0
    n_items: int = 431
    mask_id: int = 432
    vocab_size: int = 433

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

    # SPM
    spm_ratio: float = 0.7
    min_pattern_len: int = 2
    max_patterns_per_user_tl: int = 1

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = CFG()


# =========================
# 1. Reproducibility
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# =========================
# 2. Mongo 연결
# =========================

def get_mongo_client(cfg: CFG) -> MongoClient:
    if cfg.username == "" and cfg.password == "":
        return MongoClient(cfg.host, cfg.port)

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

print("src count:", src_col.estimated_document_count())


# =========================
# 3. 샘플 로드
# =========================

def load_samples_from_mongo(src_col, limit: int = 0) -> List[Dict[str, Any]]:
    q = {"has_missing_category_id": False}
    proj = {
        "_id": 0,
        "user_id": 1,
        "TL": 1,
        "date": 1,
        "category_id_sequence": 1,
    }

    cur = src_col.find(q, proj)

    samples = []
    for doc in tqdm(cur, desc="Loading samples"):
        seq = doc.get("category_id_sequence", [])

        if seq is None:
            continue

        # category_id_sequence 길이 2 이상만 사용
        if len(seq) >= 2:
            doc["user_id"] = str(doc["user_id"])
            doc["TL"] = int(doc["TL"])
            doc["category_id_sequence"] = [int(x) for x in seq]
            samples.append(doc)

        if limit and len(samples) >= limit:
            break

    return samples


samples = load_samples_from_mongo(src_col, limit=0)
print("loaded samples:", len(samples))
print("sample:", samples[0])


# =========================
# 4. Train / Val split
# =========================

def train_val_split(samples: List[Dict[str, Any]], val_ratio=0.02):
    samples = samples.copy()
    random.shuffle(samples)
    n = len(samples)
    cut = int(n * (1 - val_ratio))
    return samples[:cut], samples[cut:]


train_samples, val_samples = train_val_split(samples, val_ratio=0.02)
print("train:", len(train_samples), "val:", len(val_samples))


# =========================
# 5. MLM Dataset
# =========================

def make_mlm_example(seq: List[int], cfg: CFG) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = seq[-cfg.max_len:]
    L = len(seq)

    input_ids = seq.copy()
    labels = [-100] * L

    n_mask = max(1, int(round(L * cfg.mask_ratio)))
    mask_positions = random.sample(range(L), k=min(n_mask, L))

    for pos in mask_positions:
        original = input_ids[pos]
        labels[pos] = original

        r = random.random()
        if r < 0.8:
            input_ids[pos] = cfg.mask_id
        elif r < 0.9:
            input_ids[pos] = random.randint(1, cfg.n_items)
        else:
            pass

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
    def __init__(self, rows: List[Dict[str, Any]], cfg: CFG):
        self.rows = rows
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        doc = self.rows[idx]
        seq = doc["category_id_sequence"]
        input_ids, attn, labels = make_mlm_example(seq, self.cfg)
        return input_ids, attn, labels


train_ds = SeqDataset(train_samples, cfg)
val_ds = SeqDataset(val_samples, cfg)

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)


# =========================
# 6. BERT4RecEncoder
# =========================

class BERT4RecEncoder(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        self.item_emb = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=cfg.pad_id,
        )

        self.pos_emb = nn.Embedding(
            cfg.max_len,
            cfg.d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, L = input_ids.shape

        pos = torch.arange(
            L,
            device=input_ids.device,
        ).unsqueeze(0).expand(B, L)

        x = self.item_emb(input_ids) + self.pos_emb(pos)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        key_padding_mask = attention_mask == 0

        h = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        logits = self.mlm_head(h)

        return logits, h


model = BERT4RecEncoder(cfg).to(cfg.device)
print(model)


# =========================
# 7. MLM 학습
# =========================

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

            logits, _ = model(input_ids, attn)

            loss = ce(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
            )

            total_loss += loss.item()
            total_tokens += (labels.reshape(-1) != -100).sum().item()

    return total_loss / max(1, total_tokens)


def train_mlm(model, train_loader, val_loader, cfg: CFG):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    ce = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = float("inf")
    best_path = "bert4rec_encoder_best_spm.pt"

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

            loss = ce(
                logits.reshape(-1, cfg.vocab_size),
                labels.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            opt.step()

            running += loss.item()
            steps += 1

            pbar.set_postfix(loss=running / steps)

        val_loss = evaluate_mlm(model, val_loader, cfg)
        print(f"[Epoch {epoch}] val_mlm_loss_per_masked_token = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print("  saved best checkpoint:", best_path)

    return best_path


best_path = train_mlm(model, train_loader, val_loader, cfg)

# best checkpoint 로드 후 임베딩 생성
model.load_state_dict(torch.load(best_path, map_location=cfg.device))
model.eval()


# =========================
# 8. Mean Pooling
# =========================

def mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    hidden: (B, L, D)
    attention_mask: (B, L), 1=valid, 0=PAD
    return: (B, D)
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)

    return summed / denom


# =========================
# 9. user × TL별 날짜 sequence 그룹화
# =========================

def build_user_tl_sequences(samples: List[Dict[str, Any]]):
    """
    return:
        user_tl_map[(user_id, TL)] = [
            {"date": date, "seq": [cat1, cat2, ...]},
            ...
        ]
    """
    user_tl_map = defaultdict(list)

    for doc in samples:
        user_id = str(doc["user_id"])
        TL = int(doc["TL"])
        date = str(doc["date"])
        seq = [int(x) for x in doc["category_id_sequence"]]

        if len(seq) < 2:
            continue

        user_tl_map[(user_id, TL)].append({
            "date": date,
            "seq": seq,
        })

    # 날짜순 정렬
    for key in user_tl_map:
        user_tl_map[key].sort(key=lambda x: x["date"])

    return user_tl_map


user_tl_map = build_user_tl_sequences(samples)
print("num user×TL groups:", len(user_tl_map))


# =========================
# 10. PrefixSpan으로 Sequential Pattern 추출
# =========================

def calc_min_support(n_sequences: int, ratio: float = 0.7) -> int:
    """
    요구사항:
    Min_sup = 전체 user sequence 갯수의 70%
    소수점 첫째자리에서 반올림
    """
    return max(1, int(round(n_sequences * ratio)))


def extract_prefixspan_patterns(
    seqs: List[List[int]],
    min_support: int,
    min_pattern_len: int = 2,
):
    """
    PrefixSpan 결과:
        [(support, pattern), ...]
    """
    ps = PrefixSpan(seqs)
    patterns = ps.frequent(min_support)

    # 길이가 너무 짧은 패턴 제거
    patterns = [
        (support, pattern)
        for support, pattern in patterns
        if len(pattern) >= min_pattern_len
    ]

    return patterns


def select_best_pattern(patterns):
    """
    대표 패턴 선택 기준:
    1순위: support가 큰 패턴
    2순위: pattern 길이가 긴 패턴
    3순위: pattern 사전순
    """
    if not patterns:
        return None

    patterns = sorted(
        patterns,
        key=lambda x: (x[0], len(x[1]), x[1]),
        reverse=True,
    )

    return patterns[0]


spm_rows = []

for (user_id, TL), rows in tqdm(user_tl_map.items(), desc="Extracting PrefixSpan patterns"):
    seqs = [r["seq"][-cfg.max_len:] for r in rows]
    n_sequences = len(seqs)

    if n_sequences <= 0:
        continue

    min_support = calc_min_support(n_sequences, cfg.spm_ratio)

    patterns = extract_prefixspan_patterns(
        seqs=seqs,
        min_support=min_support,
        min_pattern_len=cfg.min_pattern_len,
    )

    best = select_best_pattern(patterns)

    if best is None:
        continue

    support, pattern = best

    # BERT 입력 max_len에 맞춤
    pattern = pattern[-cfg.max_len:]

    spm_rows.append({
        "user_id": user_id,
        "TL": int(TL),
        "pattern": [int(x) for x in pattern],
        "pattern_len": len(pattern),
        "support": int(support),
        "n_sequences": int(n_sequences),
        "min_support": int(min_support),
        "support_ratio": float(support / n_sequences),
    })

print("num extracted user×TL patterns:", len(spm_rows))
print("sample spm row:", spm_rows[0] if spm_rows else None)


# =========================
# 11. SPM Pattern Dataset
# =========================

class SPMPatternDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], cfg: CFG):
        self.rows = rows
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        seq = row["pattern"][-self.cfg.max_len:]
        L = len(seq)

        input_ids = seq + [self.cfg.pad_id] * (self.cfg.max_len - L)
        attn = [1] * L + [0] * (self.cfg.max_len - L)

        meta = {
            "user_id": row["user_id"],
            "TL": int(row["TL"]),
            "pattern": row["pattern"],
            "pattern_len": int(row["pattern_len"]),
            "support": int(row["support"]),
            "n_sequences": int(row["n_sequences"]),
            "min_support": int(row["min_support"]),
            "support_ratio": float(row["support_ratio"]),
        }

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            meta,
        )


def spm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attn = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]

    return input_ids, attn, metas


spm_ds = SPMPatternDataset(spm_rows, cfg)

spm_loader = DataLoader(
    spm_ds,
    batch_size=2048,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    collate_fn=spm_collate_fn,
)


# =========================
# 12. 저장 컬렉션 이름 결정
# =========================

# user×TL마다 min_support가 다르므로 spm70을 추천
out_collection_name = f"user_tl_day_embeddings_v2_spm{int(cfg.spm_ratio * 100)}"
out_col = db[out_collection_name]

print("output collection:", out_collection_name)


# =========================
# 13. SPM Pattern Embedding 저장
# =========================

@torch.no_grad()
def write_spm_embeddings_to_mongo(
    model,
    loader,
    cfg: CFG,
    out_col,
    upsert=True,
    chunk_size=5000,
):
    model.eval()

    # user_id, TL 기준으로 하나의 대표 SPM embedding 저장
    out_col.create_index(
        [("user_id", 1), ("TL", 1)],
        unique=True,
    )

    ops = []
    total_ops = 0

    for input_ids, attn, metas in tqdm(loader, desc="Embedding SPM patterns & writing"):
        input_ids = input_ids.to(cfg.device, non_blocking=True)
        attn = attn.to(cfg.device, non_blocking=True)

        _, h = model(input_ids, attn)
        z = mean_pooling(h, attn)
        z = F.normalize(z, p=2, dim=-1)

        z_np = z.detach().cpu().numpy().astype(np.float32)

        for i, meta in enumerate(metas):
            user_id = str(meta["user_id"])
            TL = int(meta["TL"])

            _id = f"{user_id}|{TL}"

            doc = {
                "_id": _id,
                "user_id": user_id,
                "TL": TL,
                "dim": int(cfg.d_model),

                # 최종 임베딩
                "embedding": z_np[i].tolist(),

                # SPM 정보
                "spm_method": "PrefixSpan",
                "spm_ratio": float(cfg.spm_ratio),
                "min_support": int(meta["min_support"]),
                "support": int(meta["support"]),
                "support_ratio": float(meta["support_ratio"]),
                "n_sequences": int(meta["n_sequences"]),

                # 추출된 sequential pattern
                "pattern": [int(x) for x in meta["pattern"]],
                "pattern_len": int(meta["pattern_len"]),

                # 모델 정보
                "model": {
                    "name": "BERT4Rec-Encoder-SPM-MeanPool",
                    "max_len": cfg.max_len,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "checkpoint": "bert4rec_encoder_best_spm.pt",
                },
            }

            ops.append(
                UpdateOne(
                    {"_id": _id},
                    {"$set": doc},
                    upsert=upsert,
                )
            )

        if len(ops) >= chunk_size:
            out_col.bulk_write(ops, ordered=False)
            total_ops += len(ops)
            ops = []

    if ops:
        out_col.bulk_write(ops, ordered=False)
        total_ops += len(ops)

    print("Done. bulk ops written:", total_ops)


write_spm_embeddings_to_mongo(
    model=model,
    loader=spm_loader,
    cfg=cfg,
    out_col=out_col,
    upsert=True,
    chunk_size=5000,
)

print("out collection:", out_collection_name)
print("out count:", out_col.estimated_document_count())
print(
    out_col.find_one(
        {},
        {
            "_id": 1,
            "user_id": 1,
            "TL": 1,
            "pattern": 1,
            "pattern_len": 1,
            "support": 1,
            "min_support": 1,
            "n_sequences": 1,
            "support_ratio": 1,
            "dim": 1,
            "embedding": {"$slice": 5},
        },
    )
)