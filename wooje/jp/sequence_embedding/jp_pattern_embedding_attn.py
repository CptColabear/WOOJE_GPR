"""Stage 7d/7f: BERT4Rec MLM day-embeddings + attention pattern pooling for Tokyo.

Direct port of pattern_embedding_attn.py (which itself is sequence_embedding.py's
full BERT4Rec MLM training/day-embedding pipeline, plus an appended attention
pattern-pooling stage reusing the same trained model's item-embedding table --
confirmed by reading the original end to end, it is one self-contained script,
not two). Model architecture, masking scheme, and attention pooling are kept
identical (country-agnostic methodology); only Mongo target + vocab size differ:
  - db_name: "ejoow" -> "ejoow2"
  - src_collection: "2. US_sequence_catlen_gt1_with_catid" -> "2. JP_sequence_catlen_gt1_with_catid"
  - dist_col: "3. US_user_TL_category_dist" -> "3. JP_user_TL_category_dist"
  - n_items/mask_id/vocab_size: read from "0. JP_sequence_vocab_meta" instead of
    the hardcoded US values (431/432/433) -- JP's category vocabulary size differs.
  - out_collection / pattern_col names are unchanged (country-agnostic, separate db).
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymongo import MongoClient, UpdateOne
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent


@dataclass
class CFG:
    host: str = "10.255.68.40"
    port: int = 27017
    db_name: str = "ejoow2"
    src_collection: str = "2. JP_sequence_catlen_gt1_with_catid"
    out_collection: str = "user_tl_day_embeddings_v1"
    dist_collection: str = "3. JP_user_TL_category_dist"
    pattern_collection: str = "user_tl_pattern_embeddings_attn_v1"
    checkpoint_path: str = str(THIS_DIR / "bert4rec_encoder_best_jp.pt")

    pad_id: int = 0
    n_items: int = 0    # filled from 0. JP_sequence_vocab_meta
    mask_id: int = 0
    vocab_size: int = 0

    max_len: int = 16
    mask_ratio: float = 0.5

    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.1

    seed: int = 42
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    num_workers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mlm_example(seq: List[int], cfg: CFG):
    seq = seq[-cfg.max_len :]
    L = len(seq)
    input_ids = seq.copy()
    labels = [-100] * L

    n_mask = max(1, int(round(L * cfg.mask_ratio)))
    mask_positions = random.sample(range(L), k=min(n_mask, L))
    for pos in mask_positions:
        labels[pos] = input_ids[pos]
        r = random.random()
        if r < 0.8:
            input_ids[pos] = cfg.mask_id
        elif r < 0.9:
            input_ids[pos] = random.randint(1, cfg.n_items)
        # else keep original

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
        seq = self.rows[idx]["category_id_sequence"]
        return make_mlm_example(seq, self.cfg)


class EmbedDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], cfg: CFG):
        self.rows = rows
        self.cfg = cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        doc = self.rows[idx]
        seq_full = doc["category_id_sequence"]
        seq = seq_full[-self.cfg.max_len :]
        L = len(seq)
        input_ids = seq + [self.cfg.pad_id] * (self.cfg.max_len - L)
        attn = [1] * L + [0] * (self.cfg.max_len - L)
        meta = (doc["user_id"], int(doc["TL"]), doc["date"], int(len(seq_full)))
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long), meta


def embed_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attn = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return input_ids, attn, metas


class BERT4RecEncoder(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(input_ids) + self.pos_emb(pos)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        key_padding_mask = attention_mask == 0
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.mlm_head(h), h


def evaluate_mlm(model, loader, cfg: CFG):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    with torch.no_grad():
        for input_ids, attn, labels in loader:
            input_ids, attn, labels = (t.to(cfg.device, non_blocking=True) for t in (input_ids, attn, labels))
            logits, _ = model(input_ids, attn)
            loss = ce(logits.view(-1, cfg.vocab_size), labels.view(-1))
            total_loss += loss.item()
            total_tokens += (labels.view(-1) != -100).sum().item()
    return total_loss / max(1, total_tokens)


def train_mlm(model, train_loader, val_loader, cfg: CFG):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        running, steps = 0.0, 0
        for input_ids, attn, labels in pbar:
            input_ids, attn, labels = (t.to(cfg.device, non_blocking=True) for t in (input_ids, attn, labels))
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
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print("  saved best checkpoint")


def mean_pooling(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


@torch.no_grad()
def write_day_embeddings_to_mongo(model, loader, cfg: CFG, out_col, chunk_size=5000):
    model.eval()
    out_col.create_index([("user_id", 1), ("TL", 1), ("date", 1)], unique=True)
    ops, total_ops = [], 0
    for input_ids, attn, metas in tqdm(loader, desc="Embedding & writing"):
        input_ids = input_ids.to(cfg.device, non_blocking=True)
        attn = attn.to(cfg.device, non_blocking=True)
        _, h = model(input_ids, attn)
        z = F.normalize(mean_pooling(h, attn), p=2, dim=-1)
        z_np = z.detach().cpu().numpy().astype(np.float32)
        for i, meta in enumerate(metas):
            user_id, TL, date, seq_len = str(meta[0]), int(meta[1]), str(meta[2]), int(meta[3])
            _id = f"{user_id}|{TL}|{date}"
            doc = {
                "_id": _id, "user_id": user_id, "TL": TL, "date": date, "seq_len": seq_len,
                "dim": int(cfg.d_model), "embedding": z_np[i].tolist(),
                "model": {"name": "BERT4Rec-Encoder-MeanPool", "max_len": cfg.max_len,
                          "d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads},
            }
            ops.append(UpdateOne({"_id": _id}, {"$set": doc}, upsert=True))
        if len(ops) >= chunk_size:
            out_col.bulk_write(ops, ordered=False)
            total_ops += len(ops)
            ops = []
    if ops:
        out_col.bulk_write(ops, ordered=False)
        total_ops += len(ops)
    print("Done. bulk ops written:", total_ops)


def build_query_vector(user_id, TL, E, user_tl_to_dist):
    dist = user_tl_to_dist.get((user_id, TL))
    if not dist:
        return None
    cats = [c for c, _ in dist]
    cnts = np.array([cnt for _, cnt in dist], dtype=np.float32)
    s = float(cnts.sum())
    if s <= 0:
        return None
    p = cnts / s
    emb = E[cats].float()
    w = torch.from_numpy(p).unsqueeze(1)
    return F.normalize((emb * w).sum(dim=0), p=2, dim=0)


def attention_pool(q: torch.Tensor, V: torch.Tensor):
    d = q.shape[0]
    scores = (V @ q) / np.sqrt(d)
    weights = torch.softmax(scores, dim=0)
    pattern = (weights.unsqueeze(1) * V).sum(dim=0)
    return F.normalize(pattern, p=2, dim=0), weights


def main():
    client = MongoClient("10.255.68.40", 27017, serverSelectionTimeoutMS=5000)
    db = client["ejoow2"]
    vocab_meta = db["0. JP_sequence_vocab_meta"].find_one({"_id": "sequence_vocab"})
    if vocab_meta is None:
        raise RuntimeError("0. JP_sequence_vocab_meta missing -- run jp_build_sequences.py first.")

    cfg = CFG(n_items=vocab_meta["n_items"], mask_id=vocab_meta["mask_id"], vocab_size=vocab_meta["vocab_size"])
    print(cfg)
    set_seed(cfg.seed)

    src_col = db[cfg.src_collection]
    out_col = db[cfg.out_collection]
    print("src count:", src_col.estimated_document_count())

    def load_samples(src_col):
        q = {"has_missing_category_id": False}
        proj = {"_id": 0, "user_id": 1, "TL": 1, "date": 1, "category_id_sequence": 1}
        return [doc for doc in tqdm(src_col.find(q, proj), desc="Loading samples")
                if len(doc.get("category_id_sequence", [])) >= 2]

    samples = load_samples(src_col)
    print("n_samples:", len(samples))

    shuffled = samples.copy()
    random.shuffle(shuffled)
    cut = int(len(shuffled) * 0.98)
    train_samples, val_samples = shuffled[:cut], shuffled[cut:]
    print("train/val:", len(train_samples), len(val_samples))

    train_loader = DataLoader(SeqDataset(train_samples, cfg), batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(SeqDataset(val_samples, cfg), batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = BERT4RecEncoder(cfg).to(cfg.device)
    train_mlm(model, train_loader, val_loader, cfg)

    embed_loader = DataLoader(EmbedDataset(samples, cfg), batch_size=2048, shuffle=False,
                               num_workers=cfg.num_workers, pin_memory=True, collate_fn=embed_collate_fn)
    write_day_embeddings_to_mongo(model, embed_loader, cfg, out_col)
    print("out count:", out_col.estimated_document_count())

    print("\n--- pattern embedding (attention pooling) ---")
    model.eval()
    E = model.item_emb.weight.detach().cpu()
    print("item_emb shape:", E.shape)

    dist_col = db[cfg.dist_collection]
    pattern_col = db[cfg.pattern_collection]
    pattern_col.create_index([("user_id", 1), ("TL", 1)], unique=True)

    from collections import defaultdict
    user_tl_to_dist = defaultdict(list)
    for d in tqdm(dist_col.find({}, {"_id": 0, "user_id": 1, "TL": 1, "category_id": 1, "category_count": 1}),
                  desc="Loading dist"):
        user_tl_to_dist[(d["user_id"], int(d["TL"]))].append((int(d["category_id"]), int(d["category_count"])))
    print("num user x TL with dist:", len(user_tl_to_dist))

    def fetch_day_embeddings(user_id, TL):
        cur = out_col.find({"user_id": user_id, "TL": TL}, {"_id": 0, "date": 1, "seq_len": 1, "embedding": 1})
        vecs = list(cur)
        if not vecs:
            return None
        return torch.tensor(np.array([v["embedding"] for v in vecs], dtype=np.float32))

    ops, written = [], 0
    for user_id, TL in tqdm(list(user_tl_to_dist.keys()), desc="Building patterns"):
        q = build_query_vector(user_id, TL, E, user_tl_to_dist)
        if q is None:
            continue
        V = fetch_day_embeddings(user_id, TL)
        if V is None:
            continue
        pattern, _ = attention_pool(q, V)
        doc = {
            "_id": f"{user_id}|{TL}", "user_id": user_id, "TL": TL, "dim": int(cfg.d_model),
            "pattern_embedding": pattern.detach().cpu().numpy().astype(np.float32).tolist(),
            "n_days": int(V.shape[0]),
            "query_type": "category_dist_weighted_item_embedding",
            "attn_type": "dot_softmax_scaled",
            "model_ref": {"d_model": cfg.d_model, "max_len": cfg.max_len, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads},
        }
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
        if len(ops) >= 5000:
            pattern_col.bulk_write(ops, ordered=False)
            written += len(ops)
            ops = []
    if ops:
        pattern_col.bulk_write(ops, ordered=False)
        written += len(ops)
    print("done. upsert ops:", written)
    print("pattern_col count:", pattern_col.estimated_document_count())


if __name__ == "__main__":
    main()
