"""Stage 11: POI autoencoder embedding for JP -> "poi_embeddings_mlp_v1_a".

Direct port of poiembeddingA.ipynb (confirmed via full read of the original)
converted to a plain script for reliable unattended execution. Only
db_name/src_collection/country change; model architecture, loss weights,
and training schedule are unchanged. num_categories/num_clusters are derived
dynamically from the JP data at training time (same as the original), so
they correctly adapt to JP's own category/cluster cardinality.
"""
import copy
import random
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymongo import MongoClient, UpdateOne
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass
class CFG:
    host: str = "10.255.68.40"
    port: int = 27017
    db_name: str = "ejoow2"
    src_collection: str = "2. JP_pois_distribution"
    out_collection: str = "poi_embeddings_mlp_v1_a"
    meta_collection: str = "poi_embeddings_mlp_v1_a_meta"
    country: str = "JP"
    max_docs: int = 0

    seed: int = 42
    val_ratio: float = 0.10

    poi_emb_dim: int = 16
    category_emb_dim: int = 8
    cluster_emb_dim: int = 4
    numeric_dim: int = 7
    hidden1: int = 64
    hidden2: int = 128
    latent_dim: int = 64
    dropout: float = 0.10

    batch_size: int = 2048
    embed_batch_size: int = 4096
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 6

    num_workers: int = 0
    use_amp: bool = torch.cuda.is_available()
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    loss_w_category: float = 1.0
    loss_w_cluster: float = 1.0
    loss_w_latlon: float = 1.0
    loss_w_total: float = 1.0
    loss_w_tl: float = 1.0

    bulk_write_size: int = 1000
    unknown_category_value: int = -1
    unknown_cluster_value: int = -1
    model_name: str = "poi_autoencoder_mlp_v1_a"
    save_local_checkpoint: bool = False
    checkpoint_path: str = "/tmp/poi_autoencoder_mlp_v1_a_jp.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class POIAutoEncoder(nn.Module):
    def __init__(self, num_pois, num_categories, num_clusters, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.poi_embedding = nn.Embedding(num_pois, cfg.poi_emb_dim)
        self.category_embedding = nn.Embedding(num_categories, cfg.category_emb_dim)
        self.cluster_embedding = nn.Embedding(num_clusters, cfg.cluster_emb_dim)
        self.input_dim = cfg.poi_emb_dim + cfg.category_emb_dim + cfg.cluster_emb_dim + cfg.numeric_dim
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, cfg.hidden1), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, cfg.latent_dim),
        )
        self.decoder_backbone = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden2), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, cfg.hidden1), nn.ReLU(),
        )
        self.category_head = nn.Linear(cfg.hidden1, num_categories)
        self.cluster_head = nn.Linear(cfg.hidden1, num_clusters)
        self.latlon_head = nn.Linear(cfg.hidden1, 2)
        self.total_head = nn.Linear(cfg.hidden1, 1)
        self.tl_head = nn.Linear(cfg.hidden1, 4)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.poi_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.category_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cluster_embedding.weight, mean=0.0, std=0.02)

    def encode(self, poi_idx, category_idx, cluster_idx, numeric):
        x = torch.cat([self.poi_embedding(poi_idx), self.category_embedding(category_idx),
                       self.cluster_embedding(cluster_idx), numeric], dim=-1)
        return self.encoder(self.input_norm(x))

    def forward(self, poi_idx, category_idx, cluster_idx, numeric):
        z = self.encode(poi_idx, category_idx, cluster_idx, numeric)
        h = self.decoder_backbone(z)
        return {
            "z": z, "category_logits": self.category_head(h), "cluster_logits": self.cluster_head(h),
            "latlon_pred": self.latlon_head(h), "total_pred": self.total_head(h),
            "tl_pred": torch.softmax(self.tl_head(h), dim=-1),
        }


def move_batch_to_device(batch, device):
    return [t.to(device, non_blocking=device.startswith("cuda")) for t in batch]


def compute_losses(cfg, outputs, category_target, cluster_target, latlon_target, total_target, tl_target):
    category_loss = F.cross_entropy(outputs["category_logits"], category_target)
    cluster_loss = F.cross_entropy(outputs["cluster_logits"], cluster_target)
    latlon_loss = F.mse_loss(outputs["latlon_pred"], latlon_target)
    total_loss = F.mse_loss(outputs["total_pred"], total_target)
    valid_tl_mask = tl_target.sum(dim=1) > 0
    tl_loss = F.mse_loss(outputs["tl_pred"][valid_tl_mask], tl_target[valid_tl_mask]) if valid_tl_mask.any() else outputs["tl_pred"].sum() * 0.0
    loss = (cfg.loss_w_category * category_loss + cfg.loss_w_cluster * cluster_loss +
            cfg.loss_w_latlon * latlon_loss + cfg.loss_w_total * total_loss + cfg.loss_w_tl * tl_loss)
    return loss, {"loss": float(loss.detach().item()), "category": float(category_loss.detach().item()),
                  "cluster": float(cluster_loss.detach().item()), "latlon": float(latlon_loss.detach().item()),
                  "total": float(total_loss.detach().item()), "tl": float(tl_loss.detach().item())}


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    client = MongoClient(cfg.host, cfg.port)
    db = client[cfg.db_name]
    src_col = db[cfg.src_collection]
    out_col = db[cfg.out_collection]
    meta_col = db[cfg.meta_collection]
    print(cfg)
    print("source count:", src_col.estimated_document_count())

    amp_device_type = "cuda" if cfg.device.startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if amp_device_type == "cuda" else torch.bfloat16
    autocast_enabled = cfg.use_amp and amp_device_type == "cuda"

    projection = {"_id": 0, "venue_id": 1, "latitude": 1, "longitude": 1, "category_id": 1, "category": 1,
                  "country": 1, "loc_cluster_id_k5": 1, "visit1": 1, "visit2": 1, "visit3": 1, "visit4": 1}
    rows = []
    for doc in tqdm(src_col.find({"country": cfg.country}, projection, batch_size=5000), desc="Loading POIs"):
        rows.append({
            "venue_id": doc.get("venue_id"), "latitude": doc.get("latitude"), "longitude": doc.get("longitude"),
            "category_id": doc.get("category_id"), "category": doc.get("category"), "country": doc.get("country"),
            "loc_cluster_id_k5": doc.get("loc_cluster_id_k5"),
            "visit1": doc.get("visit1", 0), "visit2": doc.get("visit2", 0),
            "visit3": doc.get("visit3", 0), "visit4": doc.get("visit4", 0),
        })
        if cfg.max_docs and len(rows) >= cfg.max_docs:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No POI documents were loaded from MongoDB.")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["venue_id", "latitude", "longitude"]).copy()
    df["venue_id"] = df["venue_id"].astype(str)
    df["category"] = df["category"].fillna("UNKNOWN")
    df["country"] = df["country"].fillna(cfg.country)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["latitude"] = df["latitude"].astype(np.float32)
    df["longitude"] = df["longitude"].astype(np.float32)
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(cfg.unknown_category_value).astype(np.int64)
    df["loc_cluster_id_k5"] = pd.to_numeric(df["loc_cluster_id_k5"], errors="coerce").fillna(cfg.unknown_cluster_value).astype(np.int64)

    visit_cols = ["visit1", "visit2", "visit3", "visit4"]
    for col in visit_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0).astype(np.float32)

    df = df.sort_values("venue_id").reset_index(drop=True)
    visit_matrix = df[visit_cols].to_numpy(dtype=np.float32)
    total_visits = visit_matrix.sum(axis=1)
    safe_total = np.where(total_visits > 0, total_visits, 1.0)
    tl_distribution = visit_matrix / safe_total[:, None]
    tl_distribution[total_visits == 0] = 0.0
    df["total_visits"] = total_visits.astype(np.float32)
    df["log_total_visits"] = np.log1p(total_visits).astype(np.float32)
    for i in range(4):
        df[f"tl_dist_{i + 1}"] = tl_distribution[:, i].astype(np.float32)

    poi_to_idx = {v: i for i, v in enumerate(df["venue_id"].tolist())}
    category_values = sorted(df["category_id"].astype(int).unique().tolist())
    cluster_values = sorted(df["loc_cluster_id_k5"].astype(int).unique().tolist())
    category_to_idx = {raw: i for i, raw in enumerate(category_values)}
    cluster_to_idx = {raw: i for i, raw in enumerate(cluster_values)}
    df["poi_idx"] = df["venue_id"].map(poi_to_idx).astype(np.int64)
    df["category_idx"] = df["category_id"].map(category_to_idx).astype(np.int64)
    df["cluster_idx"] = df["loc_cluster_id_k5"].map(cluster_to_idx).astype(np.int64)

    row_indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(row_indices, test_size=cfg.val_ratio, random_state=cfg.seed, shuffle=True)

    norm_cols = ["latitude", "longitude", "log_total_visits"]
    train_means = df.loc[train_idx, norm_cols].mean()
    train_stds = df.loc[train_idx, norm_cols].std(ddof=0).replace(0, 1.0)
    for col in norm_cols:
        df[f"{col}_norm"] = ((df[col] - train_means[col]) / train_stds[col]).astype(np.float32)

    numeric_cols = ["latitude_norm", "longitude_norm", "log_total_visits_norm", "tl_dist_1", "tl_dist_2", "tl_dist_3", "tl_dist_4"]
    num_pois, num_categories, num_clusters = len(poi_to_idx), len(category_to_idx), len(cluster_to_idx)
    print(f"rows={len(df):,} num_pois={num_pois:,} num_categories={num_categories:,} num_clusters={num_clusters:,} "
          f"train={len(train_idx):,} val={len(val_idx):,}")

    def make_tensors(idx):
        return (
            torch.from_numpy(df["poi_idx"].to_numpy(dtype=np.int64)[idx]),
            torch.from_numpy(df["category_idx"].to_numpy(dtype=np.int64)[idx]),
            torch.from_numpy(df["cluster_idx"].to_numpy(dtype=np.int64)[idx]),
            torch.from_numpy(df[numeric_cols].to_numpy(dtype=np.float32)[idx]),
            torch.from_numpy(df[["latitude_norm", "longitude_norm"]].to_numpy(dtype=np.float32)[idx]),
            torch.from_numpy(df[["log_total_visits_norm"]].to_numpy(dtype=np.float32)[idx]),
            torch.from_numpy(df[["tl_dist_1", "tl_dist_2", "tl_dist_3", "tl_dist_4"]].to_numpy(dtype=np.float32)[idx]),
        )

    train_ds = TensorDataset(*make_tensors(train_idx))
    val_ds = TensorDataset(*make_tensors(val_idx))
    embed_ds = TensorDataset(
        torch.from_numpy(df["poi_idx"].to_numpy(dtype=np.int64)),
        torch.from_numpy(df["category_idx"].to_numpy(dtype=np.int64)),
        torch.from_numpy(df["cluster_idx"].to_numpy(dtype=np.int64)),
        torch.from_numpy(df[numeric_cols].to_numpy(dtype=np.float32)),
    )

    def build_loader(dataset, batch_size, shuffle):
        kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": cfg.num_workers,
                  "pin_memory": cfg.device.startswith("cuda")}
        return DataLoader(dataset, **kwargs)

    train_loader = build_loader(train_ds, cfg.batch_size, True)
    val_loader = build_loader(val_ds, cfg.batch_size, False)
    embed_loader = build_loader(embed_ds, cfg.embed_batch_size, False)

    model = POIAutoEncoder(num_pois, num_categories, num_clusters, cfg).to(cfg.device)
    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        totals = {"loss": 0.0, "category": 0.0, "cluster": 0.0, "latlon": 0.0, "total": 0.0, "tl": 0.0}
        steps = 0
        for batch in loader:
            poi_idx, category_idx, cluster_idx, numeric, latlon_t, total_t, tl_t = move_batch_to_device(batch, cfg.device)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=autocast_enabled):
                outputs = model(poi_idx, category_idx, cluster_idx, numeric)
                _, parts = compute_losses(cfg, outputs, category_idx, cluster_idx, latlon_t, total_t, tl_t)
            for k in totals:
                totals[k] += parts[k]
            steps += 1
        return {k: v / max(1, steps) for k, v in totals.items()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled)

    best_val_loss, best_state, patience_counter = float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {"loss": 0.0, "category": 0.0, "cluster": 0.0, "latlon": 0.0, "total": 0.0, "tl": 0.0}
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for batch in pbar:
            poi_idx, category_idx, cluster_idx, numeric, latlon_t, total_t, tl_t = move_batch_to_device(batch, cfg.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=autocast_enabled):
                outputs = model(poi_idx, category_idx, cluster_idx, numeric)
                loss, parts = compute_losses(cfg, outputs, category_idx, cluster_idx, latlon_t, total_t, tl_t)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for k in running:
                running[k] += parts[k]
            steps += 1
            pbar.set_postfix(loss=f"{running['loss'] / steps:.4f}")

        val_metrics = evaluate(val_loader)
        scheduler.step(val_metrics["loss"])
        print(f"[Epoch {epoch}] train_loss={running['loss'] / steps:.4f} val_loss={val_metrics['loss']:.4f}")

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    @torch.inference_mode()
    def extract_embeddings(loader):
        model.eval()
        chunks = []
        for batch in tqdm(loader, desc="Encoding POIs"):
            poi_idx, category_idx, cluster_idx, numeric = move_batch_to_device(batch, cfg.device)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=autocast_enabled):
                z = model.encode(poi_idx, category_idx, cluster_idx, numeric)
            z = F.normalize(z, p=2, dim=-1)
            chunks.append(z.detach().cpu().to(torch.float32))
        return torch.cat(chunks, dim=0).numpy()

    poi_embeddings = extract_embeddings(embed_loader)
    print("embedding shape:", poi_embeddings.shape)

    out_col.create_index("venue_id", unique=True)
    out_col.create_index([("model_name", 1), ("poi_idx", 1)])
    timestamp = datetime.utcnow()
    ops = []
    for row, emb in tqdm(zip(df.itertuples(index=False), poi_embeddings), total=len(df), desc="Writing embeddings"):
        raw_category_id, raw_cluster_id = int(row.category_id), int(row.loc_cluster_id_k5)
        category_id_value = None if raw_category_id == cfg.unknown_category_value else raw_category_id
        cluster_id_value = None if raw_cluster_id == cfg.unknown_cluster_value else raw_cluster_id
        doc = {
            "venue_id": row.venue_id, "poi_idx": int(row.poi_idx),
            "category_id": category_id_value, "category_idx": int(row.category_idx), "category": row.category,
            "cluster_id": cluster_id_value, "cluster_idx": int(row.cluster_idx), "loc_cluster_id_k5": cluster_id_value,
            "latitude": float(row.latitude), "longitude": float(row.longitude), "country": row.country,
            "visit1": int(row.visit1), "visit2": int(row.visit2), "visit3": int(row.visit3), "visit4": int(row.visit4),
            "total_visits": int(row.total_visits), "log_total_visits": float(row.log_total_visits),
            "tl_distribution": [float(row.tl_dist_1), float(row.tl_dist_2), float(row.tl_dist_3), float(row.tl_dist_4)],
            "poi_embedding": emb.astype(np.float32).tolist(), "embedding_dim": int(cfg.latent_dim),
            "embedding_norm": "l2", "model_name": cfg.model_name, "source_collection": cfg.src_collection,
            "updated_at": timestamp,
        }
        ops.append(UpdateOne({"venue_id": row.venue_id}, {"$set": doc}, upsert=True))
        if len(ops) >= cfg.bulk_write_size:
            out_col.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        out_col.bulk_write(ops, ordered=False)

    meta_doc = {
        "_id": cfg.model_name, "model_name": cfg.model_name, "created_at": timestamp,
        "source_collection": cfg.src_collection, "output_collection": cfg.out_collection,
        "num_pois": int(num_pois), "num_categories": int(num_categories), "num_clusters": int(num_clusters),
        "train_size": int(len(train_idx)), "val_size": int(len(val_idx)), "best_val_loss": float(best_val_loss),
        "numeric_features": numeric_cols,
        "normalization": {"mean": {k: float(train_means[k]) for k in norm_cols},
                           "std": {k: float(train_stds[k]) for k in norm_cols}},
        "hyperparams": asdict(cfg),
    }
    meta_col.replace_one({"_id": cfg.model_name}, meta_doc, upsert=True)
    print("stored embeddings:", out_col.estimated_document_count())


if __name__ == "__main__":
    main()
