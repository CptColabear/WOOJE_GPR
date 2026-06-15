import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from tqdm.auto import tqdm


HOST = "10.255.68.40"
PORT = 27017
DB_NAME = "ejoow"

GROUP_COL_NAME = "group_embeddings"
POI_COL_NAME = "poi_embeddings_mlp_v1_a"

DEFAULT_KS = [5, 10, 20, 50, 100]

GROUP_PROJECTION = {
    "_id": 1,
    "Group_ID": 1,
    "VenueID": 1,
    "TL": 1,
    "cluster_id_k": 1,
    "Latitude": 1,
    "Longitude": 1,
    "group_embedding": 1,
    "dim": 1,
}

POI_PROJECTION = {
    "_id": 0,
    "venue_id": 1,
    "category_id": 1,
    "category": 1,
    "cluster_id": 1,
    "latitude": 1,
    "longitude": 1,
    "visit1": 1,
    "visit2": 1,
    "visit3": 1,
    "visit4": 1,
    "poi_embedding": 1,
    "embedding_dim": 1,
}


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    mode: str
    allow_fallback: bool


@dataclass
class RunConfig:
    seed: int
    max_groups: int
    train_ratio: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    radius_km: float
    candidate_limit: int
    topk_random: int
    output_json: str


SCENARIOS = [
    ScenarioConfig(
        name="current_progress",
        description="same TL OR same cluster_id OR nearby candidate",
        mode="union",
        allow_fallback=True,
    ),
    ScenarioConfig(
        name="hard_negative_cluster_near",
        description="same cluster_id AND nearby hard negative",
        mode="cluster_near",
        allow_fallback=False,
    ),
]


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Train group-to-POI recommender with BPR and evaluate Top-K metrics."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=4096)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--radius-km", type=float, default=5.0)
    parser.add_argument("--candidate-limit", type=int, default=300)
    parser.add_argument("--topk-random", type=int, default=20)
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(Path(__file__).with_name("rec1_results.json")),
    )
    args = parser.parse_args()

    candidate_limit = max(args.candidate_limit, max(DEFAULT_KS))
    return RunConfig(
        seed=args.seed,
        max_groups=args.max_groups,
        train_ratio=args.train_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        radius_km=args.radius_km,
        candidate_limit=candidate_limit,
        topk_random=args.topk_random,
        output_json=args.output_json,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batched(values: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def count_batches(num_items: int, batch_size: int) -> int:
    if num_items <= 0:
        return 0
    return (num_items + batch_size - 1) // batch_size


def ensure_index(collection, keys, **kwargs):
    try:
        return collection.create_index(keys, **kwargs)
    except OperationFailure as exc:
        if exc.code == 86:
            print(f"skip conflicting index on {collection.name}: {keys}")
            return None
        raise


def to_np_vector(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D vector, got shape={arr.shape}")
    return arr


def to_optional_int(value) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def tl_visit_field(tl: int) -> str:
    tl = int(tl)
    if tl not in (1, 2, 3, 4):
        raise ValueError(f"unsupported TL: {tl}")
    return f"visit{tl}"


def bbox_from_radius_km(latitude: float, radius_km: float) -> Tuple[float, float]:
    lat_margin = radius_km / 111.32
    cos_lat = max(math.cos(math.radians(float(latitude))), 1e-6)
    lon_margin = radius_km / (111.32 * cos_lat)
    return lat_margin, lon_margin


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371.0088 * (2.0 * math.asin(math.sqrt(a)))


class MongoEmbeddingStore:
    def __init__(self, group_col, poi_col):
        self.group_col = group_col
        self.poi_col = poi_col
        self.positive_poi_cache: Dict[str, dict] = {}
        self.negative_candidate_cache: Dict[Tuple[str, str, float, int], List[dict]] = {}

    def warm_indexes(self) -> None:
        ensure_index(self.group_col, [("VenueID", 1), ("TL", 1)])
        ensure_index(self.group_col, [("cluster_id_k", 1)])
        ensure_index(self.poi_col, "venue_id", unique=True)
        ensure_index(self.poi_col, [("cluster_id", 1)])
        ensure_index(self.poi_col, [("latitude", 1), ("longitude", 1)])
        for tl in range(1, 5):
            ensure_index(self.poi_col, [(f"visit{tl}", 1)])

    def load_positive_poi(self, group_doc: dict) -> dict:
        venue_id = str(group_doc["VenueID"])
        cached = self.positive_poi_cache.get(venue_id)
        if cached is not None:
            return cached

        doc = self.poi_col.find_one(
            {"venue_id": venue_id, "poi_embedding": {"$exists": True}},
            POI_PROJECTION,
        )
        if doc is None:
            raise ValueError(f"positive POI embedding not found: venue_id={venue_id}")

        self.positive_poi_cache[venue_id] = doc
        return doc

    def load_valid_groups(self, limit: int) -> List[dict]:
        query = {
            "group_embedding": {"$exists": True},
            "VenueID": {"$exists": True, "$ne": None},
            "TL": {"$in": [1, 2, 3, 4]},
            "Latitude": {"$exists": True, "$ne": None},
            "Longitude": {"$exists": True, "$ne": None},
        }

        target_total = min(limit, self.group_col.count_documents(query))
        print(f"[load_valid_groups] loading group docs from Mongo: target={target_total}")

        raw_groups = []
        cursor = self.group_col.find(query, GROUP_PROJECTION).limit(limit).batch_size(512)
        load_progress = tqdm(total=target_total, desc="load group docs", unit="group")
        for doc in cursor:
            raw_groups.append(doc)
            load_progress.update(1)
            load_progress.set_postfix(loaded=len(raw_groups))
        load_progress.close()

        if not raw_groups:
            return []

        print(f"[load_valid_groups] loaded raw groups: {len(raw_groups)}")
        venue_ids = sorted({str(doc["VenueID"]) for doc in raw_groups})
        valid_venue_ids = set()
        total_batches = count_batches(len(venue_ids), 500)
        venue_progress = tqdm(
            batched(venue_ids, 500),
            total=total_batches,
            desc="validate positive pois",
            unit="batch",
        )
        for venue_batch in venue_progress:
            cursor = self.poi_col.find(
                {"venue_id": {"$in": list(venue_batch)}, "poi_embedding": {"$exists": True}},
                {"_id": 0, "venue_id": 1},
            )
            valid_venue_ids.update(str(doc["venue_id"]) for doc in cursor)
            venue_progress.set_postfix(valid_venues=len(valid_venue_ids))

        valid_groups = [doc for doc in raw_groups if str(doc["VenueID"]) in valid_venue_ids]
        print(
            "[load_valid_groups] finished "
            f"raw_groups={len(raw_groups)} valid_venues={len(valid_venue_ids)} valid_groups={len(valid_groups)}"
        )

        return valid_groups

    def _build_candidate_query(
        self,
        group_doc: dict,
        positive_venue_id: str,
        scenario: ScenarioConfig,
        radius_km: float,
    ) -> Optional[dict]:
        tl = int(group_doc["TL"])
        cluster_id = to_optional_int(group_doc.get("cluster_id_k"))
        latitude = float(group_doc["Latitude"])
        longitude = float(group_doc["Longitude"])
        visit_field = tl_visit_field(tl)
        lat_margin, lon_margin = bbox_from_radius_km(latitude, radius_km)

        base_query = {
            "venue_id": {"$ne": str(positive_venue_id)},
            "poi_embedding": {"$exists": True},
        }

        if scenario.mode == "union":
            or_filters = [{visit_field: {"$gt": 0}}]
            if cluster_id is not None:
                or_filters.append({"cluster_id": cluster_id})
            or_filters.append(
                {
                    "latitude": {"$gte": latitude - lat_margin, "$lte": latitude + lat_margin},
                    "longitude": {"$gte": longitude - lon_margin, "$lte": longitude + lon_margin},
                }
            )
            base_query["$or"] = or_filters
            return base_query

        if scenario.mode == "cluster_near":
            if cluster_id is None:
                return None
            base_query["cluster_id"] = cluster_id
            base_query["latitude"] = {"$gte": latitude - lat_margin, "$lte": latitude + lat_margin}
            base_query["longitude"] = {"$gte": longitude - lon_margin, "$lte": longitude + lon_margin}
            return base_query

        raise ValueError(f"unsupported scenario mode: {scenario.mode}")

    def _annotate_candidate(
        self,
        candidate_doc: dict,
        group_doc: dict,
        radius_km: float,
    ) -> dict:
        tl = int(group_doc["TL"])
        cluster_id = to_optional_int(group_doc.get("cluster_id_k"))
        latitude = float(group_doc["Latitude"])
        longitude = float(group_doc["Longitude"])
        visit_field = tl_visit_field(tl)

        same_tl = int(candidate_doc.get(visit_field, 0) or 0) > 0
        same_cluster = cluster_id is not None and to_optional_int(candidate_doc.get("cluster_id")) == cluster_id

        if candidate_doc.get("latitude") is None or candidate_doc.get("longitude") is None:
            distance_km = float("inf")
            nearby = False
        else:
            distance_km = haversine_km(latitude, longitude, candidate_doc["latitude"], candidate_doc["longitude"])
            nearby = distance_km <= radius_km

        item = dict(candidate_doc)
        item["same_tl"] = bool(same_tl)
        item["same_cluster"] = bool(same_cluster)
        item["nearby"] = bool(nearby)
        item["distance_km"] = float(distance_km)
        return item

    def _candidate_matches(self, item: dict, scenario: ScenarioConfig) -> bool:
        if scenario.mode == "union":
            return bool(item["same_tl"] or item["same_cluster"] or item["nearby"])
        if scenario.mode == "cluster_near":
            return bool(item["same_cluster"] and item["nearby"])
        return False

    def _candidate_priority(self, item: dict, scenario: ScenarioConfig) -> Tuple[int, float, str]:
        if scenario.mode == "union":
            priority = int(item["same_tl"]) + int(item["same_cluster"]) + int(item["nearby"])
            return (-priority, float(item["distance_km"]), str(item["venue_id"]))
        if scenario.mode == "cluster_near":
            priority = int(item["same_cluster"]) + int(item["nearby"])
            return (-priority, float(item["distance_km"]), str(item["venue_id"]))
        return (0, float(item["distance_km"]), str(item["venue_id"]))

    def fetch_negative_candidates(
        self,
        group_doc: dict,
        positive_venue_id: str,
        scenario: ScenarioConfig,
        radius_km: float,
        limit: int,
    ) -> List[dict]:
        cache_key = (scenario.name, str(group_doc["_id"]), float(radius_km), int(limit))
        cached = self.negative_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        query = self._build_candidate_query(group_doc, positive_venue_id, scenario, radius_km)
        if query is None:
            return []

        seen = set()
        candidates: List[dict] = []
        cursor = self.poi_col.find(query, POI_PROJECTION).limit(limit * 5)

        for doc in cursor:
            venue_id = str(doc["venue_id"])
            if venue_id in seen:
                continue
            seen.add(venue_id)

            item = self._annotate_candidate(doc, group_doc, radius_km)
            if not self._candidate_matches(item, scenario):
                continue
            candidates.append(item)

        candidates.sort(key=lambda item: self._candidate_priority(item, scenario))

        if scenario.allow_fallback and len(candidates) < limit:
            fallback_query = {
                "venue_id": {"$ne": str(positive_venue_id)},
                "poi_embedding": {"$exists": True},
            }
            fallback_cursor = self.poi_col.find(fallback_query, POI_PROJECTION).limit(limit)
            for doc in fallback_cursor:
                venue_id = str(doc["venue_id"])
                if venue_id in seen:
                    continue
                seen.add(venue_id)

                item = self._annotate_candidate(doc, group_doc, radius_km)
                candidates.append(item)
                if len(candidates) >= limit:
                    break

            candidates.sort(key=lambda item: self._candidate_priority(item, scenario))

        result = candidates[:limit]
        self.negative_candidate_cache[cache_key] = result
        return result


class GroupPoiScorer(nn.Module):
    def __init__(self, group_dim: int, poi_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.group_proj = nn.Sequential(
            nn.Linear(group_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.poi_proj = nn.Sequential(
            nn.Linear(poi_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, group_emb: torch.Tensor, poi_emb: torch.Tensor) -> torch.Tensor:
        g = F.normalize(self.group_proj(group_emb), p=2, dim=-1)
        p = F.normalize(self.poi_proj(poi_emb), p=2, dim=-1)
        pair = torch.cat([g, p, g * p, torch.abs(g - p)], dim=-1)
        return self.head(pair).squeeze(-1)


def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_score - neg_score).mean()


def split_groups(groups: Sequence[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    train_size = max(1, int(len(groups) * train_ratio))
    train_size = min(train_size, len(groups) - 1)
    train_groups = groups[:train_size]
    test_groups = groups[train_size:]
    return train_groups, test_groups


def build_triplet(
    store: MongoEmbeddingStore,
    group_doc: dict,
    scenario: ScenarioConfig,
    radius_km: float,
    candidate_limit: int,
    topk_random: int,
) -> Optional[dict]:
    try:
        positive_doc = store.load_positive_poi(group_doc)
    except ValueError:
        return None

    negative_candidates = store.fetch_negative_candidates(
        group_doc=group_doc,
        positive_venue_id=str(positive_doc["venue_id"]),
        scenario=scenario,
        radius_km=radius_km,
        limit=candidate_limit,
    )
    if not negative_candidates:
        return None

    topk = min(len(negative_candidates), max(1, int(topk_random)))
    negative_doc = random.choice(negative_candidates[:topk])

    return {
        "group_doc": group_doc,
        "positive_doc": positive_doc,
        "negative_doc": negative_doc,
        "group_vec": to_np_vector(group_doc["group_embedding"]),
        "positive_vec": to_np_vector(positive_doc["poi_embedding"]),
        "negative_vec": to_np_vector(negative_doc["poi_embedding"]),
    }


def prepare_groups_for_scenario(
    store: MongoEmbeddingStore,
    groups: Sequence[dict],
    scenario: ScenarioConfig,
    radius_km: float,
    candidate_limit: int,
) -> List[dict]:
    valid_groups = []
    skipped = 0
    progress = tqdm(groups, total=len(groups), desc=f"prepare {scenario.name}", leave=False, unit="group")
    for group_doc in progress:
        try:
            positive_doc = store.load_positive_poi(group_doc)
        except ValueError:
            skipped += 1
            progress.set_postfix(valid=len(valid_groups), skipped=skipped)
            continue

        negative_candidates = store.fetch_negative_candidates(
            group_doc=group_doc,
            positive_venue_id=str(positive_doc["venue_id"]),
            scenario=scenario,
            radius_km=radius_km,
            limit=candidate_limit,
        )
        if negative_candidates:
            valid_groups.append(group_doc)
        else:
            skipped += 1
        progress.set_postfix(valid=len(valid_groups), skipped=skipped)
    progress.close()
    print(
        f"[prepare {scenario.name}] input={len(groups)} valid={len(valid_groups)} skipped={skipped}"
    )
    return valid_groups


def infer_dimensions(store: MongoEmbeddingStore, groups: Sequence[dict]) -> Tuple[int, int]:
    for group_doc in groups:
        try:
            positive_doc = store.load_positive_poi(group_doc)
        except ValueError:
            continue
        group_dim = int(to_np_vector(group_doc["group_embedding"]).shape[0])
        poi_dim = int(to_np_vector(positive_doc["poi_embedding"]).shape[0])
        return group_dim, poi_dim
    raise RuntimeError("failed to infer embedding dimensions from loaded groups")


def train_model(
    store: MongoEmbeddingStore,
    model: GroupPoiScorer,
    optimizer: torch.optim.Optimizer,
    train_groups: Sequence[dict],
    scenario: ScenarioConfig,
    config: RunConfig,
    device: str,
) -> List[dict]:
    history = []
    train_groups = list(train_groups)
    num_batches = count_batches(len(train_groups), config.batch_size)

    for epoch in range(config.epochs):
        random.shuffle(train_groups)
        epoch_losses = []
        skipped = 0

        progress = tqdm(
            range(0, len(train_groups), config.batch_size),
            desc=f"{scenario.name} epoch {epoch + 1}/{config.epochs}",
            total=num_batches,
            unit="batch",
        )
        for start in progress:
            batch_docs = train_groups[start : start + config.batch_size]
            batch_items = []
            for group_doc in batch_docs:
                triplet = build_triplet(
                    store=store,
                    group_doc=group_doc,
                    scenario=scenario,
                    radius_km=config.radius_km,
                    candidate_limit=config.candidate_limit,
                    topk_random=config.topk_random,
                )
                if triplet is not None:
                    batch_items.append(triplet)

            skipped += len(batch_docs) - len(batch_items)
            if not batch_items:
                continue

            group_batch = torch.tensor(
                np.stack([item["group_vec"] for item in batch_items]),
                dtype=torch.float32,
                device=device,
            )
            pos_batch = torch.tensor(
                np.stack([item["positive_vec"] for item in batch_items]),
                dtype=torch.float32,
                device=device,
            )
            neg_batch = torch.tensor(
                np.stack([item["negative_vec"] for item in batch_items]),
                dtype=torch.float32,
                device=device,
            )

            model.train()
            optimizer.zero_grad()
            pos_scores = model(group_batch, pos_batch)
            neg_scores = model(group_batch, neg_batch)
            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            mean_loss_so_far = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                mean=f"{mean_loss_so_far:.4f}",
                valid=len(batch_items),
                skipped=skipped,
            )

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "num_steps": len(epoch_losses),
                "skipped": skipped,
            }
        )
        print(
            f"[train {scenario.name}] epoch={epoch + 1}/{config.epochs} "
            f"mean_loss={mean_loss:.4f} steps={len(epoch_losses)} skipped={skipped}"
        )
    return history


def compute_group_rank(
    model: GroupPoiScorer,
    store: MongoEmbeddingStore,
    group_doc: dict,
    scenario: ScenarioConfig,
    radius_km: float,
    candidate_limit: int,
    device: str,
) -> Optional[dict]:
    try:
        positive_doc = store.load_positive_poi(group_doc)
    except ValueError:
        return None

    negative_candidates = store.fetch_negative_candidates(
        group_doc=group_doc,
        positive_venue_id=str(positive_doc["venue_id"]),
        scenario=scenario,
        radius_km=radius_km,
        limit=candidate_limit,
    )
    if not negative_candidates:
        return None

    tl = int(group_doc["TL"])
    positive_item = dict(positive_doc)
    positive_item["same_tl"] = int(positive_doc.get(tl_visit_field(tl), 0) or 0) > 0
    group_cluster_id = to_optional_int(group_doc.get("cluster_id_k"))
    positive_cluster_id = to_optional_int(positive_doc.get("cluster_id"))
    positive_item["same_cluster"] = group_cluster_id is not None and positive_cluster_id == group_cluster_id
    if positive_doc.get("latitude") is None or positive_doc.get("longitude") is None:
        positive_item["distance_km"] = float("inf")
        positive_item["nearby"] = False
    else:
        distance_km = haversine_km(
            float(group_doc["Latitude"]),
            float(group_doc["Longitude"]),
            float(positive_doc["latitude"]),
            float(positive_doc["longitude"]),
        )
        positive_item["distance_km"] = float(distance_km)
        positive_item["nearby"] = distance_km <= radius_km

    rank_items = [positive_item] + list(negative_candidates)
    poi_batch = torch.tensor(
        np.stack([to_np_vector(item["poi_embedding"]) for item in rank_items]),
        dtype=torch.float32,
        device=device,
    )
    group_vec = torch.tensor(
        to_np_vector(group_doc["group_embedding"]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    group_batch = group_vec.expand(poi_batch.shape[0], -1)

    model.eval()
    with torch.inference_mode():
        scores = model(group_batch, poi_batch).detach().cpu().numpy()

    ranked_entries = []
    for item, score in sorted(zip(rank_items, scores), key=lambda pair: float(pair[1]), reverse=True):
        ranked_entries.append(
            {
                "venue_id": str(item["venue_id"]),
                "category": item.get("category"),
                "score": float(score),
                "is_positive": str(item["venue_id"]) == str(positive_doc["venue_id"]),
                "same_tl": bool(item.get("same_tl", False)),
                "same_cluster": bool(item.get("same_cluster", False)),
                "nearby": bool(item.get("nearby", False)),
                "distance_km": float(item.get("distance_km", float("inf"))),
            }
        )

    positive_rank = None
    for idx, entry in enumerate(ranked_entries, start=1):
        if entry["is_positive"]:
            positive_rank = idx
            break

    if positive_rank is None:
        return None

    return {
        "group_id": str(group_doc["_id"]),
        "group_meta": {
            "Group_ID": group_doc.get("Group_ID"),
            "VenueID": str(group_doc["VenueID"]),
            "TL": int(group_doc["TL"]),
            "cluster_id_k": to_optional_int(group_doc.get("cluster_id_k")),
            "Latitude": float(group_doc["Latitude"]),
            "Longitude": float(group_doc["Longitude"]),
        },
        "positive_rank": int(positive_rank),
        "ranked_entries": ranked_entries,
    }


def metrics_at_k(positive_rank: int, k: int) -> Dict[str, float]:
    if positive_rank <= k:
        recall = 1.0
        precision = 1.0 / float(k)
        f1 = 2.0 * precision * recall / (precision + recall)
        ndcg = 1.0 / math.log2(positive_rank + 1.0)
    else:
        recall = 0.0
        precision = 0.0
        f1 = 0.0
        ndcg = 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "ndcg": ndcg,
    }


def evaluate_model(
    model: GroupPoiScorer,
    store: MongoEmbeddingStore,
    eval_groups: Sequence[dict],
    scenario: ScenarioConfig,
    config: RunConfig,
    device: str,
    ks: Sequence[int],
) -> Dict[str, object]:
    metric_sums = {
        int(k): {"recall": 0.0, "precision": 0.0, "f1": 0.0, "ndcg": 0.0}
        for k in ks
    }
    sample_rank_result = None
    evaluated = 0

    progress = tqdm(eval_groups, total=len(eval_groups), desc=f"evaluate {scenario.name}", unit="group")
    for group_doc in progress:
        rank_result = compute_group_rank(
            model=model,
            store=store,
            group_doc=group_doc,
            scenario=scenario,
            radius_km=config.radius_km,
            candidate_limit=config.candidate_limit,
            device=device,
        )
        if rank_result is None:
            progress.set_postfix(evaluated=evaluated, last_rank="-")
            continue

        evaluated += 1
        if sample_rank_result is None:
            sample_rank_result = rank_result

        for k in ks:
            one_result = metrics_at_k(rank_result["positive_rank"], int(k))
            for key, value in one_result.items():
                metric_sums[int(k)][key] += float(value)
        progress.set_postfix(evaluated=evaluated, last_rank=rank_result["positive_rank"])

    metrics = {}
    for k in ks:
        if evaluated == 0:
            metrics[int(k)] = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "ndcg": 0.0}
            continue
        metrics[int(k)] = {
            key: float(metric_sums[int(k)][key] / evaluated)
            for key in ("recall", "precision", "f1", "ndcg")
        }

    topk_samples = {}
    if sample_rank_result is not None:
        for k in ks:
            topk_samples[int(k)] = sample_rank_result["ranked_entries"][: int(k)]

    return {
        "evaluated_groups": evaluated,
        "metrics": metrics,
        "sample_group": None if sample_rank_result is None else sample_rank_result["group_meta"],
        "sample_positive_rank": None if sample_rank_result is None else sample_rank_result["positive_rank"],
        "sample_topk": topk_samples,
    }


def run_scenario(
    store: MongoEmbeddingStore,
    train_groups: Sequence[dict],
    test_groups: Sequence[dict],
    scenario: ScenarioConfig,
    config: RunConfig,
    device: str,
    ks: Sequence[int],
) -> Dict[str, object]:
    print(f"[{scenario.name}] prepare train groups...")
    scenario_train_groups = prepare_groups_for_scenario(
        store=store,
        groups=train_groups,
        scenario=scenario,
        radius_km=config.radius_km,
        candidate_limit=config.candidate_limit,
    )
    print(f"[{scenario.name}] prepare test groups...")
    scenario_test_groups = prepare_groups_for_scenario(
        store=store,
        groups=test_groups,
        scenario=scenario,
        radius_km=config.radius_km,
        candidate_limit=config.candidate_limit,
    )

    if not scenario_train_groups:
        return {
            "description": scenario.description,
            "train_groups": 0,
            "test_groups": len(scenario_test_groups),
            "history": [],
            "evaluation": {
                "evaluated_groups": 0,
                "metrics": {int(k): {"recall": 0.0, "precision": 0.0, "f1": 0.0, "ndcg": 0.0} for k in ks},
                "sample_group": None,
                "sample_positive_rank": None,
                "sample_topk": {},
            },
        }

    group_dim, poi_dim = infer_dimensions(store, scenario_train_groups)
    model = GroupPoiScorer(
        group_dim=group_dim,
        poi_dim=poi_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    print(
        f"[{scenario.name}] start training "
        f"train_groups={len(scenario_train_groups)} test_groups={len(scenario_test_groups)}"
    )
    history = train_model(
        store=store,
        model=model,
        optimizer=optimizer,
        train_groups=scenario_train_groups,
        scenario=scenario,
        config=config,
        device=device,
    )
    print(f"[{scenario.name}] training complete. start evaluation...")
    evaluation = evaluate_model(
        model=model,
        store=store,
        eval_groups=scenario_test_groups,
        scenario=scenario,
        config=config,
        device=device,
        ks=ks,
    )

    return {
        "description": scenario.description,
        "train_groups": len(scenario_train_groups),
        "test_groups": len(scenario_test_groups),
        "history": history,
        "evaluation": evaluation,
    }


def print_metric_table(name: str, result: Dict[str, object], ks: Sequence[int]) -> None:
    evaluation = result["evaluation"]
    metrics = evaluation["metrics"]
    print()
    print(f"[{name}]")
    print(f"description: {result['description']}")
    print(f"train_groups: {result['train_groups']}")
    print(f"test_groups: {result['test_groups']}")
    print(f"evaluated_groups: {evaluation['evaluated_groups']}")
    print("K\trecall\tprecision\tf1\tndcg")
    for k in ks:
        row = metrics[int(k)]
        print(
            f"{k}\t"
            f"{row['recall']:.6f}\t"
            f"{row['precision']:.6f}\t"
            f"{row['f1']:.6f}\t"
            f"{row['ndcg']:.6f}"
        )

    sample_group = evaluation.get("sample_group")
    sample_positive_rank = evaluation.get("sample_positive_rank")
    if sample_group is not None:
        print("sample_group:", json.dumps(sample_group, ensure_ascii=False))
        print(f"sample_positive_rank: {sample_positive_rank}")
        sample_top5 = evaluation.get("sample_topk", {}).get(5, [])
        sample_top5_ids = [entry["venue_id"] for entry in sample_top5]
        print("sample_top5_venue_ids:", json.dumps(sample_top5_ids, ensure_ascii=False))


def make_jsonable(results: Dict[str, object]) -> Dict[str, object]:
    return json.loads(json.dumps(results, ensure_ascii=False))


def main() -> None:
    config = parse_args()
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("config:", json.dumps(asdict(config), ensure_ascii=False))

    client = MongoClient(HOST, PORT)
    db = client[DB_NAME]
    store = MongoEmbeddingStore(
        group_col=db[GROUP_COL_NAME],
        poi_col=db[POI_COL_NAME],
    )
    print("[setup] warming Mongo indexes...")
    store.warm_indexes()

    groups = store.load_valid_groups(config.max_groups)
    if len(groups) < 2:
        raise RuntimeError("not enough valid groups to split train/test")

    train_groups, test_groups = split_groups(groups, config.train_ratio, config.seed)
    print(f"loaded valid groups: {len(groups)}")
    print(f"train split: {len(train_groups)}")
    print(f"test split: {len(test_groups)}")

    results = {
        "config": asdict(config),
        "device": device,
        "num_loaded_groups": len(groups),
        "num_train_groups": len(train_groups),
        "num_test_groups": len(test_groups),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        print()
        print(f"running scenario: {scenario.name}")
        scenario_result = run_scenario(
            store=store,
            train_groups=train_groups,
            test_groups=test_groups,
            scenario=scenario,
            config=config,
            device=device,
            ks=DEFAULT_KS,
        )
        results["scenarios"][scenario.name] = scenario_result
        print_metric_table(scenario.name, scenario_result, DEFAULT_KS)

    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"saved results to: {output_path}")


if __name__ == "__main__":
    main()
