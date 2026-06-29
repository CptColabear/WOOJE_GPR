# -*- coding: utf-8 -*-
"""
Group Embedding + POI Embedding 기반 MLP Ranking 추천 (Tokyo / ejoow2)
- 실험 1) baseline: 같은 cluster negative
- 실험 2) hard negative: 같은 cluster + 가까운 거리
- 평가: Recall@K, Precision@K, F1@K, NDCG@K
- K = 5, 10, 20, 50, 100
- 중간 진행 상황 로그 포함

rec2copy.py의 JP(Tokyo)/ejoow2 버전. DB_NAME만 바뀌고 나머지 로직(모델, 학습,
후보군 구성, 평가)은 동일합니다 (그대로 재사용).

주의: 원본의 STEP 6 fixed-count split(TARGET_TRAIN_GROUPS=1671 /
TARGET_TEST_GROUPS=361)은 US 전용 값이라 JP의 실제 usable_group_docs 수보다
클 경우 import 시점에 ValueError로 죽는다. rec2_2_jp.py가 import 직후
train_docs/valid_docs/test_docs를 70/20/10 비율로 즉시 덮어쓰므로, 여기서는
이 모듈이 단독으로 안전하게 import될 수 있도록 자리표시자(placeholder) 분할만
수행한다 (개수와 무관하게 절대 실패하지 않음).

실행 방법:
python -m wooje.jp.recommedation.rec2_2_jp
"""

import math
import random
import json
import time
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymongo import MongoClient

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import wooje.recommedation.rec1 as base


# =========================================================
# 1. 설정
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONGO_URI = "mongodb://10.255.68.40:27017/"
DB_NAME = "ejoow2"

GROUP_COLLECTION = "99. group_embeddings_v2"
POI_COLLECTION = "poi_embeddings_mlp_v1_a"

EMB_DIM = 64
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 15
TOP_K_LIST = [5, 10, 20, 50, 100]

MAX_CANDIDATES = 300
HARD_NEG_TOP_M = 30

TARGET_TRAIN_GROUPS = 1671  # unused for JP -- kept only for reference; see note above
TARGET_TEST_GROUPS = 361    # unused for JP -- rec2_2_jp.py applies a 70/20/10 ratio split instead
LOAD_VALID_GROUP_LIMIT = 5000

PRINT_EVERY_BATCH = 50
TOPK_PROGRESS_EVERY = 200

print("=" * 100)
print("[START] rec2copy_jp.py (Tokyo / ejoow2)")
print(f"[INFO] DEVICE = {DEVICE}")
print("=" * 100)


# =========================================================
# 2. 유틸 함수
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None

    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def precision_at_k(ranked_ids, gt_id, k):
    topk = ranked_ids[:k]
    return 1.0 / k if gt_id in topk else 0.0


def recall_at_k(ranked_ids, gt_id, k):
    topk = ranked_ids[:k]
    return 1.0 if gt_id in topk else 0.0


def f1_at_k(ranked_ids, gt_id, k):
    p = precision_at_k(ranked_ids, gt_id, k)
    r = recall_at_k(ranked_ids, gt_id, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(ranked_ids, gt_id, k):
    topk = ranked_ids[:k]
    if gt_id not in topk:
        return 0.0
    rank = topk.index(gt_id) + 1
    return 1.0 / math.log2(rank + 1)


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()


def split_groups_fixed_counts(group_docs, train_size, test_size, seed):
    group_docs = list(group_docs)
    rng = random.Random(seed)
    rng.shuffle(group_docs)

    required = train_size + test_size
    if len(group_docs) < required:
        raise ValueError(
            f"usable group docs가 부족합니다. required={required}, available={len(group_docs)}"
        )

    selected = group_docs[:required]
    train_docs = selected[:train_size]
    test_docs = selected[train_size : train_size + test_size]
    return train_docs, test_docs


# =========================================================
# 3. 모델
# =========================================================
class GroupPOIScorerMLP(nn.Module):
    def __init__(self, emb_dim=64, extra_dim=2, hidden_dims=(256, 128), dropout=0.1):
        super().__init__()
        input_dim = emb_dim * 4 + extra_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, group_emb, poi_emb, extra_feat):
        prod = group_emb * poi_emb
        diff = torch.abs(group_emb - poi_emb)
        x = torch.cat([group_emb, poi_emb, prod, diff, extra_feat], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score


# =========================================================
# 4. MongoDB 연결 및 데이터 로딩
# =========================================================
print("\n[STEP 1] MongoDB 연결 시작")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
group_col = db[GROUP_COLLECTION]
poi_col = db[POI_COLLECTION]
store = base.MongoEmbeddingStore(group_col=group_col, poi_col=poi_col)
print("[STEP 1] Mongo index 준비 시작")
store.warm_indexes()
print("[STEP 1] MongoDB 연결 완료")

print("\n[STEP 2] load_valid_groups 시작")
t0 = time.time()
valid_group_docs_raw = store.load_valid_groups(limit=LOAD_VALID_GROUP_LIMIT)
print(f"[STEP 2] load_valid_groups 완료: {len(valid_group_docs_raw)}개 / {time.time() - t0:.2f}초")

print("\n[STEP 3] poi_embeddings_mlp_v1_a 로딩 시작")
t0 = time.time()
poi_docs_raw = list(poi_col.find({}, {
    "venue_id": 1,
    "poi_embedding": 1,
    "cluster_id": 1,
    "latitude": 1,
    "longitude": 1,
    "category": 1,
    "log_total_visits": 1,
    "total_visits": 1
}))
print(f"[STEP 3] poi_embeddings_mlp_v1_a 로딩 완료: {len(poi_docs_raw)}개 / {time.time() - t0:.2f}초")


# =========================================================
# 5. POI 인덱스 구성
# =========================================================
print("\n[STEP 4] POI 인덱스 구성 시작")

poi_by_venue = {}
poi_by_cluster = defaultdict(list)
all_pois = []

for i, doc in enumerate(poi_docs_raw, start=1):
    venue_id = doc.get("venue_id")
    emb = doc.get("poi_embedding")

    if venue_id is None or emb is None:
        continue
    if len(emb) != EMB_DIM:
        continue

    row = {
        "venue_id": venue_id,
        "embedding": np.array(emb, dtype=np.float32),
        "cluster_id": doc.get("cluster_id"),
        "latitude": doc.get("latitude"),
        "longitude": doc.get("longitude"),
        "category": doc.get("category"),
        "log_total_visits": float(doc.get("log_total_visits", 0.0)),
        "total_visits": int(doc.get("total_visits", 0)),
    }

    poi_by_venue[venue_id] = row
    all_pois.append(row)

    cluster_id = row["cluster_id"]
    if cluster_id is not None:
        poi_by_cluster[cluster_id].append(row)

    if i % 50000 == 0:
        print(f"[STEP 4] POI 인덱스 진행 중... {i}/{len(poi_docs_raw)}")

print(f"[STEP 4] usable pois = {len(all_pois)}")
print(f"[STEP 4] cluster count = {len(poi_by_cluster)}")

# Tokyo's location clustering converged on k=4 (vs US's k=5), so each cluster
# holds ~3x more POIs on average (~49k vs ~15.6k). The original hard-negative
# sampler below looped over every POI in a cluster in pure Python per training
# example per epoch -- fine at US's scale, but at JP's scale this is the
# dominant cost (52,500 calls x ~49k-item python loop for a 15-epoch run).
# Precompute per-cluster coordinate arrays + venue_id->index maps once here so
# sample_negative_hard can do a single vectorized numpy haversine call instead.
poi_cluster_coords = {
    cid: np.array([[p["latitude"], p["longitude"]] for p in pois], dtype=np.float64)
    for cid, pois in poi_by_cluster.items()
}
poi_cluster_index = {
    cid: {p["venue_id"]: i for i, p in enumerate(pois)}
    for cid, pois in poi_by_cluster.items()
}
all_pois_coords = np.array([[p["latitude"], p["longitude"]] for p in all_pois], dtype=np.float64)
all_pois_index = {p["venue_id"]: i for i, p in enumerate(all_pois)}


def haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r = np.radians(lat2_arr)
    lon2r = np.radians(lon2_arr)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c


# =========================================================
# 6. 그룹 데이터 정제
# =========================================================
print("\n[STEP 5] 그룹 데이터 정제 시작")

usable_group_docs = []

for i, gdoc in enumerate(valid_group_docs_raw, start=1):
    group_emb = gdoc.get("group_embedding")
    pos_venue_id = gdoc.get("VenueID")

    if group_emb is None or pos_venue_id is None:
        continue
    if pos_venue_id not in poi_by_venue:
        continue
    if len(group_emb) != EMB_DIM:
        continue

    usable_group_docs.append({
        "Group_ID": gdoc.get("Group_ID"),
        "group_embedding": np.array(group_emb, dtype=np.float32),
        "VenueID": pos_venue_id,
        "cluster_id_k": gdoc.get("cluster_id_k"),
        "Latitude": gdoc.get("Latitude"),
        "Longitude": gdoc.get("Longitude"),
        "TL": gdoc.get("TL"),
    })

    if i % 1000 == 0:
        print(f"[STEP 5] 그룹 정제 진행 중... {i}/{len(valid_group_docs_raw)}")

print(f"[STEP 5] usable group docs = {len(usable_group_docs)}")

if len(usable_group_docs) < 10:
    raise ValueError("usable_group_docs 수가 너무 적습니다. 데이터 확인이 필요합니다.")


# =========================================================
# 7. Train / Valid / Test 분리 (placeholder -- rec2_2_jp.py가 즉시 덮어씀)
# =========================================================
print("\n[STEP 6] train/valid/test 분리 시작 (placeholder, rec2_2_jp.py에서 70/20/10으로 재설정됨)")

_rng = random.Random(SEED)
_shuffled = list(usable_group_docs)
_rng.shuffle(_shuffled)
_cut = max(1, int(len(_shuffled) * 0.5))
train_docs = _shuffled[:_cut]
test_docs = _shuffled[_cut:] or _shuffled[:1]
valid_docs = list(test_docs)

print(f"[STEP 6] (placeholder) train size = {len(train_docs)}")
print(f"[STEP 6] (placeholder) valid size = {len(valid_docs)}")
print(f"[STEP 6] (placeholder) test size  = {len(test_docs)}")


# =========================================================
# 8. Negative Sampling
# =========================================================
def sample_negative_baseline(pos_venue_id, cluster_id=None, max_tries=30):
    if cluster_id is not None and cluster_id in poi_by_cluster:
        candidates = poi_by_cluster[cluster_id]
        for _ in range(max_tries):
            neg = random.choice(candidates)
            if neg["venue_id"] != pos_venue_id:
                return neg

    for _ in range(max_tries * 5):
        neg = random.choice(all_pois)
        if neg["venue_id"] != pos_venue_id:
            return neg

    return None


def sample_negative_hard(group_lat, group_lon, pos_venue_id, cluster_id=None, top_m=HARD_NEG_TOP_M):
    if cluster_id is not None and cluster_id in poi_by_cluster:
        pool = poi_by_cluster[cluster_id]
        coords = poi_cluster_coords[cluster_id]
        index = poi_cluster_index[cluster_id]
    else:
        pool = all_pois
        coords = all_pois_coords
        index = all_pois_index

    if group_lat is None or group_lon is None or len(pool) == 0:
        return sample_negative_baseline(pos_venue_id, cluster_id)

    dists = haversine_km_vec(group_lat, group_lon, coords[:, 0], coords[:, 1])

    pos_idx = index.get(pos_venue_id)
    n_exclude = 0
    if pos_idx is not None:
        dists = dists.copy()
        dists[pos_idx] = np.inf
        n_exclude = 1

    m = min(top_m, len(pool) - n_exclude)
    if m <= 0:
        return sample_negative_baseline(pos_venue_id, cluster_id)

    nearest_idx = np.argpartition(dists, m - 1)[:m]
    chosen = nearest_idx[np.random.randint(len(nearest_idx))]
    return pool[chosen]


# =========================================================
# 9. Dataset
# =========================================================
class GroupPOIRankingDataset(Dataset):
    def __init__(self, group_docs, sampling_mode="baseline"):
        assert sampling_mode in ["baseline", "hard"]
        self.group_docs = group_docs
        self.sampling_mode = sampling_mode

    def __len__(self):
        return len(self.group_docs)

    def __getitem__(self, idx):
        gdoc = self.group_docs[idx]

        group_emb = gdoc["group_embedding"]
        pos_venue_id = gdoc["VenueID"]
        cluster_id = gdoc.get("cluster_id_k")
        group_lat = gdoc.get("Latitude")
        group_lon = gdoc.get("Longitude")

        pos_poi = poi_by_venue[pos_venue_id]

        if self.sampling_mode == "baseline":
            neg_poi = sample_negative_baseline(pos_venue_id, cluster_id)
        else:
            neg_poi = sample_negative_hard(group_lat, group_lon, pos_venue_id, cluster_id)

        if neg_poi is None:
            neg_poi = random.choice(all_pois)

        pos_dist = haversine_km(group_lat, group_lon, pos_poi["latitude"], pos_poi["longitude"])
        neg_dist = haversine_km(group_lat, group_lon, neg_poi["latitude"], neg_poi["longitude"])

        pos_dist = 999.0 if pos_dist is None else float(pos_dist)
        neg_dist = 999.0 if neg_dist is None else float(neg_dist)

        pos_pop = float(pos_poi.get("log_total_visits", 0.0))
        neg_pop = float(neg_poi.get("log_total_visits", 0.0))

        return {
            "group_emb": torch.tensor(group_emb, dtype=torch.float32),
            "pos_poi_emb": torch.tensor(pos_poi["embedding"], dtype=torch.float32),
            "neg_poi_emb": torch.tensor(neg_poi["embedding"], dtype=torch.float32),
            "pos_extra": torch.tensor([pos_dist, pos_pop], dtype=torch.float32),
            "neg_extra": torch.tensor([neg_dist, neg_pop], dtype=torch.float32),
            "group_id": gdoc.get("Group_ID"),
            "pos_venue_id": pos_venue_id,
            "neg_venue_id": neg_poi["venue_id"],
        }


def make_loaders(sampling_mode):
    print(f"\n[STEP 7] DataLoader 생성 시작 / mode={sampling_mode}")

    train_dataset = GroupPOIRankingDataset(train_docs, sampling_mode=sampling_mode)
    valid_dataset = GroupPOIRankingDataset(valid_docs, sampling_mode=sampling_mode)
    test_dataset = GroupPOIRankingDataset(test_docs, sampling_mode=sampling_mode)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[STEP 7] train batches = {len(train_loader)}")
    print(f"[STEP 7] valid batches = {len(valid_loader)}")
    print(f"[STEP 7] test batches  = {len(test_loader)}")

    return train_loader, valid_loader, test_loader


# =========================================================
# 10. 학습 / 검증
# =========================================================
def train_one_epoch(model, loader, optimizer, device, epoch_idx, exp_name):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        g = batch["group_emb"].to(device)
        pos_p = batch["pos_poi_emb"].to(device)
        neg_p = batch["neg_poi_emb"].to(device)
        pos_x = batch["pos_extra"].to(device)
        neg_x = batch["neg_extra"].to(device)

        pos_scores = model(g, pos_p, pos_x)
        neg_scores = model(g, neg_p, neg_x)

        loss = bpr_loss(pos_scores, neg_scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * g.size(0)

        if batch_idx % PRINT_EVERY_BATCH == 0 or batch_idx == len(loader):
            avg_so_far = total_loss / (batch_idx * loader.batch_size)
            print(
                f"[TRAIN][{exp_name}][Epoch {epoch_idx:02d}] "
                f"batch {batch_idx}/{len(loader)} "
                f"| batch_loss={loss.item():.4f} "
                f"| running_avg_loss={avg_so_far:.4f}"
            )

    epoch_loss = total_loss / len(loader.dataset)
    elapsed = time.time() - start_time
    print(f"[TRAIN][{exp_name}][Epoch {epoch_idx:02d}] 완료 | epoch_loss={epoch_loss:.4f} | {elapsed:.2f}초")
    return epoch_loss


@torch.no_grad()
def evaluate_pairwise(model, loader, device, epoch_idx, exp_name):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader, start=1):
        g = batch["group_emb"].to(device)
        pos_p = batch["pos_poi_emb"].to(device)
        neg_p = batch["neg_poi_emb"].to(device)
        pos_x = batch["pos_extra"].to(device)
        neg_x = batch["neg_extra"].to(device)

        pos_scores = model(g, pos_p, pos_x)
        neg_scores = model(g, neg_p, neg_x)

        loss = bpr_loss(pos_scores, neg_scores)
        total_loss += loss.item() * g.size(0)

        correct += (pos_scores > neg_scores).sum().item()
        total += g.size(0)

        if batch_idx % PRINT_EVERY_BATCH == 0 or batch_idx == len(loader):
            print(
                f"[VALID][{exp_name}][Epoch {epoch_idx:02d}] "
                f"batch {batch_idx}/{len(loader)} "
                f"| batch_loss={loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader.dataset)
    pair_acc = correct / total if total > 0 else 0.0
    elapsed = time.time() - start_time
    print(f"[VALID][{exp_name}][Epoch {epoch_idx:02d}] 완료 | valid_loss={avg_loss:.4f} | pair_acc={pair_acc:.4f} | {elapsed:.2f}초")
    return avg_loss, pair_acc


# =========================================================
# 11. 후보군 구성
# =========================================================
def build_candidate_pool(gdoc, mode="baseline", max_candidates=MAX_CANDIDATES):
    pos_venue_id = gdoc["VenueID"]
    cluster_id = gdoc.get("cluster_id_k")
    glat = gdoc.get("Latitude")
    glon = gdoc.get("Longitude")

    if cluster_id is not None and cluster_id in poi_by_cluster:
        base_pool = poi_by_cluster[cluster_id]
        base_coords = poi_cluster_coords[cluster_id]
    else:
        base_pool = all_pois
        base_coords = all_pois_coords

    if mode == "hard" and glat is not None and glon is not None:
        # Vectorized distance + argsort instead of a per-POI python loop with
        # haversine_km -- same scale issue as sample_negative_hard (JP's
        # clusters average ~3x more POIs than US's), and this path is also
        # hit by show_topk_recommendations's sample output at the very end.
        dists = haversine_km_vec(glat, glon, base_coords[:, 0], base_coords[:, 1])
        order = np.argsort(dists)
        top_n = min(max_candidates, len(base_pool))
        candidates = [base_pool[i] for i in order[:top_n]]
    else:
        candidates = base_pool.copy()
        if mode == "baseline":
            random.shuffle(candidates)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

    if pos_venue_id not in {x["venue_id"] for x in candidates}:
        if pos_venue_id in poi_by_venue:
            candidates.append(poi_by_venue[pos_venue_id])

    return candidates


# =========================================================
# 12. 후보 점수 계산
# =========================================================
@torch.no_grad()
def score_candidates(model, gdoc, candidate_pois, device):
    model.eval()

    g = torch.tensor(gdoc["group_embedding"], dtype=torch.float32).unsqueeze(0).to(device)
    glat = gdoc.get("Latitude")
    glon = gdoc.get("Longitude")

    poi_embs = []
    extras = []

    for poi in candidate_pois:
        poi_embs.append(poi["embedding"])

        dist = haversine_km(glat, glon, poi["latitude"], poi["longitude"])
        dist = 999.0 if dist is None else float(dist)
        pop = float(poi.get("log_total_visits", 0.0))

        extras.append([dist, pop])

    poi_embs = torch.tensor(np.stack(poi_embs), dtype=torch.float32).to(device)
    extras = torch.tensor(np.array(extras, dtype=np.float32), dtype=torch.float32).to(device)

    g_expand = g.expand(poi_embs.size(0), -1)
    scores = model(g_expand, poi_embs, extras).cpu().numpy()

    results = []
    for poi, score in zip(candidate_pois, scores):
        results.append({
            "venue_id": poi["venue_id"],
            "category": poi.get("category"),
            "score": float(score),
            "distance_km": haversine_km(glat, glon, poi["latitude"], poi["longitude"]),
            "total_visits": poi.get("total_visits", 0),
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results


# =========================================================
# 13. Top-K 평가
# =========================================================
@torch.no_grad()
def evaluate_topk(model, eval_docs, candidate_mode="baseline", top_k_list=None, max_candidates=MAX_CANDIDATES):
    if top_k_list is None:
        top_k_list = TOP_K_LIST

    print(f"\n[STEP 8] Top-K 평가 시작 / mode={candidate_mode} / samples={len(eval_docs)}")

    metrics = {
        k: {"precision": [], "recall": [], "f1": [], "ndcg": []}
        for k in top_k_list
    }

    start_time = time.time()

    for idx, gdoc in enumerate(eval_docs, start=1):
        gt_id = gdoc["VenueID"]
        candidates = build_candidate_pool(gdoc, mode=candidate_mode, max_candidates=max_candidates)
        ranked = score_candidates(model, gdoc, candidates, DEVICE)
        ranked_ids = [x["venue_id"] for x in ranked]

        for k in top_k_list:
            metrics[k]["precision"].append(precision_at_k(ranked_ids, gt_id, k))
            metrics[k]["recall"].append(recall_at_k(ranked_ids, gt_id, k))
            metrics[k]["f1"].append(f1_at_k(ranked_ids, gt_id, k))
            metrics[k]["ndcg"].append(ndcg_at_k(ranked_ids, gt_id, k))

        if idx % TOPK_PROGRESS_EVERY == 0 or idx == len(eval_docs):
            elapsed = time.time() - start_time
            print(f"[STEP 8] Top-K 평가 진행 중... {idx}/{len(eval_docs)} | {elapsed:.2f}초")

    summary = {}
    for k in top_k_list:
        summary[k] = {
            "Precision": float(np.mean(metrics[k]["precision"])) if metrics[k]["precision"] else 0.0,
            "Recall": float(np.mean(metrics[k]["recall"])) if metrics[k]["recall"] else 0.0,
            "F1-score": float(np.mean(metrics[k]["f1"])) if metrics[k]["f1"] else 0.0,
            "NDCG": float(np.mean(metrics[k]["ndcg"])) if metrics[k]["ndcg"] else 0.0,
        }

    print(f"[STEP 8] Top-K 평가 완료 / mode={candidate_mode}")
    print(f"[STEP 8] evaluated_groups = {len(eval_docs)}")
    return summary


# =========================================================
# 14. 결과 출력
# =========================================================
def print_metric_table(title, result_dict):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    for k in sorted(result_dict.keys()):
        vals = result_dict[k]
        print(
            f"K={k:3d} | "
            f"Recall={vals['Recall']:.4f} | "
            f"Precision={vals['Precision']:.4f} | "
            f"F1-score={vals['F1-score']:.4f} | "
            f"NDCG={vals['NDCG']:.4f}"
        )


def compare_results(title, baseline_result, hard_result):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for k in TOP_K_LIST:
        b = baseline_result[k]
        h = hard_result[k]
        print(f"\n[K={k}]")
        print(f"Recall    | baseline={b['Recall']:.4f} | hard={h['Recall']:.4f} | diff={h['Recall'] - b['Recall']:+.4f}")
        print(f"Precision | baseline={b['Precision']:.4f} | hard={h['Precision']:.4f} | diff={h['Precision'] - b['Precision']:+.4f}")
        print(f"F1-score  | baseline={b['F1-score']:.4f} | hard={h['F1-score']:.4f} | diff={h['F1-score'] - b['F1-score']:+.4f}")
        print(f"NDCG      | baseline={b['NDCG']:.4f} | hard={h['NDCG']:.4f} | diff={h['NDCG'] - b['NDCG']:+.4f}")


# =========================================================
# 15. 실험 실행
# =========================================================
def run_experiment(exp_name, sampling_mode):
    print("\n" + "#" * 100)
    print(f"[EXPERIMENT START] {exp_name}")
    print("#" * 100)
    print(f"[EXPERIMENT {exp_name}] train_groups: {len(train_docs)}")
    print(f"[EXPERIMENT {exp_name}] test_groups: {len(test_docs)}")
    print(f"[EXPERIMENT {exp_name}] evaluated_groups: {len(test_docs)}")

    train_loader, valid_loader, test_loader = make_loaders(sampling_mode)

    model = GroupPOIScorerMLP(
        emb_dim=EMB_DIM,
        extra_dim=2,
        hidden_dims=(256, 128),
        dropout=0.1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_valid_loss = float("inf")
    save_path = str(THIS_DIR / f"best_{exp_name}_jp.pt")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[EXPERIMENT {exp_name}] Epoch {epoch:02d}/{EPOCHS} 시작")
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch, exp_name)
        valid_loss, valid_pair_acc = evaluate_pairwise(model, valid_loader, DEVICE, epoch, exp_name)

        print(
            f"[EXPERIMENT {exp_name}] Epoch {epoch:02d} 결과 | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"valid_pair_acc={valid_pair_acc:.4f}"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"[EXPERIMENT {exp_name}] best model 저장: {save_path}")

    print(f"\n[EXPERIMENT {exp_name}] best model 로드")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()

    print(f"\n[EXPERIMENT {exp_name}] VALID Top-K 평가")
    valid_topk = evaluate_topk(
        model=model,
        eval_docs=valid_docs,
        candidate_mode=sampling_mode,
        top_k_list=TOP_K_LIST,
        max_candidates=MAX_CANDIDATES
    )

    print(f"\n[EXPERIMENT {exp_name}] TEST Top-K 평가")
    test_topk = evaluate_topk(
        model=model,
        eval_docs=test_docs,
        candidate_mode=sampling_mode,
        top_k_list=TOP_K_LIST,
        max_candidates=MAX_CANDIDATES
    )

    print(f"\n[EXPERIMENT END] {exp_name}")
    return model, valid_topk, test_topk


# =========================================================
# 16. 추천 결과 보기
# =========================================================
@torch.no_grad()
def show_topk_recommendations(model, gdoc, candidate_mode="baseline", topk=10, max_candidates=MAX_CANDIDATES):
    print(f"\n[STEP 9] 추천 결과 계산 시작 / Group_ID={gdoc.get('Group_ID')} / mode={candidate_mode}")
    candidates = build_candidate_pool(gdoc, mode=candidate_mode, max_candidates=max_candidates)
    ranked = score_candidates(model, gdoc, candidates, DEVICE)

    print("\n" + "=" * 100)
    print(
        f"Group_ID={gdoc.get('Group_ID')} | "
        f"GT VenueID={gdoc.get('VenueID')} | "
        f"mode={candidate_mode}"
    )
    print("=" * 100)

    for i, row in enumerate(ranked[:topk], start=1):
        print(
            f"{i:2d}. "
            f"venue_id={row['venue_id']} | "
            f"score={row['score']:.6f} | "
            f"category={row['category']} | "
            f"distance_km={row['distance_km']} | "
            f"total_visits={row['total_visits']}"
        )

    gt_id = gdoc["VenueID"]
    ranked_ids = [x["venue_id"] for x in ranked]
    if gt_id in ranked_ids:
        gt_rank = ranked_ids.index(gt_id) + 1
        print(f"\nGT venue rank = {gt_rank}")
    else:
        print("\nGT venue not found in candidate pool")

    print(f"[STEP 9] 추천 결과 계산 완료 / Group_ID={gdoc.get('Group_ID')} / mode={candidate_mode}")
