"""Stage 6: per-user cluster-visit embedding -> "4. user_cluster_embedding_bestk".

Direct DB_NAME/CHECKIN_COL port of k-means_embedding.ipynb -- logic
unchanged (confirmed via full read of the original notebook), including its
independent re-aggregation of cluster frequency straight from checkin_v2
(it does not reuse "3. user_cluster_frequency_bestk").
"""
from datetime import datetime, timezone

import numpy as np
from pymongo import MongoClient, UpdateOne
from tqdm.auto import tqdm

HOST = "10.255.68.40"
PORT = 27017
DB_NAME = "ejoow2"

CHECKIN_COL = "2. JP_checkin_v2"
CLUSTER_INFO_COL = "3. loc_cluster_centroids_bestk"
META_COL = "3. loc_kmeans_bestk_meta"
OUT_COL = "4. user_cluster_embedding_bestk"
ASSIGN_FIELD = "loc_cluster_id_bestk"
NORMALIZE = "logprob"
BULK_SIZE = 5000

client = MongoClient(host=HOST, port=PORT)
db = client[DB_NAME]
checkin_col = db[CHECKIN_COL]
cluster_info_col = db[CLUSTER_INFO_COL]
meta_col = db[META_COL]
out_col = db[OUT_COL]

print("db:", DB_NAME)
print("checkin input:", CHECKIN_COL)

meta_doc = meta_col.find_one(sort=[("updated_at", -1)])
if meta_doc is None:
    raise ValueError("bestk meta document missing -- run jp_location_clustering.py first.")

K = int(meta_doc["k"])
cluster_docs = list(
    cluster_info_col.find({"k": K}, {"_id": 0, "cluster_id": 1, "latitude": 1, "longitude": 1}).sort("cluster_id", 1)
)
if not cluster_docs:
    meta_centroids = meta_doc.get("centroids", [])
    cluster_docs = [
        {"cluster_id": int(cid), "latitude": float(lat), "longitude": float(lon)}
        for cid, (lat, lon) in enumerate(meta_centroids)
    ]
    print("centroid collection is empty; using centroids from meta document")

cluster_ids = [int(d["cluster_id"]) for d in cluster_docs]
print("cluster_ids:", cluster_ids, "best_k:", K)
if len(cluster_docs) != K:
    raise ValueError(f"expected {K} clusters, found {len(cluster_docs)}")

pipeline = [
    {"$match": {ASSIGN_FIELD: {"$exists": True}}},
    {"$group": {"_id": {"user_id": "$user_id", "cluster_id": f"${ASSIGN_FIELD}"}, "count": {"$sum": 1}}},
    {"$group": {
        "_id": "$_id.user_id",
        "total_checkins": {"$sum": "$count"},
        "pairs": {"$push": {"k": {"$toString": "$_id.cluster_id"}, "v": "$count"}},
    }},
    {"$project": {"_id": 0, "user_id": "$_id", "total_checkins": 1, "frequency": {"$arrayToObject": "$pairs"}}},
]
user_cluster_rows = list(checkin_col.aggregate(pipeline, allowDiskUse=True))
print("users:", len(user_cluster_rows))


def build_cluster_embedding(freq_dict, k, normalize="prob"):
    vec = np.zeros(k, dtype=np.float32)
    for cluster_id_str, count in freq_dict.items():
        cid = int(cluster_id_str)
        if 0 <= cid < k:
            vec[cid] = float(count)
    if normalize == "prob":
        total = vec.sum()
        if total > 0:
            vec /= total
    elif normalize == "logprob":
        vec = np.log1p(vec)
        total = vec.sum()
        if total > 0:
            vec /= total
    return vec


out_col.create_index("user_id", unique=True)

ops = []
written = 0
now = datetime.now(timezone.utc)
for row in tqdm(user_cluster_rows, desc="Writing user cluster embeddings"):
    user_id = row["user_id"]
    freq = row.get("frequency", {})
    vec = build_cluster_embedding(freq, k=K, normalize=NORMALIZE)
    out_doc = {
        "_id": user_id,
        "user_id": user_id,
        "k": int(K),
        "assign_field": ASSIGN_FIELD,
        "source_collection": CHECKIN_COL,
        "cluster_info_collection": CLUSTER_INFO_COL,
        "normalize": NORMALIZE,
        "total_checkins": int(row.get("total_checkins", 0)),
        "nonzero": int((vec > 0).sum()),
        "frequency": freq,
        "embedding": vec.tolist(),
        "updated_at": now,
    }
    ops.append(UpdateOne({"_id": user_id}, {"$set": out_doc}, upsert=True))
    if len(ops) >= BULK_SIZE:
        out_col.bulk_write(ops, ordered=False)
        written += len(ops)
        ops.clear()
if ops:
    out_col.bulk_write(ops, ordered=False)
    written += len(ops)

print("written:", written)
print("out count:", out_col.estimated_document_count())
