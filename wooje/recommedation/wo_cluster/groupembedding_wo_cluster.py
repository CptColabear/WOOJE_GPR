#!/usr/bin/env python
# coding: utf-8

# In[20]:


from pymongo import MongoClient, UpdateOne
from pymongo.errors import OperationFailure
from tqdm.auto import tqdm
from datetime import UTC, datetime
from itertools import islice
import numpy as np
import math


# In[21]:


HOST = "10.255.68.40"
PORT = 27017
DB_NAME = "ejoow"

client = MongoClient(HOST, PORT)
db = client[DB_NAME]

group_checkin_col = db["0. US_group_checkin_labeled"]
user_tl_emb_col = db["user_tl_pattern_embeddings_attn_v1"]
user_cluster_vec_col = db["4. user_cluster_embedding_bestk"]
group_emb_col = db["9. group_embeddings_wo_cluster_v2"]

print("group_checkin count:", group_checkin_col.estimated_document_count())
print("user_tl_emb count:", user_tl_emb_col.estimated_document_count())
print("user_cluster_vec count:", user_cluster_vec_col.estimated_document_count())


# In[22]:


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm

def softmax(x):
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)




# In[23]:


def batched(iterable, batch_size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def load_user_tl_embedding_cache(user_tl_keys, query_batch_size=500):
    cache = {}
    key_list = list(user_tl_keys)

    for key_batch in batched(key_list, query_batch_size):
        query = {
            "$or": [
                {"user_id": user_id, "TL": tl}
                for user_id, tl in key_batch
            ]
        }
        cursor = user_tl_emb_col.find(
            query,
            {"_id": 0, "user_id": 1, "TL": 1, "pattern_embedding": 1}
        )
        for doc in cursor:
            cache[(str(doc["user_id"]), int(doc["TL"]))] = np.array(
                doc["pattern_embedding"],
                dtype=np.float32,
            )

    return cache


# In[24]:


def load_user_cluster_score_cache(user_ids):
    cache = {}
    user_id_list = list(user_ids)
    if not user_id_list:
        return cache

    cursor = user_cluster_vec_col.find(
        {"user_id": {"$in": user_id_list}},
        {"_id": 0, "user_id": 1, "embedding": 1}
    )
    for doc in cursor:
        vec = doc.get("embedding")
        if vec is None:
            continue
        cache[str(doc["user_id"])] = np.array(vec, dtype=np.float32)

    return cache


# In[25]:


# def build_group_embedding(users, TL, cluster_id_k, tl_embedding_cache, cluster_score_cache, beta=10.0):
#     valid_user_ids = []
#     user_embeddings = []
#     raw_scores = []
#     cid = int(cluster_id_k)

#     for user_id in users:
#         user_id = str(user_id)
#         emb = tl_embedding_cache.get((user_id, int(TL)))
#         score_vec = cluster_score_cache.get(user_id)

#         if emb is None:
#             continue
#         if score_vec is None:
#             continue
#         if cid < 0 or cid >= len(score_vec):
#             continue

#         valid_user_ids.append(user_id)
#         user_embeddings.append(emb)
#         raw_scores.append(float(score_vec[cid]))

#     if len(valid_user_ids) == 0:
#         return None

#     user_embeddings = np.stack(user_embeddings, axis=0)  # (n, 64)
#     raw_scores = np.array(raw_scores, dtype=np.float32)

#     # attention weights
#     attn_weights = softmax(beta * raw_scores)  # (n,)

#     # weighted sum
#     group_emb = np.sum(user_embeddings * attn_weights[:, None], axis=0)
#     group_emb = l2_normalize(group_emb)

#     return {
#         "valid_user_ids": valid_user_ids,
#         "raw_scores": raw_scores.tolist(),
#         "attention_weights": attn_weights.tolist(),
#         "group_embedding": group_emb.tolist(),
#         "n_valid_users": len(valid_user_ids),
#     }

def build_group_embedding(users, TL, tl_embedding_cache):
    valid_user_ids = []
    user_embeddings = []

    for user_id in users:
        user_id = str(user_id)
        emb = tl_embedding_cache.get((user_id, int(TL)))

        if emb is None:
            continue

        valid_user_ids.append(user_id)
        user_embeddings.append(emb)

    if len(valid_user_ids) == 0:
        return None

    user_embeddings = np.stack(user_embeddings, axis=0)

    # 친숙도 점수를 사용하지 않고 사용자 TL 임베딩을 단순 합산
    # group_emb = np.sum(user_embeddings, axis=0)
    group_emb = np.mean(user_embeddings, axis=0)


    # 최종 그룹 임베딩 정규화
    group_emb = l2_normalize(group_emb)

    return {
        "valid_user_ids": valid_user_ids,
        "group_embedding": group_emb.tolist(),
        "n_valid_users": len(valid_user_ids),
    }


# In[26]:


sample_group = group_checkin_col.find_one({})
sample_group


# In[27]:


users = sample_group["Users"]
TL = int(sample_group["TL"])
cluster_id_k = int(sample_group["cluster_id_k"])

sample_user_tl_keys = {(str(user_id), TL) for user_id in users}
sample_tl_embedding_cache = load_user_tl_embedding_cache(sample_user_tl_keys)

# result = build_group_embedding(
#     users,
#     TL,
#     cluster_id_k,
#     sample_tl_embedding_cache,
#     sample_cluster_score_cache,
#     beta=10.0,
# )

result = build_group_embedding(
    users,
    TL,
    sample_tl_embedding_cache,
)
result


# In[28]:


def ensure_index(collection, keys, **kwargs):
    try:
        return collection.create_index(keys, **kwargs)
    except OperationFailure as exc:
        if exc.code == 86:
            print(f"skip existing conflicting index on {collection.name}: {keys}")
            return None
        raise

ensure_index(user_tl_emb_col, [("user_id", 1), ("TL", 1)], unique=True)
ensure_index(user_cluster_vec_col, "user_id")
ensure_index(group_emb_col, "source_group_checkin_id", unique=True)
ensure_index(group_emb_col, [("Group_ID", 1), ("TL", 1)])


# In[29]:


GROUP_BATCH_SIZE = 1000
BULK_WRITE_SIZE = 1000

projection = {
    "_id": 1,
    "Group_ID": 1,
    "Users": 1,
    "TL": 1,
    "cluster_id_k": 1,
    "Latitude": 1,
    "Longitude": 1,
    "VenueID": 1,
    "VenueCategoryname": 1,
    "users": 1,
}

written = 0
processed = 0
skipped = 0
cursor = group_checkin_col.find({}, projection, batch_size=GROUP_BATCH_SIZE)
total_docs = group_checkin_col.estimated_document_count()

for doc_batch in tqdm(
    batched(cursor, GROUP_BATCH_SIZE),
    total=math.ceil(total_docs / GROUP_BATCH_SIZE),
    desc="Building group embeddings (batch)",
):
    user_tl_keys = {
        (str(user_id), int(doc["TL"]))
        for doc in doc_batch
        for user_id in doc["Users"]
    }

    tl_embedding_cache = load_user_tl_embedding_cache(user_tl_keys)
    ops = []

    for doc in doc_batch:
        processed += 1

        source_id = str(doc["_id"])
        group_id = int(doc["Group_ID"])
        TL = int(doc["TL"])
        cluster_id_k = int(doc["cluster_id_k"])
        users = doc["Users"]

        result = build_group_embedding(
            users,
            TL,
            tl_embedding_cache,
        )

        if result is None:
            skipped += 1
            continue

        out_doc = {
            "_id": source_id,
            "source_group_checkin_id": source_id,
            "Group_ID": group_id,
            "TL": TL,
            "cluster_id_k": cluster_id_k,
            "Latitude": float(doc["Latitude"]),
            "Longitude": float(doc["Longitude"]),
            "VenueID": doc.get("VenueID"),
            "VenueCategoryname": doc.get("VenueCategoryname"),
            "group_size": int(doc.get("users", len(users))),
            "input_user_ids": [str(u) for u in users],
            "valid_user_ids": result["valid_user_ids"],
            "n_valid_users": result["n_valid_users"],
            "group_embedding": result["group_embedding"],
            "dim": len(result["group_embedding"]),
            "aggregation": "sum_l2_normalized",
            "value_source": "user_tl_pattern_embeddings_attn_v1",
            "updated_at": datetime.now(UTC),
        }

        ops.append(UpdateOne(
            {"_id": source_id},
            {"$set": out_doc},
            upsert=True,
        ))

        if len(ops) >= BULK_WRITE_SIZE:
            group_emb_col.bulk_write(ops, ordered=False)
            written += len(ops)
            ops = []

    if ops:
        group_emb_col.bulk_write(ops, ordered=False)
        written += len(ops)

    print(
        f"processed={processed:,} written={written:,} skipped={skipped:,} "
        f"cached_tl={len(tl_embedding_cache):,}"
    )


# In[30]:


sample_out = group_emb_col.find_one({})


# In[31]:


if sample_out is None:
    print("No group embeddings found in collection.")
else:
    print("Group_ID:", sample_out.get("Group_ID"))
    print("TL:", sample_out.get("TL"))
    print("cluster_id_k:", sample_out.get("cluster_id_k"))
    print("valid_user_ids:", sample_out.get("valid_user_ids", []))
    print("n_valid_users:", sample_out.get("n_valid_users", 0))
    print("embedding dim:", len(sample_out.get("group_embedding", [])))
    if "raw_scores" in sample_out:
        print("raw_scores:", sample_out["raw_scores"])
    if "attention_weights" in sample_out:
        print("attention_weights:", sample_out["attention_weights"])
    print("aggregation:", sample_out.get("aggregation"))
