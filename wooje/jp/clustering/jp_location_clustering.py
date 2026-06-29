"""Stage 4/5 (unified): silhouette-optimal location clustering for Tokyo.

The US pipeline had two separate location-clustering tracks that happened to
converge on the same k=5 by coincidence:
  - a "fixed k=5" MiniBatchKMeans (`3. loc_cluster_info_k5`)
  - a silhouette-swept "best k" MiniBatchKMeans (`3. loc_kmeans_bestk_meta`),
    which for the US data also picked k=5 (silhouette=0.5548)

Per the user's explicit request, JP does ONE silhouette-validated sweep and
uses that single chosen k for both purposes -- there is no separate "fixed 5"
assumption. The same centroids/k are written into both sets of collections
so downstream scripts (group-checkin-cluster-id-k5, k-means_embedding) that
read either naming scheme keep working unmodified.
"""
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient, UpdateOne
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"
CHECKIN_COL = "2. JP_checkin_v2"
K_MIN, K_MAX = 2, 50
SAMPLE_SIZE = 20000
SEED = 42


def main():
    t0 = time.time()
    client = MongoClient(HOST, PORT, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    col = db[CHECKIN_COL]

    print("Loading checkin coordinates ...")
    docs = list(col.find({}, {"_id": 1, "latitude": 1, "longitude": 1}))
    ids = [d["_id"] for d in docs]
    coords = np.array([[d["latitude"], d["longitude"]] for d in docs], dtype=np.float64)
    n_points = len(coords)
    print(f"  n_points={n_points:,}")

    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(n_points, size=min(SAMPLE_SIZE, n_points), replace=False)
    sample_coords = coords[sample_idx]

    print(f"\nSilhouette sweep k={K_MIN}..{K_MAX} ...")
    silhouette_results = []
    best_k, best_score = None, -1.0
    for k in range(K_MIN, K_MAX + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10, batch_size=4096)
        km.fit(sample_coords)
        sample_labels = km.predict(sample_coords)
        score = silhouette_score(sample_coords, sample_labels)
        silhouette_results.append({"k": k, "silhouette": float(score)})
        if score > best_score:
            best_k, best_score = k, score
    print(f"  -> best k={best_k} (silhouette={best_score:.4f})")
    top5 = sorted(silhouette_results, key=lambda r: -r["silhouette"])[:5]
    print(f"  top 5: {top5}")

    print(f"\nRefitting MiniBatchKMeans(k={best_k}) on the full {n_points:,} points ...")
    final_model = MiniBatchKMeans(n_clusters=best_k, random_state=SEED, n_init=10, batch_size=4096)
    labels = final_model.fit_predict(coords)
    centroids = final_model.cluster_centers_

    print("\nWriting cluster info + meta collections ...")
    now = pd.Timestamp.utcnow().to_pydatetime()

    cluster_docs = [
        {
            "_id": f"k_{best_k}_cluster_{i}",
            "cluster_id": i,
            "k": best_k,
            "latitude": float(centroids[i][0]),
            "longitude": float(centroids[i][1]),
            "assign_field": "loc_cluster_id_k5",
            "source_collection": CHECKIN_COL,
            "model": "MiniBatchKMeans",
            "updated_at": now,
        }
        for i in range(best_k)
    ]
    db["3. loc_cluster_info_k5"].delete_many({})
    db["3. loc_cluster_info_k5"].insert_many(cluster_docs)

    centroid_list = centroids.tolist()
    db["3. loc_cluster_info_k5_meta"].delete_many({})
    db["3. loc_cluster_info_k5_meta"].insert_one({
        "_id": f"k_{best_k}_summary",
        "k": best_k,
        "n_points": n_points,
        "assign_field": "loc_cluster_id_k5",
        "cluster_info_collection": "3. loc_cluster_info_k5",
        "centroids": centroid_list,
        "updated_at": now,
    })

    db["3. loc_kmeans_bestk_meta"].delete_many({})
    db["3. loc_kmeans_bestk_meta"].insert_one({
        "_id": f"best_k_{best_k}",
        "k": best_k,
        "silhouette": float(best_score),
        "n_points": n_points,
        "sample_size": min(SAMPLE_SIZE, n_points),
        "assign_field": "loc_cluster_id_bestk",
        "freq_collection": "3. user_cluster_frequency_bestk",
        "centroids": centroid_list,
        "silhouette_results": silhouette_results,
        "updated_at": now,
    })
    print(f"  3. loc_cluster_info_k5: {best_k} docs")
    print("  3. loc_cluster_info_k5_meta / 3. loc_kmeans_bestk_meta written")

    print("\nAssigning loc_cluster_id_k5 + loc_cluster_id_bestk onto checkin_v2 ...")
    BATCH = 20000
    ops = []
    for _id, label in zip(ids, labels):
        ops.append(UpdateOne(
            {"_id": _id},
            {"$set": {"loc_cluster_id_k5": int(label), "loc_cluster_id_bestk": int(label)}},
        ))
        if len(ops) >= BATCH:
            col.bulk_write(ops, ordered=False)
            ops = []
    if ops:
        col.bulk_write(ops, ordered=False)
    print("  done.")

    print("\nBuilding 3. user_cluster_frequency_bestk ...")
    label_by_id = dict(zip(ids, labels))
    user_docs = list(col.find({}, {"_id": 1, "user_id": 1}))
    freq = {}
    for d in user_docs:
        uid = d["user_id"]
        cid = label_by_id[d["_id"]]
        freq.setdefault(uid, {}).setdefault(cid, 0)
        freq[uid][cid] += 1

    db["3. user_cluster_frequency_bestk"].delete_many({})
    freq_docs = []
    for uid, cdict in freq.items():
        total = sum(cdict.values())
        freq_docs.append({
            "_id": uid,
            "user_id": uid,
            "total_checkins": total,
            "freq": {str(k): v for k, v in cdict.items()},
            "k": best_k,
            "silhouette": float(best_score),
            "model": "MiniBatchKMeans",
            "assignment_field": "loc_cluster_id_bestk",
            "updated_at": now,
        })
    if freq_docs:
        db["3. user_cluster_frequency_bestk"].insert_many(freq_docs)
    print(f"  3. user_cluster_frequency_bestk: {len(freq_docs):,} docs")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
