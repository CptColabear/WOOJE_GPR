"""Adapts group-checkin-cluster-id-k5.ipynb for JP: assigns cluster_id_k onto
"0. JP_group_checkin_labeled" via nearest-haversine-centroid against
"3. loc_cluster_info_k5" (the unified silhouette-optimal centroids from
jp_location_clustering.py). Logic unchanged from the original, confirmed via
full read of the source notebook -- only DB_NAME/collection name differ.
"""
import numpy as np
from pymongo import MongoClient, UpdateOne
from tqdm.auto import tqdm

HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"
GROUP_CHECKIN_COL = "0. JP_group_checkin_labeled"
CLUSTER_INFO_COL = "3. loc_cluster_info_k5"
ASSIGN_FIELD = "cluster_id_k"
ASSIGN_BATCH = 20000

client = MongoClient(host=HOST, port=PORT)
db = client[DB_NAME]
group_checkin_col = db[GROUP_CHECKIN_COL]
cluster_info_col = db[CLUSTER_INFO_COL]

query = {"Latitude": {"$ne": None}, "Longitude": {"$ne": None}}
n_group_points = group_checkin_col.count_documents(query)
print("group checkin input:", GROUP_CHECKIN_COL, "count:", n_group_points)

cluster_docs = list(cluster_info_col.find({}, {"_id": 1, "cluster_id": 1, "latitude": 1, "longitude": 1}).sort("cluster_id", 1))
print("clusters:", cluster_docs)

cluster_ids = np.asarray([d["cluster_id"] for d in cluster_docs], dtype=np.int32)
cluster_coords = np.radians(np.asarray([[d["latitude"], d["longitude"]] for d in cluster_docs], dtype=np.float64))
EARTH_RADIUS_KM = 6371.0088


def nearest_cluster_ids(batch_xy):
    coords = np.radians(np.asarray(batch_xy, dtype=np.float64))
    lat1, lon1 = coords[:, 0][:, None], coords[:, 1][:, None]
    lat2, lon2 = cluster_coords[:, 0][None, :], cluster_coords[:, 1][None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    distances = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    nearest_idx = np.argmin(distances, axis=1)
    return cluster_ids[nearest_idx], distances[np.arange(len(nearest_idx)), nearest_idx]


cursor = group_checkin_col.find(query, {"_id": 1, "Latitude": 1, "Longitude": 1}, no_cursor_timeout=True)
ops, buf_ids, buf_xy = [], [], []
pbar = tqdm(total=n_group_points, desc=f"Assigning {ASSIGN_FIELD}")
try:
    for doc in cursor:
        buf_ids.append(doc["_id"])
        buf_xy.append([doc["Latitude"], doc["Longitude"]])
        if len(buf_ids) >= ASSIGN_BATCH:
            nearest_ids, _ = nearest_cluster_ids(buf_xy)
            for _id, cid in zip(buf_ids, nearest_ids):
                ops.append(UpdateOne({"_id": _id}, {"$set": {ASSIGN_FIELD: int(cid)}}))
            group_checkin_col.bulk_write(ops, ordered=False)
            pbar.update(len(buf_ids))
            ops.clear(); buf_ids.clear(); buf_xy.clear()
    if buf_ids:
        nearest_ids, _ = nearest_cluster_ids(buf_xy)
        for _id, cid in zip(buf_ids, nearest_ids):
            ops.append(UpdateOne({"_id": _id}, {"$set": {ASSIGN_FIELD: int(cid)}}))
        group_checkin_col.bulk_write(ops, ordered=False)
        pbar.update(len(buf_ids))
finally:
    pbar.close()
    cursor.close()

print("done:", ASSIGN_FIELD, "updated in", GROUP_CHECKIN_COL)
print("distinct values:", sorted(group_checkin_col.distinct(ASSIGN_FIELD)))
