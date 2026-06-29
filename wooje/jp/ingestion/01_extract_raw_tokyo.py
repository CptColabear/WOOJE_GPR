"""Stage 1: filter raw global Foursquare dump to Tokyo, load into ejoow2.

Two clustering passes are used for two different purposes:
  1. A silhouette-score k-means sweep over JP POI coordinates -- this is a
     diagnostic confirming Tokyo is a statistically distinct, well-separated
     urban center (reported for transparency, not used for the boundary
     itself: k-means assigns every point to its nearest centroid with no
     outlier rejection, so its cluster boundary balloons to include far-flung
     sparse points -- confirmed empirically: the k=6 best-silhouette "Tokyo"
     cluster centroid was correct (35.73, 139.55) but its bbox spanned lat
     24-37 / lon 137-154, i.e. most of Japan including Okinawa).
  2. DBSCAN (haversine, eps=2km, min_samples=30) -- density-based, rejects
     sparse far-away points as noise instead of forcing them into the nearest
     centroid, so it gives a tight, accurate Tokyo metro boundary. Parameters
     were chosen by sweeping eps in {2,3,5,8}km / min_samples in {30,50,100}
     and checking that the resulting top-5 clusters' centroids land on real
     Japanese metro areas (Tokyo/Osaka/Nagoya/Fukuoka/Sapporo all separated
     cleanly at eps=2km, min_samples=30 -- see scratchpad/tune_dbscan.py).

Checkins have no country field, so they can only be scoped by joining on
venue_id against the Tokyo-filtered POIs.
"""
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.metrics import silhouette_score

DATA_DIR = "/home/gpuadmin/dev/group/dataset/foursquare_global/dataset_WWW2019"
HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"
COUNTRY = "JP"
K_CANDIDATES = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
SILHOUETTE_SAMPLE = 20000
SEED = 42
EARTH_RADIUS_KM = 6371.0088
DBSCAN_EPS_KM = 2.0
DBSCAN_MIN_SAMPLES = 30

TOKYO_REF_LAT = (35.4, 36.0)
TOKYO_REF_LON = (139.3, 140.2)


def run_silhouette_diagnostic(coords):
    print("\n[2/5] Silhouette-score KMeans sweep on JP POI coordinates (diagnostic) ...")
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(coords), size=min(SILHOUETTE_SAMPLE, len(coords)), replace=False)
    sample_coords = coords[sample_idx]

    best_k, best_score = None, -1.0
    sweep_results = []
    for k in K_CANDIDATES:
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10, batch_size=4096)
        labels = km.fit_predict(coords)
        sample_labels = km.predict(sample_coords)
        score = silhouette_score(sample_coords, sample_labels)
        sizes = np.bincount(labels, minlength=k).tolist()
        sweep_results.append({"k": k, "silhouette": float(score), "sizes": sizes})
        print(f"  k={k:>2}  silhouette={score:.4f}  sizes={sizes}")
        if score > best_score:
            best_k, best_score = k, score

    print(f"  -> best k by silhouette score: {best_k} (silhouette={best_score:.4f})")
    return sweep_results, best_k, best_score


def run_dbscan_boundary(pois_jp):
    print("\n[3/5] DBSCAN density clustering to find the precise Tokyo boundary ...")
    coords_rad = np.radians(pois_jp[["latitude", "longitude"]].to_numpy())
    db = DBSCAN(
        eps=DBSCAN_EPS_KM / EARTH_RADIUS_KM,
        min_samples=DBSCAN_MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1,
    )
    labels = db.fit_predict(coords_rad)
    n_noise = int((labels == -1).sum())
    uniq, counts = np.unique(labels[labels != -1], return_counts=True)
    order = np.argsort(-counts)
    print(f"  eps={DBSCAN_EPS_KM}km min_samples={DBSCAN_MIN_SAMPLES}: "
          f"n_clusters={len(uniq)} noise={n_noise:,} ({n_noise / len(labels):.1%})")

    top5 = []
    for rank in order[:5]:
        cid = int(uniq[rank])
        mask = labels == cid
        lat = pois_jp["latitude"].to_numpy()[mask]
        lon = pois_jp["longitude"].to_numpy()[mask]
        top5.append({"cluster_id": cid, "size": int(mask.sum()),
                     "centroid": (float(lat.mean()), float(lon.mean()))})
        print(f"  cluster {cid}: size={mask.sum():,} centroid=({lat.mean():.4f},{lon.mean():.4f}) "
              f"bbox lat=({lat.min():.4f},{lat.max():.4f}) lon=({lon.min():.4f},{lon.max():.4f})")

    tokyo_candidates = [
        c for c in top5
        if TOKYO_REF_LAT[0] <= c["centroid"][0] <= TOKYO_REF_LAT[1]
        and TOKYO_REF_LON[0] <= c["centroid"][1] <= TOKYO_REF_LON[1]
    ]
    if not tokyo_candidates:
        raise RuntimeError("No DBSCAN cluster centroid fell inside the Tokyo reference box. Inspect top5 above.")
    tokyo_cluster = max(tokyo_candidates, key=lambda c: c["size"])
    tokyo_mask = labels == tokyo_cluster["cluster_id"]
    print(f"  -> cluster {tokyo_cluster['cluster_id']} selected as Tokyo "
          f"(centroid={tokyo_cluster['centroid']}, size={tokyo_cluster['size']:,})")
    return tokyo_mask, top5, tokyo_cluster


def main():
    t0 = time.time()
    print("=" * 80)
    print("[1/5] Loading raw_POIs.txt and filtering to country == JP ...")
    pois = pd.read_csv(
        f"{DATA_DIR}/raw_POIs.txt", sep="\t", header=None,
        names=["venue_id", "latitude", "longitude", "category", "country"],
        dtype={"venue_id": str, "category": str, "country": str},
    )
    pois_jp = pois[pois["country"] == COUNTRY].copy().reset_index(drop=True)
    del pois
    print(f"  JP POIs: {len(pois_jp):,}")

    coords = pois_jp[["latitude", "longitude"]].to_numpy()
    sweep_results, best_k, best_silhouette = run_silhouette_diagnostic(coords)
    tokyo_mask, dbscan_top5, tokyo_cluster = run_dbscan_boundary(pois_jp)

    pois_tokyo = pois_jp[tokyo_mask].copy()
    lat_min, lat_max = pois_tokyo["latitude"].min(), pois_tokyo["latitude"].max()
    lon_min, lon_max = pois_tokyo["longitude"].min(), pois_tokyo["longitude"].max()
    print(f"\n  Tokyo POIs: {len(pois_tokyo):,}  bbox lat=({lat_min:.4f},{lat_max:.4f}) lon=({lon_min:.4f},{lon_max:.4f})")

    pois_tokyo["category_id"] = pois_tokyo["category"].astype("category").cat.codes
    venue_ids = set(pois_tokyo["venue_id"])

    print("\n[4/5] Streaming raw_Checkins_anonymized.txt, filtering to Tokyo venues ...")
    chunks = []
    total_rows = 0
    reader = pd.read_csv(
        f"{DATA_DIR}/raw_Checkins_anonymized.txt", sep="\t", header=None,
        names=["user_id", "venue_id", "utc_time", "offset_min"],
        dtype={"user_id": str, "venue_id": str, "offset_min": "int32"},
        chunksize=5_000_000,
    )
    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        filtered = chunk[chunk["venue_id"].isin(venue_ids)]
        if len(filtered):
            chunks.append(filtered)
        print(f"  chunk {i + 1}: scanned {total_rows:,} rows so far, kept {sum(len(c) for c in chunks):,}")

    checkins_tokyo = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["user_id", "venue_id", "utc_time", "offset_min"]
    )
    print(f"  Tokyo checkins: {len(checkins_tokyo):,}")
    print(f"  distinct users: {checkins_tokyo['user_id'].nunique():,}")
    print(f"  distinct venues touched: {checkins_tokyo['venue_id'].nunique():,}")

    print("\n[5/5] Writing to MongoDB ejoow2 ...")
    client = MongoClient(HOST, PORT, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    db["1. JP_pois"].delete_many({})
    poi_docs = pois_tokyo.to_dict("records")
    if poi_docs:
        db["1. JP_pois"].insert_many(poi_docs)
    print(f"  1. JP_pois: {db['1. JP_pois'].estimated_document_count():,} docs")

    db["1. JP_checkin"].delete_many({})
    BATCH = 200_000
    records = checkins_tokyo.to_dict("records")
    for i in range(0, len(records), BATCH):
        db["1. JP_checkin"].insert_many(records[i : i + BATCH])
    print(f"  1. JP_checkin: {db['1. JP_checkin'].estimated_document_count():,} docs")

    meta = {
        "_id": "tokyo_geo_scope",
        "country": COUNTRY,
        "city": "Tokyo",
        "kmeans_silhouette_diagnostic": {
            "k_candidates": K_CANDIDATES,
            "sweep": sweep_results,
            "best_k": int(best_k),
            "best_silhouette": float(best_silhouette),
            "note": "diagnostic only -- confirms Tokyo is a statistically distinct urban cluster; "
                    "NOT used as the filtering boundary (k-means has no outlier rejection)",
        },
        "dbscan_boundary": {
            "eps_km": DBSCAN_EPS_KM,
            "min_samples": DBSCAN_MIN_SAMPLES,
            "top5_clusters": dbscan_top5,
            "selected_cluster_id": tokyo_cluster["cluster_id"],
            "selected_centroid": tokyo_cluster["centroid"],
        },
        "bbox": {
            "lat_min": float(lat_min), "lat_max": float(lat_max),
            "lon_min": float(lon_min), "lon_max": float(lon_max),
        },
        "n_pois": int(len(pois_tokyo)),
        "n_checkins": int(len(checkins_tokyo)),
        "n_users": int(checkins_tokyo["user_id"].nunique()),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    db["0. JP_geo_scope_meta"].replace_one({"_id": "tokyo_geo_scope"}, meta, upsert=True)
    print("  wrote 0. JP_geo_scope_meta")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
