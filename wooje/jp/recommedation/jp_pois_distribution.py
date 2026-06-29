"""Stage 10: per-POI visit distribution for JP -> "2. JP_pois_distribution".

Original script not found (confirmed missing for the US side too). Inferred
from the observed `2. US_pois_distribution` schema: one doc per venue_id with
visit1..visit4 (checkin counts bucketed by TL) plus venue metadata
(latitude/longitude/category/category_id/loc_cluster_id_k5/country) joined
from `2. JP_checkin_v2`.
"""
import pandas as pd
from pymongo import MongoClient

HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"


def main():
    client = MongoClient(HOST, PORT, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    print("Loading 2. JP_checkin_v2 ...")
    checkins = pd.DataFrame(list(db["2. JP_checkin_v2"].find(
        {}, {"_id": 0, "venue_id": 1, "latitude": 1, "longitude": 1, "category": 1,
             "category_id": 1, "loc_cluster_id_k5": 1, "TL": 1}
    )))
    print(f"  rows: {len(checkins):,}")

    print("Aggregating per-venue TL visit counts ...")
    visit_counts = (
        checkins.groupby(["venue_id", "TL"]).size().unstack(fill_value=0)
    )
    for tl in [1, 2, 3, 4]:
        if tl not in visit_counts.columns:
            visit_counts[tl] = 0
    visit_counts = visit_counts[[1, 2, 3, 4]]
    visit_counts.columns = ["visit1", "visit2", "visit3", "visit4"]

    venue_meta = (
        checkins.drop_duplicates(subset="venue_id", keep="first")
        .set_index("venue_id")[["latitude", "longitude", "category", "category_id", "loc_cluster_id_k5"]]
    )

    dist = venue_meta.join(visit_counts).reset_index()
    dist["country"] = "JP"
    dist["category_id"] = dist["category_id"].astype(int)
    dist["loc_cluster_id_k5"] = dist["loc_cluster_id_k5"].astype(int)
    for col in ["visit1", "visit2", "visit3", "visit4"]:
        dist[col] = dist[col].astype(int)

    print(f"  venues: {len(dist):,}")

    docs = dist.to_dict("records")
    db["2. JP_pois_distribution"].delete_many({})
    BATCH = 50000
    for i in range(0, len(docs), BATCH):
        db["2. JP_pois_distribution"].insert_many(docs[i : i + BATCH])
    print(f"  2. JP_pois_distribution: {db['2. JP_pois_distribution'].estimated_document_count():,} docs")


if __name__ == "__main__":
    main()
