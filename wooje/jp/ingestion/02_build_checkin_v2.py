"""Stage 2/3: enrich "1. JP_checkin" into "2. JP_checkin_v2".

Adds local time (utc_time + offset_min), TL (time-label 1-4, using the
boundaries empirically confirmed from the US pipeline's `2. US_checkin_v2`:
TL1=05-10h, TL2=11-15h, TL3=16-20h, TL4=21-04h wrapping midnight), and
category_id (rank index over the alphabetically sorted distinct category
names present in the Tokyo POI set -- JP's category vocabulary differs from
US's, so JP's category_id values are NOT expected to numerically match US's
for the same category name; this is fine since each country's pipeline is
self-contained).

loc_cluster_id_k5 / loc_cluster_id_bestk are intentionally left unset here --
they get added in place by the stage 4/5 clustering scripts, exactly as the
US pipeline does.
"""
import numpy as np
import pandas as pd
from pymongo import MongoClient

HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"


def assign_tl(hour: int) -> int:
    if 5 <= hour <= 10:
        return 1
    if 11 <= hour <= 15:
        return 2
    if 16 <= hour <= 20:
        return 3
    return 4  # 21-23 and 0-4, wraps midnight


def main():
    client = MongoClient(HOST, PORT, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    print("Loading 1. JP_checkin ...")
    checkins = pd.DataFrame(list(db["1. JP_checkin"].find({}, {"_id": 0})))
    print(f"  rows: {len(checkins):,}")

    print("Loading 1. JP_pois for category/lat/lon join ...")
    pois = pd.DataFrame(list(db["1. JP_pois"].find(
        {}, {"_id": 0, "venue_id": 1, "latitude": 1, "longitude": 1, "category": 1}
    )))
    print(f"  rows: {len(pois):,}")

    print("Computing local time + TL ...")
    utc_dt = pd.to_datetime(checkins["utc_time"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    local_dt = (utc_dt + pd.to_timedelta(checkins["offset_min"], unit="m")).dt.tz_localize(None)
    checkins["local_datetime"] = local_dt
    checkins["local_time"] = local_dt.astype(str)
    checkins["TL"] = local_dt.dt.hour.map(assign_tl)

    print("Mapping category_id (rank over sorted distinct JP category names) ...")
    categories_sorted = sorted(pois["category"].dropna().unique().tolist())
    category_to_id = {name: idx for idx, name in enumerate(categories_sorted)}
    print(f"  distinct categories: {len(categories_sorted):,}")

    checkins = checkins.merge(pois, on="venue_id", how="left")
    checkins["category_id"] = checkins["category"].map(category_to_id)

    print("Computing user_id_count (per-user total checkins) ...")
    user_counts = checkins.groupby("user_id")["user_id"].transform("count")
    checkins["user_id_count"] = user_counts

    print(f"\nTL distribution:\n{checkins['TL'].value_counts().sort_index()}")
    print(f"\ncategory_id range: {checkins['category_id'].min()}..{checkins['category_id'].max()}")

    print("\nWriting 2. JP_checkin_v2 ...")
    db["2. JP_checkin_v2"].delete_many({})
    records = checkins.to_dict("records")
    BATCH = 200_000
    for i in range(0, len(records), BATCH):
        db["2. JP_checkin_v2"].insert_many(records[i : i + BATCH])
    print(f"  2. JP_checkin_v2: {db['2. JP_checkin_v2'].estimated_document_count():,} docs")

    db["0. JP_category_map"].delete_many({})
    db["0. JP_category_map"].insert_many(
        [{"category": name, "category_id": idx} for name, idx in category_to_id.items()]
    )
    print(f"  0. JP_category_map: {len(category_to_id):,} docs")


if __name__ == "__main__":
    main()
