"""Stage 8: group generation/labeling for Tokyo -> "0. JP_group_checkin_labeled".

Original script for `0. US_group_checkin_labeled` was not found anywhere in
the repo (confirmed missing). Logic is adapted from the clique-detection
approach in STSPGR/.../preprocess_Kmeans.py (lines 167-224, fully read) --
sliding 30-minute time window per venue, find cliques (size >= 2) within the
friendship graph among the users present in that window -- with two
deliberate differences from that reference script:
  1. Output goes to MongoDB in the `0. US_group_checkin_labeled` schema
     instead of parquet.
  2. Per the user's explicit decision, the Tokyo bounding box is already
     applied consistently from stage 1 onward (checkins are already
     Tokyo-only here), fixing the US pipeline's inconsistency where group
     labeling was nationwide while checkin/POI clustering was city-bound.

Group_ID semantics: matches the observed US schema where distinct Group_ID
count (172,099) is smaller than total document count (184,256) -- i.e. the
same friend-clique can recur at the same venue on a different day/time,
producing multiple rows sharing one Group_ID. Reproduced here by assigning
one Group_ID per distinct (venue_id, frozenset(members)) pair, with one row
per actual occurrence (time window) of that pair.

`cluster_label` (present in the US schema, separate from `cluster_id_k`) is
intentionally omitted -- its source script was never found, and a direct
grep confirms groupembedding.py never reads it, so it is not required
downstream.
"""
import time

import networkx as nx
import numpy as np
import pandas as pd
from pymongo import MongoClient

DATA_DIR = "/home/gpuadmin/dev/group/dataset/foursquare_global/dataset_WWW2019"
HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"
TIME_WINDOW_MIN = 30


def main():
    t0 = time.time()
    client = MongoClient(HOST, PORT, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    print("Loading 2. JP_checkin_v2 ...")
    checkins = pd.DataFrame(list(db["2. JP_checkin_v2"].find(
        {}, {"_id": 0, "user_id": 1, "venue_id": 1, "latitude": 1, "longitude": 1,
             "category": 1, "TL": 1, "local_datetime": 1}
    )))
    checkins["local_datetime"] = pd.to_datetime(checkins["local_datetime"])
    print(f"  rows: {len(checkins):,}, users: {checkins['user_id'].nunique():,}")

    tokyo_user_ids = set(checkins["user_id"].unique())

    print("\nLoading friendship edges, restricting to Tokyo users ...")
    friends = pd.read_csv(
        f"{DATA_DIR}/dataset_WWW_friendship_new.txt", sep="\t", header=None,
        names=["u1", "u2"], dtype=str,
    )
    friends_tokyo = friends[friends["u1"].isin(tokyo_user_ids) & friends["u2"].isin(tokyo_user_ids)]
    print(f"  global edges: {len(friends):,}, Tokyo-restricted edges: {len(friends_tokyo):,}")

    G_social = nx.Graph()
    G_social.add_edges_from(friends_tokyo[["u1", "u2"]].values)
    print(f"  social graph: {G_social.number_of_nodes():,} nodes, {G_social.number_of_edges():,} edges")

    print(f"\nSliding {TIME_WINDOW_MIN}-min window clique detection per venue ...")
    checkins = checkins.sort_values(["venue_id", "local_datetime"])
    groups_list = []
    n_venues = checkins["venue_id"].nunique()
    for vi, (vid, group) in enumerate(checkins.groupby("venue_id", sort=False)):
        if vi % 20000 == 0:
            print(f"  venue {vi:,}/{n_venues:,} ...")
        times = group["local_datetime"].values
        users = group["user_id"].values
        lat, lon, cat = group["latitude"].iloc[0], group["longitude"].iloc[0], group["category"].iloc[0]
        n = len(group)
        left = 0
        for right in range(n):
            while (times[right] - times[left]) > np.timedelta64(TIME_WINDOW_MIN, "m"):
                left += 1
            window_users = users[left : right + 1]
            unique_users = list(set(window_users))
            if len(unique_users) < 2:
                continue
            subg = G_social.subgraph(unique_users)
            if subg.number_of_edges() == 0:
                continue
            for clq in nx.find_cliques(subg):
                if len(clq) >= 2:
                    groups_list.append({
                        "venue_id": vid,
                        "members": tuple(sorted(clq)),
                        "time": times[right],
                        "TL": int(group["TL"].iloc[right]),
                        "Latitude": float(lat),
                        "Longitude": float(lon),
                        "VenueCategoryname": cat,
                    })

    print(f"\nRaw clique events: {len(groups_list):,}")
    groups_df = pd.DataFrame(groups_list)
    groups_df = groups_df.drop_duplicates(subset=["venue_id", "members", "time"])
    print(f"Deduplicated events: {len(groups_df):,}")

    print("\nAssigning stable Group_ID per (venue_id, members) pair ...")
    key = list(zip(groups_df["venue_id"], groups_df["members"]))
    unique_keys = sorted(set(key))
    key_to_gid = {k: i + 1 for i, k in enumerate(unique_keys)}
    groups_df["Group_ID"] = [key_to_gid[k] for k in key]
    print(f"  distinct Group_ID: {groups_df['Group_ID'].nunique():,}, total rows: {len(groups_df):,}")

    docs = []
    for row in groups_df.itertuples(index=False):
        users_int = [int(u) for u in row.members]
        docs.append({
            "Group_ID": int(row.Group_ID),
            "Users": users_int,
            "users": len(users_int),
            "VenueID": row.venue_id,
            "Latitude": row.Latitude,
            "Longitude": row.Longitude,
            "VenueCategoryname": row.VenueCategoryname,
            "TL": row.TL,
        })

    print("\nWriting 0. JP_group_checkin_labeled ...")
    db["0. JP_group_checkin_labeled"].delete_many({})
    BATCH = 20000
    for i in range(0, len(docs), BATCH):
        db["0. JP_group_checkin_labeled"].insert_many(docs[i : i + BATCH])
    print(f"  0. JP_group_checkin_labeled: {db['0. JP_group_checkin_labeled'].estimated_document_count():,} docs")

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
