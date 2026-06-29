"""Stage 7a-c: build the per-(user,date,TL) sequence chain for Tokyo, plus the
user x TL x category visit-count distribution.

Original scripts for these specific transforms were not found in the repo
(confirmed missing for the US side too). Inferred from the observed US
schemas (`2. US_sequence`, `2. US_sequence_catlen_gt1_with_catid`,
`3. US_user_TL_category_dist`) plus a direct read of
`pattern_embedding_attn.py`, which comments `cat = int(d["category_id"])
# 1..431` when reading `3. US_user_TL_category_dist` -- i.e. that
collection's category_id is in the *sequence* vocabulary (1-indexed,
reserving 0=pad), not the checkin_v2/POI 0-indexed vocabulary used elsewhere.

Design choice made explicit here: the sequence vocabulary is simply
checkin_v2's existing 0-indexed category_id (`0. JP_category_map`) shifted by
+1, reserving id 0 for pad and id (n_items+1) for the MLM mask token. This is
internally consistent (every category that can appear in a sequence is
guaranteed to have a sequence-vocab id) even though it cannot be verified
byte-for-byte against whatever the original US mapping script did (it is
lost) -- the US side shows n_items=431 vs checkin_v2's 429 categories, a
small, unexplained discrepancy that this design intentionally does not
attempt to reproduce.
"""
import pandas as pd
from pymongo import MongoClient

HOST, PORT, DB_NAME = "10.255.68.40", 27017, "ejoow2"


def main():
    client = MongoClient(HOST, PORT, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]

    print("Loading 2. JP_checkin_v2 ...")
    checkins = pd.DataFrame(list(db["2. JP_checkin_v2"].find(
        {}, {"_id": 0, "user_id": 1, "venue_id": 1, "category": 1, "category_id": 1, "TL": 1, "local_datetime": 1}
    )))
    print(f"  rows: {len(checkins):,}")

    checkins["local_datetime"] = pd.to_datetime(checkins["local_datetime"])
    checkins["date"] = checkins["local_datetime"].dt.date.astype(str)
    checkins = checkins.sort_values("local_datetime")

    print("\nLoading category vocabulary (0. JP_category_map) ...")
    cat_map = pd.DataFrame(list(db["0. JP_category_map"].find({}, {"_id": 0})))
    category_to_seq_id = {row["category"]: int(row["category_id"]) + 1 for _, row in cat_map.iterrows()}
    n_items = len(category_to_seq_id)
    mask_id = n_items + 1
    vocab_size = n_items + 2
    print(f"  n_items={n_items}, pad_id=0, mask_id={mask_id}, vocab_size={vocab_size}")

    print("\n[1/3] Building 2. JP_sequence (group by user_id/date/TL) ...")
    seq_docs = []
    for (user_id, date, tl), g in checkins.groupby(["user_id", "date", "TL"], sort=False):
        seq_docs.append({
            "user_id": user_id,
            "date": date,
            "TL": int(tl),
            "venue_sequence": g["venue_id"].tolist(),
            "category_sequence": g["category"].tolist(),
        })
    print(f"  total sequences: {len(seq_docs):,}")

    db["2. JP_sequence"].delete_many({})
    BATCH = 50000
    for i in range(0, len(seq_docs), BATCH):
        db["2. JP_sequence"].insert_many(seq_docs[i : i + BATCH])
    print(f"  2. JP_sequence: {db['2. JP_sequence'].estimated_document_count():,} docs")

    print("\n[2/3] Filtering to category_sequence length > 1 -> 2. JP_sequence_catlen_gt1_with_catid ...")
    catlen_gt1 = [d for d in seq_docs if len(d["category_sequence"]) > 1]
    print(f"  catlen_gt1: {len(catlen_gt1):,} docs")

    db["2. JP_sequence_catlen_gt1"].delete_many({})
    for i in range(0, len(catlen_gt1), BATCH):
        db["2. JP_sequence_catlen_gt1"].insert_many(
            [dict(d) for d in catlen_gt1[i : i + BATCH]]
        )
    print(f"  2. JP_sequence_catlen_gt1: {db['2. JP_sequence_catlen_gt1'].estimated_document_count():,} docs")

    with_catid_docs = []
    for d in catlen_gt1:
        cat_ids = [category_to_seq_id.get(c) for c in d["category_sequence"]]
        has_missing = any(c is None for c in cat_ids)
        # category_to_seq_id is built from checkin_v2's own full category set, the same
        # source category_sequence is drawn from, so a missing id should be impossible --
        # kept as a defensive flag (consistent with the original schema) rather than an
        # assertion, but the list-comprehension-drops-None branch is intentionally not
        # used since that would desync category_id_sequence's length from category_sequence.
        with_catid_docs.append({
            **d,
            "category_id_sequence": cat_ids,
            "has_missing_category_id": has_missing,
        })
    db["2. JP_sequence_catlen_gt1_with_catid"].delete_many({})
    for i in range(0, len(with_catid_docs), BATCH):
        db["2. JP_sequence_catlen_gt1_with_catid"].insert_many(with_catid_docs[i : i + BATCH])
    n_missing = sum(1 for d in with_catid_docs if d["has_missing_category_id"])
    print(f"  2. JP_sequence_catlen_gt1_with_catid: {db['2. JP_sequence_catlen_gt1_with_catid'].estimated_document_count():,} docs "
          f"(has_missing_category_id=True: {n_missing})")

    print("\n[3/3] Building 3. JP_user_TL_category_dist (sequence-vocab category_id) ...")
    dist = (
        checkins.groupby(["user_id", "TL", "category"])
        .size()
        .reset_index(name="category_count")
    )
    dist["category_id"] = dist["category"].map(category_to_seq_id)
    dist_docs = dist.to_dict("records")
    db["3. JP_user_TL_category_dist"].delete_many({})
    for i in range(0, len(dist_docs), BATCH):
        db["3. JP_user_TL_category_dist"].insert_many(dist_docs[i : i + BATCH])
    print(f"  3. JP_user_TL_category_dist: {db['3. JP_user_TL_category_dist'].estimated_document_count():,} docs")

    db["0. JP_sequence_vocab_meta"].replace_one(
        {"_id": "sequence_vocab"},
        {
            "_id": "sequence_vocab",
            "n_items": n_items,
            "pad_id": 0,
            "mask_id": mask_id,
            "vocab_size": vocab_size,
            "source": "0. JP_category_map (category_id + 1)",
        },
        upsert=True,
    )
    print("\nWrote 0. JP_sequence_vocab_meta")


if __name__ == "__main__":
    main()
