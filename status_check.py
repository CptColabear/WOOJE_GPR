# status_check.py

from datetime import datetime
import sys
import platform
import yaml
from pymongo import MongoClient


CONFIG_PATH = "config.yaml"

COLLECTION_NAMES = [
    "0. US_group_checkin_labeled",
    "2. US_checkin_v2",
]


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    required_keys = ["MONGO_HOST", "MONGO_PORT", "DB_NAME"]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"config.yaml에 {key} 값이 없습니다.")

    return config


def check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_mongodb_and_collections(config):
    host = config["MONGO_HOST"]
    port = int(config["MONGO_PORT"])
    db_name = config["DB_NAME"]

    mongo_uri = f"mongodb://{host}:{port}"

    result = {
        "connected": False,
        "collections": {},
    }

    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=3000
        )

        # MongoDB 연결 확인
        client.admin.command("ping")
        result["connected"] = True

        db = client[db_name]

        existing_collections = db.list_collection_names()

        for collection_name in COLLECTION_NAMES:
            if collection_name not in existing_collections:
                result["collections"][collection_name] = {
                    "exists": False,
                    "has_data": False,
                    "count": 0,
                }
                continue

            collection = db[collection_name]

            # 전체 count가 부담될 수 있으므로, 우선 데이터 존재 여부만 빠르게 확인
            first_doc = collection.find_one()

            if first_doc is None:
                count = 0
                has_data = False
            else:
                count = collection.estimated_document_count()
                has_data = True

            result["collections"][collection_name] = {
                "exists": True,
                "has_data": has_data,
                "count": count,
            }

        client.close()
        return result

    except Exception as e:
        result["error"] = str(e)
        return result


def run(config_path=CONFIG_PATH):
    config = load_config(config_path)

    cuda_ok = check_cuda()
    mongo_result = check_mongodb_and_collections(config)

    mongo_ok = mongo_result["connected"]

    collections_ok = all(
        info["exists"] and info["has_data"]
        for info in mongo_result["collections"].values()
    )

    print("================")
    print(" System Spec Snapshot ")
    print("================")
    print(f"Time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python: {sys.version}")
    print(f"OS: {platform.platform()}")

    print("\n[GPU]")
    if cuda_ok:
        print("CUDA Enable")
    else:
        print("CUDA Disable")

    print("\n[MongoDB]")
    if mongo_ok:
        print("Successfully Connected")
        print(f"Host: {config['MONGO_HOST']}")
        print(f"Port: {config['MONGO_PORT']}")
        print(f"DB: {config['DB_NAME']}")
    else:
        print("Connection Failed")
        if "error" in mongo_result:
            print(f"Error: {mongo_result['error']}")

    print("\n[Collections]")
    for collection_name, info in mongo_result["collections"].items():
        print(f"- {collection_name}")

        if not info["exists"]:
            print("  Status: Not Found")
        elif not info["has_data"]:
            print("  Status: Exists, but Empty")
        else:
            print("  Status: Exists")
            print(f"  Documents: {info['count']}")

    return cuda_ok and mongo_ok and collections_ok


if __name__ == "__main__":
    result = run()
    print("\nResult:", result)