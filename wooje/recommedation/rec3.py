import json
import random
from dataclasses import asdict
from pathlib import Path

import torch
from pymongo import MongoClient

import wooje.recommedation.rec1 as base


FIXED_TEST_SAMPLE_SIZE = 361


def split_groups_fixed_test(groups, fixed_test_size, seed):
    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)

    if len(groups) <= fixed_test_size:
        raise RuntimeError(
            f"not enough valid groups to allocate test={fixed_test_size}. "
            f"loaded={len(groups)}"
        )

    test_groups = groups[:fixed_test_size]
    train_groups = groups[fixed_test_size:]
    return train_groups, test_groups


def main():
    config = base.parse_args()
    base.set_seed(config.seed)

    default_rec1_output = str(Path(base.__file__).with_name("rec1_results.json"))
    if config.output_json == default_rec1_output:
        config.output_json = str(Path(__file__).with_name("rec3_results.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("config:", json.dumps(asdict(config), ensure_ascii=False))
    print(f"fixed_test_sample_size: {FIXED_TEST_SAMPLE_SIZE}")

    client = MongoClient(base.HOST, base.PORT)
    db = client[base.DB_NAME]
    store = base.MongoEmbeddingStore(
        group_col=db[base.GROUP_COL_NAME],
        poi_col=db[base.POI_COL_NAME],
    )

    print("[setup] warming Mongo indexes...")
    store.warm_indexes()

    groups = store.load_valid_groups(config.max_groups)
    train_groups, test_groups = split_groups_fixed_test(
        groups=groups,
        fixed_test_size=FIXED_TEST_SAMPLE_SIZE,
        seed=config.seed,
    )

    print(f"loaded valid groups: {len(groups)}")
    print(f"train split: {len(train_groups)}")
    print(f"test split: {len(test_groups)}")

    results = {
        "config": asdict(config),
        "device": device,
        "fixed_test_sample_size": FIXED_TEST_SAMPLE_SIZE,
        "num_loaded_groups": len(groups),
        "num_train_groups": len(train_groups),
        "num_test_groups": len(test_groups),
        "scenarios": {},
    }

    for scenario in base.SCENARIOS:
        print()
        print(f"running scenario: {scenario.name}")
        scenario_result = base.run_scenario(
            store=store,
            train_groups=train_groups,
            test_groups=test_groups,
            scenario=scenario,
            config=config,
            device=device,
            ks=base.DEFAULT_KS,
        )
        results["scenarios"][scenario.name] = scenario_result
        base.print_metric_table(scenario.name, scenario_result, base.DEFAULT_KS)

    output_path = Path(config.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(base.make_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"saved results to: {output_path}")


if __name__ == "__main__":
    main()
