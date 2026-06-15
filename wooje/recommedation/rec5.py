import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch
from pymongo import MongoClient

import wooje.recommedation.rec1 as base


TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
TEST_RATIO = 0.2


def split_groups_train_valid_test(
    groups,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[list, list, list]:
    groups = list(groups)
    if len(groups) < 3:
        raise RuntimeError("not enough valid groups to split into train/valid/test")

    rng = random.Random(seed)
    rng.shuffle(groups)

    total = len(groups)
    train_count = int(total * train_ratio)
    valid_count = int(total * valid_ratio)
    test_count = total - train_count - valid_count

    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise RuntimeError(
            f"invalid split sizes: total={total}, train={train_count}, valid={valid_count}, test={test_count}"
        )

    train_groups = groups[:train_count]
    valid_groups = groups[train_count : train_count + valid_count]
    test_groups = groups[train_count + valid_count :]
    return train_groups, valid_groups, test_groups


def run_scenario_with_validation(
    store: base.MongoEmbeddingStore,
    train_groups: Sequence[dict],
    valid_groups: Sequence[dict],
    test_groups: Sequence[dict],
    scenario: base.ScenarioConfig,
    config,
    device: str,
    ks: Sequence[int],
) -> Dict[str, object]:
    print(f"[{scenario.name}] prepare train groups...")
    scenario_train_groups = base.prepare_groups_for_scenario(
        store=store,
        groups=train_groups,
        scenario=scenario,
        radius_km=config.radius_km,
        candidate_limit=config.candidate_limit,
    )
    print(f"[{scenario.name}] prepare valid groups...")
    scenario_valid_groups = base.prepare_groups_for_scenario(
        store=store,
        groups=valid_groups,
        scenario=scenario,
        radius_km=config.radius_km,
        candidate_limit=config.candidate_limit,
    )
    print(f"[{scenario.name}] prepare test groups...")
    scenario_test_groups = base.prepare_groups_for_scenario(
        store=store,
        groups=test_groups,
        scenario=scenario,
        radius_km=config.radius_km,
        candidate_limit=config.candidate_limit,
    )

    if not scenario_train_groups:
        empty_eval = {
            "evaluated_groups": 0,
            "metrics": {int(k): {"recall": 0.0, "precision": 0.0, "f1": 0.0, "ndcg": 0.0} for k in ks},
            "sample_group": None,
            "sample_positive_rank": None,
            "sample_topk": {},
        }
        return {
            "description": scenario.description,
            "train_groups": 0,
            "valid_groups": len(scenario_valid_groups),
            "test_groups": len(scenario_test_groups),
            "history": [],
            "validation": empty_eval,
            "test": empty_eval,
        }

    group_dim, poi_dim = base.infer_dimensions(store, scenario_train_groups)
    model = base.GroupPoiScorer(
        group_dim=group_dim,
        poi_dim=poi_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    print(
        f"[{scenario.name}] start training "
        f"train_groups={len(scenario_train_groups)} valid_groups={len(scenario_valid_groups)} "
        f"test_groups={len(scenario_test_groups)}"
    )
    history = base.train_model(
        store=store,
        model=model,
        optimizer=optimizer,
        train_groups=scenario_train_groups,
        scenario=scenario,
        config=config,
        device=device,
    )

    print(f"[{scenario.name}] training complete. start validation evaluation...")
    valid_evaluation = base.evaluate_model(
        model=model,
        store=store,
        eval_groups=scenario_valid_groups,
        scenario=scenario,
        config=config,
        device=device,
        ks=ks,
    )

    print(f"[{scenario.name}] validation complete. start test evaluation...")
    test_evaluation = base.evaluate_model(
        model=model,
        store=store,
        eval_groups=scenario_test_groups,
        scenario=scenario,
        config=config,
        device=device,
        ks=ks,
    )

    return {
        "description": scenario.description,
        "train_groups": len(scenario_train_groups),
        "valid_groups": len(scenario_valid_groups),
        "test_groups": len(scenario_test_groups),
        "history": history,
        "validation": valid_evaluation,
        "test": test_evaluation,
    }


def print_metric_table(split_name: str, scenario_name: str, evaluation: Dict[str, object], ks: Sequence[int]) -> None:
    metrics = evaluation["metrics"]
    print()
    print(f"[{scenario_name}][{split_name}]")
    print(f"evaluated_groups: {evaluation['evaluated_groups']}")
    print("K\trecall\tprecision\tf1\tndcg")
    for k in ks:
        row = metrics[int(k)]
        print(
            f"{k}\t"
            f"{row['recall']:.6f}\t"
            f"{row['precision']:.6f}\t"
            f"{row['f1']:.6f}\t"
            f"{row['ndcg']:.6f}"
        )


def main() -> None:
    config = base.parse_args()
    base.set_seed(config.seed)

    default_rec1_output = str(Path(base.__file__).with_name("rec1_results.json"))
    if config.output_json == default_rec1_output:
        config.output_json = str(Path(__file__).with_name("rec5_results.json"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("config:", json.dumps(asdict(config), ensure_ascii=False))
    print(
        f"split_ratio: train={TRAIN_RATIO:.1%}, valid={VALID_RATIO:.1%}, test={TEST_RATIO:.1%}"
    )

    client = MongoClient(base.HOST, base.PORT)
    db = client[base.DB_NAME]
    store = base.MongoEmbeddingStore(
        group_col=db[base.GROUP_COL_NAME],
        poi_col=db[base.POI_COL_NAME],
    )

    print("[setup] warming Mongo indexes...")
    store.warm_indexes()

    groups = store.load_valid_groups(config.max_groups)
    train_groups, valid_groups, test_groups = split_groups_train_valid_test(
        groups=groups,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        seed=config.seed,
    )

    print(f"loaded valid groups: {len(groups)}")
    print(f"train split: {len(train_groups)}")
    print(f"valid split: {len(valid_groups)}")
    print(f"test split: {len(test_groups)}")

    results = {
        "config": asdict(config),
        "device": device,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "valid": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "num_loaded_groups": len(groups),
        "num_train_groups": len(train_groups),
        "num_valid_groups": len(valid_groups),
        "num_test_groups": len(test_groups),
        "scenarios": {},
    }

    for scenario in base.SCENARIOS:
        print()
        print(f"running scenario: {scenario.name}")
        scenario_result = run_scenario_with_validation(
            store=store,
            train_groups=train_groups,
            valid_groups=valid_groups,
            test_groups=test_groups,
            scenario=scenario,
            config=config,
            device=device,
            ks=base.DEFAULT_KS,
        )
        results["scenarios"][scenario.name] = scenario_result
        print(f"[{scenario.name}] train_groups: {scenario_result['train_groups']}")
        print(f"[{scenario.name}] valid_groups: {scenario_result['valid_groups']}")
        print(f"[{scenario.name}] test_groups: {scenario_result['test_groups']}")
        print_metric_table("validation", scenario.name, scenario_result["validation"], base.DEFAULT_KS)
        print_metric_table("test", scenario.name, scenario_result["test"], base.DEFAULT_KS)

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
