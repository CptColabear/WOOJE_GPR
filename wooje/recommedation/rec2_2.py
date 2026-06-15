import json
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wooje.recommedation.rec2copy as base_exp


TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

RESULT_JSON_PATH = Path(__file__).with_name("topk_results_summary_70_20_10.json")


def split_groups_by_ratio(group_docs, train_ratio, valid_ratio, test_ratio, seed):
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"split ratio sum must be 1.0, got {total_ratio}")

    group_docs = list(group_docs)
    if len(group_docs) < 10:
        raise ValueError("usable_group_docs is too small to split reliably.")

    rng = random.Random(seed)
    rng.shuffle(group_docs)

    total = len(group_docs)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_docs = group_docs[:train_end]
    valid_docs = group_docs[train_end:valid_end]
    test_docs = group_docs[valid_end:]

    if not train_docs or not valid_docs or not test_docs:
        raise ValueError(
            f"invalid split sizes: train={len(train_docs)}, valid={len(valid_docs)}, test={len(test_docs)}"
        )

    return train_docs, valid_docs, test_docs


def configure_split():
    train_docs, valid_docs, test_docs = split_groups_by_ratio(
        base_exp.usable_group_docs,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO,
        seed=base_exp.SEED,
    )

    base_exp.train_docs = train_docs
    base_exp.valid_docs = valid_docs
    base_exp.test_docs = test_docs

    print("\n" + "=" * 100)
    print("[rec2_2] Using split ratio train/valid/test = 70/20/10")
    print(f"[rec2_2] total usable groups = {len(base_exp.usable_group_docs)}")
    print(f"[rec2_2] train size = {len(base_exp.train_docs)}")
    print(f"[rec2_2] valid size = {len(base_exp.valid_docs)}")
    print(f"[rec2_2] test size  = {len(base_exp.test_docs)}")
    print("=" * 100)


def main():
    configure_split()

    baseline_model, baseline_valid_topk, baseline_test_topk = base_exp.run_experiment(
        exp_name="baseline_cluster_negative_70_20_10",
        sampling_mode="baseline",
    )

    base_exp.print_metric_table("1) baseline cluster negative - VALID (70/20/10)", baseline_valid_topk)
    base_exp.print_metric_table("1) baseline cluster negative - TEST (70/20/10)", baseline_test_topk)

    hard_model, hard_valid_topk, hard_test_topk = base_exp.run_experiment(
        exp_name="hard_cluster_distance_negative_70_20_10",
        sampling_mode="hard",
    )

    base_exp.print_metric_table("2) hard cluster distance negative - VALID (70/20/10)", hard_valid_topk)
    base_exp.print_metric_table("2) hard cluster distance negative - TEST (70/20/10)", hard_test_topk)

    base_exp.compare_results(
        title="TEST performance comparison (70/20/10): baseline vs hard negative",
        baseline_result=baseline_test_topk,
        hard_result=hard_test_topk,
    )

    if len(base_exp.test_docs) > 0:
        sample_gdoc = base_exp.test_docs[0]

        base_exp.show_topk_recommendations(
            model=baseline_model,
            gdoc=sample_gdoc,
            candidate_mode="baseline",
            topk=10,
            max_candidates=base_exp.MAX_CANDIDATES,
        )

        base_exp.show_topk_recommendations(
            model=hard_model,
            gdoc=sample_gdoc,
            candidate_mode="hard",
            topk=10,
            max_candidates=base_exp.MAX_CANDIDATES,
        )

    results_summary = {
        "split_ratio": {
            "train": TRAIN_RATIO,
            "valid": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            "train": len(base_exp.train_docs),
            "valid": len(base_exp.valid_docs),
            "test": len(base_exp.test_docs),
        },
        "baseline_valid": baseline_valid_topk,
        "baseline_test": baseline_test_topk,
        "hard_valid": hard_valid_topk,
        "hard_test": hard_test_topk,
    }

    with RESULT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"[rec2_2] saved results to: {RESULT_JSON_PATH.name}")


if __name__ == "__main__":
    main()
