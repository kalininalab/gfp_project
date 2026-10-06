"""Integrity checks for the frozen target-protein adaptation design."""

import csv
import json
import ast
from pathlib import Path


ROOT = Path("results/target_adaptation_seed42_46")
DATA = Path("data/processed/baseline_v1/sequences.csv")
REFERENCE = Path("results/transfer_cgreGFP_seed42_51/subsets")


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_frozen_budget_and_arms():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["rounds"] * protocol["target_sequences_per_round"] == 960
    assert protocol["one_shot_target_count"] == 960
    assert set(protocol["arms"]) == {
        "iterative_fancy", "iterative_random", "oneshot_fancy", "oneshot_random"
    }
    assert len(protocol["pairs"]) == 6


def test_source_target_and_test_pool_are_disjoint():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    rows = read(DATA)
    aliases = {"cgreGFP": "cgre", "amacGFP": "amac", "ppluGFP": "pplu"}
    for pair in protocol["pairs"]:
        source, target = pair.split("_to_")
        target_pool = {row["record_id"] for row in rows if row["gene"] == target and row["split"] != "test"}
        target_test = {row["record_id"] for row in rows if row["gene"] == target and row["split"] == "test"}
        assert len(target_pool) == protocol["target_pool_counts"][target]
        assert len(target_test) == protocol["target_test_counts"][target]
        assert target_pool.isdisjoint(target_test)
        for seed in protocol["seeds"]:
            subset = read(REFERENCE / f"{aliases[source]}_seed{seed}.csv")
            source_train = {row["record_id"] for row in subset if row["split"] == "train"}
            source_validation = {row["record_id"] for row in subset if row["split"] == "validation"}
            assert len(source_train) == protocol["source_train_count"]
            assert len(source_validation) == protocol["source_validation_count"]
            groups = [source_train, source_validation, target_pool, target_test]
            assert all(groups[i].isdisjoint(groups[j]) for i in range(4) for j in range(i + 1, 4))


def test_pair_parsing_is_reachable_after_argument_validation():
    """Protect the runner from accidentally nesting initialization below raise."""
    tree = ast.parse(Path("scripts/target_adaptation/run.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = [node for node in main.body if isinstance(node, ast.Assign)]
    assigned_names = {
        item.id
        for node in assignments
        for target in node.targets
        for item in (target.elts if isinstance(target, ast.Tuple) else [target])
        if isinstance(item, ast.Name)
    }
    assert {"source", "domains"} <= assigned_names
