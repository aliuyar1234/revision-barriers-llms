from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter
from copy import deepcopy
from typing import Any

from src.data.mhst_worlds import CHALLENGER_POSITION_ORDER

RCICL_OPTION_LABELS = ["dax", "wug"]
RCICL_LATE_PACKET_LENGTH = 6

FEATURE_ORDER = ["color", "shape", "size", "texture"]
RULE_FEATURES = ["color", "shape", "size"]
FEATURE_VALUES: dict[str, list[str]] = {
    "color": ["blue", "red"],
    "shape": ["circle", "triangle"],
    "size": ["large", "small"],
    "texture": ["solid", "striped"],
}

RULE_LIBRARY: dict[str, dict[str, str]] = {
    "color_red": {"feature": "color", "positive_value": "red"},
    "color_blue": {"feature": "color", "positive_value": "blue"},
    "shape_circle": {"feature": "shape", "positive_value": "circle"},
    "shape_triangle": {"feature": "shape", "positive_value": "triangle"},
    "size_small": {"feature": "size", "positive_value": "small"},
    "size_large": {"feature": "size", "positive_value": "large"},
}


def stable_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _rng(*parts: Any) -> random.Random:
    return random.Random(stable_seed(*parts))


def object_id(features: dict[str, str]) -> str:
    return "__".join(f"{feature}={features[feature]}" for feature in FEATURE_ORDER)


def build_object_universe() -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    for color, shape, size, texture in itertools.product(
        FEATURE_VALUES["color"],
        FEATURE_VALUES["shape"],
        FEATURE_VALUES["size"],
        FEATURE_VALUES["texture"],
    ):
        objects.append(
            {
                "color": color,
                "shape": shape,
                "size": size,
                "texture": texture,
            }
        )
    objects.sort(key=object_id)
    return objects


OBJECT_UNIVERSE = build_object_universe()


def evaluate_rule(rule_id: str, features: dict[str, str]) -> str:
    rule = RULE_LIBRARY[rule_id]
    return "dax" if features[rule["feature"]] == rule["positive_value"] else "wug"


def rule_base_feature(rule_id: str) -> str:
    return RULE_LIBRARY[rule_id]["feature"]


def eligible_rule_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for incumbent_rule_id in RULE_LIBRARY:
        for challenger_rule_id in RULE_LIBRARY:
            if incumbent_rule_id == challenger_rule_id:
                continue
            if rule_base_feature(incumbent_rule_id) == rule_base_feature(challenger_rule_id):
                continue
            pairs.append((incumbent_rule_id, challenger_rule_id))
    return pairs


ELIGIBLE_RULE_PAIRS = eligible_rule_pairs()


def _object_with_id(features: dict[str, str]) -> dict[str, Any]:
    return {
        "object_id": object_id(features),
        "features": deepcopy(features),
    }


def _example_from_object(features: dict[str, str], *, rule_id: str) -> dict[str, Any]:
    return {
        **_object_with_id(features),
        "label": evaluate_rule(rule_id, features),
        "rule_id": rule_id,
    }


def _shuffle_objects(
    seed: int,
    split: str,
    m: int,
    k: int,
    world_index: int,
    tag: str,
    rule_pair: tuple[str, str],
    objects: list[dict[str, str]],
) -> list[dict[str, str]]:
    shuffled = list(objects)
    _rng(seed, split, m, k, world_index, tag, *rule_pair).shuffle(shuffled)
    return shuffled


def _disagreement_and_agreement_objects(
    incumbent_rule_id: str,
    challenger_rule_id: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    disagreement: list[dict[str, str]] = []
    agreement: list[dict[str, str]] = []
    for features in OBJECT_UNIVERSE:
        incumbent_label = evaluate_rule(incumbent_rule_id, features)
        challenger_label = evaluate_rule(challenger_rule_id, features)
        if incumbent_label == challenger_label:
            agreement.append(features)
        else:
            disagreement.append(features)
    return disagreement, agreement


def _late_challenger_cycle(
    disagreement_schedule: list[dict[str, str]],
    *,
    prefix_challenger_count: int,
    prefix_incumbent_count: int,
) -> list[dict[str, str]]:
    used_count = prefix_challenger_count + prefix_incumbent_count + 1
    if not disagreement_schedule:
        return []
    start = used_count % len(disagreement_schedule)
    return disagreement_schedule[start:] + disagreement_schedule[:start]


def _late_examples_by_dose(
    challenger_rule_id: str,
    disagreement_schedule: list[dict[str, str]],
    agreement_schedule: list[dict[str, str]],
    *,
    prefix_challenger_count: int,
    prefix_incumbent_count: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    challenger_cycle = _late_challenger_cycle(
        disagreement_schedule,
        prefix_challenger_count=prefix_challenger_count,
        prefix_incumbent_count=prefix_incumbent_count,
    )
    neutral_cycle = agreement_schedule[:RCICL_LATE_PACKET_LENGTH]
    if len(neutral_cycle) < RCICL_LATE_PACKET_LENGTH:
        raise ValueError("RC-ICL agreement schedule is too short for the locked late packet.")

    examples_by_dose: dict[str, list[dict[str, Any]]] = {}
    object_ids_by_dose: dict[str, list[str]] = {}
    for dose in range(RCICL_LATE_PACKET_LENGTH + 1):
        challenger_positions = set(CHALLENGER_POSITION_ORDER[:dose])
        challenger_index = 0
        neutral_index = 0
        examples: list[dict[str, Any]] = []
        object_ids: list[str] = []
        for position in range(RCICL_LATE_PACKET_LENGTH):
            if position in challenger_positions:
                features = challenger_cycle[challenger_index % len(challenger_cycle)]
                challenger_index += 1
                example = _example_from_object(features, rule_id=challenger_rule_id)
            else:
                features = neutral_cycle[neutral_index]
                neutral_index += 1
                example = _example_from_object(features, rule_id=challenger_rule_id)
            examples.append(example)
            object_ids.append(example["object_id"])
        if len(set(object_ids[position] for position in range(RCICL_LATE_PACKET_LENGTH) if position not in challenger_positions)) != (
            RCICL_LATE_PACKET_LENGTH - dose
        ):
            raise ValueError("Neutral RC-ICL late fillers must be unique within one late packet.")
        examples_by_dose[str(dose)] = examples
        object_ids_by_dose[str(dose)] = object_ids
    return examples_by_dose, object_ids_by_dose


def _pair_checks(
    fresh: dict[str, Any],
    committed: dict[str, Any],
) -> dict[str, bool]:
    return {
        "same_incumbent_rule": fresh["incumbent_rule_id"] == committed["incumbent_rule_id"],
        "same_challenger_rule": fresh["challenger_rule_id"] == committed["challenger_rule_id"],
        "same_design_margin": fresh["m"] == committed["m"],
        "same_query_object": fresh["query_object"]["object_id"] == committed["query_object"]["object_id"],
        "same_prefix_multiset": Counter(fresh["prefix_object_ids"]) == Counter(committed["prefix_object_ids"]),
        "same_disagreement_schedule": fresh["disagreement_schedule_ids"] == committed["disagreement_schedule_ids"],
        "same_agreement_schedule": fresh["agreement_schedule_ids"] == committed["agreement_schedule_ids"],
        "same_late_packet_structure": fresh["late_object_ids_by_dose"] == committed["late_object_ids_by_dose"],
    }


def _build_pair_examples(
    seed: int,
    split: str,
    m: int,
    k: int,
    world_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_attempts = 256
    for attempt in range(max_attempts):
        rng = _rng(seed, split, m, k, world_index, "pair", attempt)
        incumbent_rule_id, challenger_rule_id = rng.choice(ELIGIBLE_RULE_PAIRS)
        disagreement_raw, agreement_raw = _disagreement_and_agreement_objects(incumbent_rule_id, challenger_rule_id)
        if len(disagreement_raw) < m + 2 * k + 1:
            continue

        disagreement_schedule = _shuffle_objects(
            seed,
            split,
            m,
            k,
            world_index,
            f"disagreement__attempt{attempt}",
            (incumbent_rule_id, challenger_rule_id),
            disagreement_raw,
        )
        agreement_schedule = _shuffle_objects(
            seed,
            split,
            m,
            k,
            world_index,
            f"agreement__attempt{attempt}",
            (incumbent_rule_id, challenger_rule_id),
            agreement_raw,
        )

        challenger_prefix_objects = disagreement_schedule[:k]
        incumbent_prefix_objects = disagreement_schedule[k : k + m + k]
        query_object = disagreement_schedule[k + m + k]

        if len({object_id(features) for features in challenger_prefix_objects + incumbent_prefix_objects}) != (m + 2 * k):
            continue
        if object_id(query_object) in {object_id(features) for features in challenger_prefix_objects + incumbent_prefix_objects}:
            continue

        fresh_prefix_examples = [
            _example_from_object(features, rule_id=challenger_rule_id) for features in challenger_prefix_objects
        ] + [_example_from_object(features, rule_id=incumbent_rule_id) for features in incumbent_prefix_objects]
        committed_prefix_examples = [
            _example_from_object(features, rule_id=incumbent_rule_id) for features in incumbent_prefix_objects
        ] + [_example_from_object(features, rule_id=challenger_rule_id) for features in challenger_prefix_objects]

        late_examples_by_dose, late_object_ids_by_dose = _late_examples_by_dose(
            challenger_rule_id,
            disagreement_schedule,
            agreement_schedule,
            prefix_challenger_count=k,
            prefix_incumbent_count=m + k,
        )

        query_object_payload = _object_with_id(query_object)
        shared_base = {
            "task_family": "rcicl",
            "task_variant": "pair",
            "world_id": f"rcicl_world__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
            "pair_id": f"rcicl_pair__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
            "split": split,
            "m": m,
            "k": k,
            "incumbent": incumbent_rule_id,
            "challenger": challenger_rule_id,
            "incumbent_rule_id": incumbent_rule_id,
            "challenger_rule_id": challenger_rule_id,
            "incumbent_option": evaluate_rule(incumbent_rule_id, query_object),
            "challenger_option": evaluate_rule(challenger_rule_id, query_object),
            "distractors": [],
            "late_distractor_cycle": [],
            "query_object": query_object_payload,
            "disagreement_size": len(disagreement_schedule),
            "agreement_size": len(agreement_schedule),
            "disagreement_schedule_ids": [object_id(features) for features in disagreement_schedule],
            "agreement_schedule_ids": [object_id(features) for features in agreement_schedule],
            "generation_attempt": attempt,
        }

        records: list[dict[str, Any]] = []
        for history_type, prefix_examples in [
            ("fresh", fresh_prefix_examples),
            ("committed", committed_prefix_examples),
        ]:
            records.append(
                {
                    **shared_base,
                    "example_id": f"{shared_base['pair_id']}__{history_type}",
                    "history_type": history_type,
                    "prefix_examples": deepcopy(prefix_examples),
                    "prefix_object_ids": [example["object_id"] for example in prefix_examples],
                    "late_examples_by_dose": deepcopy(late_examples_by_dose),
                    "late_object_ids_by_dose": deepcopy(late_object_ids_by_dose),
                    "valid_main_locked": True,
                }
            )

        pair_checks = _pair_checks(records[0], records[1])
        for record in records:
            record["pair_checks"] = pair_checks
        return records, pair_checks

    raise RuntimeError(f"Failed to generate RC-ICL record after {max_attempts} attempts.")


def generate_rcicl_pairs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["seed"])
    splits: dict[str, int] = config["splits"]
    margins = config["cells"]["margins"]
    history_depths = config["cells"]["history_depths"]

    records: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for split, count in splits.items():
        for m in margins:
            for k in history_depths:
                for world_index in range(count):
                    pair_records, pair_checks = _build_pair_examples(seed, split, m, k, world_index)
                    records.extend(pair_records)
                    pair_summaries.append(
                        {
                            "pair_id": pair_records[0]["pair_id"],
                            "split": split,
                            "m": m,
                            "k": k,
                            "pair_checks": pair_checks,
                        }
                    )
    return records, pair_summaries


def _sanity_examples_for_rule(
    incumbent_rule_id: str,
    challenger_rule_id: str,
    *,
    query_features: dict[str, str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    positive_pool = [
        features
        for features in OBJECT_UNIVERSE
        if evaluate_rule(incumbent_rule_id, features) == "dax" and object_id(features) != object_id(query_features)
    ]
    negative_pool = [
        features
        for features in OBJECT_UNIVERSE
        if evaluate_rule(incumbent_rule_id, features) == "wug" and object_id(features) != object_id(query_features)
    ]
    disagreement_pool = [
        features
        for features in OBJECT_UNIVERSE
        if evaluate_rule(incumbent_rule_id, features) != evaluate_rule(challenger_rule_id, features)
        and object_id(features) != object_id(query_features)
    ]
    disagreement_ids = {object_id(features) for features in disagreement_pool}

    rng.shuffle(positive_pool)
    rng.shuffle(negative_pool)
    prefix_objects = positive_pool[:3] + negative_pool[:3]
    if len(prefix_objects) < 6:
        raise RuntimeError("RC-ICL sanity pool unexpectedly too small.")
    if sum(1 for features in prefix_objects if object_id(features) in disagreement_ids) < 2:
        extra_candidates = [features for features in disagreement_pool if object_id(features) not in {object_id(obj) for obj in prefix_objects}]
        while sum(1 for features in prefix_objects if object_id(features) in disagreement_ids) < 2 and extra_candidates:
            prefix_objects[-1] = extra_candidates.pop()

    return [_example_from_object(features, rule_id=incumbent_rule_id) for features in prefix_objects]


def generate_rcicl_sanity(config: dict[str, Any]) -> list[dict[str, Any]]:
    sanity = config["sanity"]
    split = sanity.get("split", "dev")
    count_per_rule_pair = int(sanity["count_per_rule_pair"])
    seed = int(config["seed"])

    records: list[dict[str, Any]] = []
    for pair_index, (incumbent_rule_id, challenger_rule_id) in enumerate(ELIGIBLE_RULE_PAIRS):
        disagreement_raw, _ = _disagreement_and_agreement_objects(incumbent_rule_id, challenger_rule_id)
        disagreement_schedule = _shuffle_objects(
            seed,
            split,
            0,
            0,
            pair_index,
            "sanity_disagreement",
            (incumbent_rule_id, challenger_rule_id),
            disagreement_raw,
        )
        query_by_label: dict[str, list[dict[str, str]]] = {"dax": [], "wug": []}
        for features in disagreement_schedule:
            query_by_label[evaluate_rule(incumbent_rule_id, features)].append(features)

        for example_index in range(count_per_rule_pair):
            target_label = RCICL_OPTION_LABELS[(pair_index * count_per_rule_pair + example_index) % len(RCICL_OPTION_LABELS)]
            if not query_by_label[target_label]:
                raise RuntimeError(f"No RC-ICL sanity query available for label={target_label}.")
            query_features = query_by_label[target_label][example_index % len(query_by_label[target_label])]
            rng = _rng(seed, split, incumbent_rule_id, challenger_rule_id, "sanity", example_index)
            prefix_examples = _sanity_examples_for_rule(
                incumbent_rule_id,
                challenger_rule_id,
                query_features=query_features,
                rng=rng,
            )

            record_id = f"rcicl_sanity__{split}__pair{pair_index:02d}__example{example_index:02d}__seed{seed}"
            records.append(
                {
                    "task_family": "rcicl",
                    "task_variant": "sanity",
                    "example_id": record_id,
                    "world_id": record_id,
                    "pair_id": record_id,
                    "split": split,
                    "m": 0,
                    "k": 0,
                    "history_type": "sanity",
                    "incumbent": incumbent_rule_id,
                    "challenger": challenger_rule_id,
                    "incumbent_rule_id": incumbent_rule_id,
                    "challenger_rule_id": challenger_rule_id,
                    "incumbent_option": target_label,
                    "challenger_option": RCICL_OPTION_LABELS[0] if target_label == RCICL_OPTION_LABELS[1] else RCICL_OPTION_LABELS[1],
                    "expected_option": target_label,
                    "distractors": [],
                    "late_distractor_cycle": [],
                    "prefix_examples": prefix_examples,
                    "prefix_object_ids": [example["object_id"] for example in prefix_examples],
                    "query_object": _object_with_id(query_features),
                    "late_examples_by_dose": {},
                    "late_object_ids_by_dose": {},
                    "disagreement_size": len(disagreement_schedule),
                    "agreement_size": None,
                    "disagreement_schedule_ids": [object_id(features) for features in disagreement_schedule],
                    "agreement_schedule_ids": [],
                    "pair_checks": {},
                }
            )
    return records
