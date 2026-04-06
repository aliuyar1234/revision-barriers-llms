from __future__ import annotations

import hashlib
import random
from collections import Counter
from copy import deepcopy
from typing import Any

CATEGORY_ORDER = ["vehicle", "clothing", "job", "schedule", "hobby"]
OPTION_LABELS = ["Option A", "Option B", "Option C", "Option D"]
LATE_CATEGORY_ORDER = ["job", "vehicle", "hobby", "clothing", "schedule", "job"]
CHALLENGER_POSITION_ORDER = [2, 0, 4, 1, 5, 3]
SUSPECT_IDS = ["s0", "s1", "s2", "s3"]

CATEGORY_BANKS: dict[str, list[str]] = {
    "vehicle": [
        "bicycle",
        "scooter",
        "van",
        "taxi",
        "motorcycle",
        "pickup truck",
        "hatchback",
        "minibus",
        "tram pass",
        "subway card",
        "cargo bike",
        "station wagon",
    ],
    "clothing": [
        "red scarf",
        "blue cap",
        "green jacket",
        "yellow gloves",
        "black boots",
        "white hoodie",
        "orange tie",
        "gray sweater",
        "purple bandana",
        "brown coat",
        "denim vest",
        "wool hat",
    ],
    "job": [
        "locksmith",
        "florist",
        "paramedic",
        "teacher",
        "baker",
        "carpenter",
        "pharmacist",
        "electrician",
        "librarian",
        "mechanic",
        "tailor",
        "accountant",
    ],
    "schedule": [
        "night shift",
        "day shift",
        "evening shift",
        "weekend shift",
        "dawn shift",
        "midday shift",
        "split shift",
        "holiday rotation",
        "Monday rota",
        "Thursday rota",
        "early shift",
        "late shift",
    ],
    "hobby": [
        "chess club",
        "choir",
        "climbing gym",
        "running group",
        "pottery class",
        "book club",
        "painting class",
        "debate club",
        "gardening club",
        "yoga studio",
        "film society",
        "photography club",
    ],
}

TEMPLATE_BANKS: dict[str, list[str]] = {
    "vehicle": [
        "Witnesses mention {value}.",
        "A note references {value}.",
        "The travel record lists {value}.",
    ],
    "clothing": [
        "A {value} was recovered.",
        "The person was seen with a {value}.",
        "A report mentions {value}.",
    ],
    "job": [
        "Records fit a {value}.",
        "Employment files point to a {value}.",
        "The work record matches a {value}.",
    ],
    "schedule": [
        "The person appears on the {value} roster.",
        "The schedule matches the {value}.",
        "A staffing note lists the {value}.",
    ],
    "hobby": [
        "A {value} flyer was found in the bag.",
        "A note mentions the {value}.",
        "A membership card points to the {value}.",
    ],
}


def stable_seed(*parts: Any) -> int:
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _rng(*parts: Any) -> random.Random:
    return random.Random(stable_seed(*parts))


def _option_rank(option_label: str) -> int:
    return OPTION_LABELS.index(option_label)


def _sample_world(rng: random.Random) -> dict[str, dict[str, str]]:
    world: dict[str, dict[str, str]] = {suspect_id: {} for suspect_id in SUSPECT_IDS}
    for category in CATEGORY_ORDER:
        sampled_values = rng.sample(CATEGORY_BANKS[category], len(SUSPECT_IDS))
        for suspect_id, value in zip(SUSPECT_IDS, sampled_values, strict=True):
            world[suspect_id][category] = value
    return world


def _sample_option_mapping(rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    shuffled_suspects = rng.sample(SUSPECT_IDS, len(SUSPECT_IDS))
    option_to_suspect = dict(zip(OPTION_LABELS, shuffled_suspects, strict=True))
    suspect_to_option = {suspect: option for option, suspect in option_to_suspect.items()}
    return option_to_suspect, suspect_to_option


def _build_late_distractor_cycle(
    suspect_to_option: dict[str, str],
    distractors: list[str],
) -> list[str]:
    return sorted(distractors, key=lambda suspect_id: _option_rank(suspect_to_option[suspect_id]))


def _render_clue_sequence(
    world: dict[str, dict[str, str]],
    slots: list[tuple[str, str]],
    suspect_to_option: dict[str, str],
    encounter_counts: Counter[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]]]:
    counts = Counter() if encounter_counts is None else Counter(encounter_counts)
    rendered: list[dict[str, Any]] = []
    for index, (target, category) in enumerate(slots, start=1):
        key = (target, category)
        template_id = counts[key]
        if template_id >= len(TEMPLATE_BANKS[category]):
            raise ValueError(f"Too many repeats for {key}; expected at most 3 occurrences in one prompt.")
        counts[key] += 1
        value = world[target][category]
        rendered.append(
            {
                "index": index,
                "target": target,
                "option_label": suspect_to_option[target],
                "category": category,
                "value": value,
                "template_id": template_id,
                "text": TEMPLATE_BANKS[category][template_id].format(value=value),
            }
        )
    return rendered, counts


def _late_targets(challenger: str, late_distractor_cycle: list[str], dose: int) -> list[str]:
    challenger_positions = set(CHALLENGER_POSITION_ORDER[:dose])
    targets: list[str] = []
    distractor_index = 0
    for position in range(len(LATE_CATEGORY_ORDER)):
        if position in challenger_positions:
            targets.append(challenger)
        else:
            targets.append(late_distractor_cycle[distractor_index % len(late_distractor_cycle)])
            distractor_index += 1
    return targets


def _support_counts(prefix_slots: list[tuple[str, str]], late_targets: list[str]) -> dict[str, int]:
    counts = {suspect_id: 0 for suspect_id in SUSPECT_IDS}
    for target, _ in prefix_slots:
        counts[target] += 1
    for target in late_targets:
        counts[target] += 1
    return counts


def _build_visible_profiles(
    world: dict[str, dict[str, str]],
    option_to_suspect: dict[str, str],
) -> dict[str, dict[str, str]]:
    return {option_label: world[suspect_id] for option_label, suspect_id in option_to_suspect.items()}


def _pair_checks(
    fresh: dict[str, Any],
    committed: dict[str, Any],
    late_distractor_cycle: list[str],
) -> dict[str, bool]:
    fresh_prefix_targets = fresh["prefix_slot_targets"]
    committed_prefix_targets = committed["prefix_slot_targets"]
    return {
        "same_incumbent": fresh["incumbent"] == committed["incumbent"],
        "same_challenger": fresh["challenger"] == committed["challenger"],
        "same_design_margin": fresh["m"] == committed["m"],
        "same_counts": Counter(fresh_prefix_targets) == Counter(committed_prefix_targets),
        "same_slot_structure": len(fresh["prefix_clues"]) == len(committed["prefix_clues"]),
        "same_late_packet_structure": (
            fresh["late_categories"] == committed["late_categories"]
            and fresh["late_distractor_cycle"] == committed["late_distractor_cycle"] == late_distractor_cycle
        ),
    }


def _build_pair_examples(
    seed: int,
    split: str,
    m: int,
    k: int,
    world_index: int,
    sanity_mode: bool = False,
    expected_option: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_attempts = 512
    for attempt in range(max_attempts):
        rng = _rng(seed, split, m, k, world_index, "sanity" if sanity_mode else "pair", attempt)
        world = _sample_world(rng)
        option_to_suspect, suspect_to_option = _sample_option_mapping(rng)

        if sanity_mode:
            winner = rng.choice(SUSPECT_IDS)
            if expected_option is not None:
                current_label = suspect_to_option[winner]
                swapped = option_to_suspect[expected_option]
                option_to_suspect[current_label] = swapped
                option_to_suspect[expected_option] = winner
                suspect_to_option = {suspect: option for option, suspect in option_to_suspect.items()}

            other_suspects = [suspect_id for suspect_id in SUSPECT_IDS if suspect_id != winner]
            winner_categories = rng.sample(CATEGORY_ORDER, 4)
            other_categories = rng.sample(CATEGORY_ORDER, len(other_suspects))
            prefix_slots = [(winner, category) for category in winner_categories] + [
                (suspect_id, category) for suspect_id, category in zip(other_suspects, other_categories, strict=True)
            ]
            prefix_clues, _ = _render_clue_sequence(world, prefix_slots, suspect_to_option)
            example = {
                "task_family": "mhst",
                "task_variant": "sanity",
                "example_id": f"mhst_sanity__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
                "world_id": f"mhst_sanity_world__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
                "pair_id": f"mhst_sanity__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
                "split": split,
                "m": m,
                "k": k,
                "history_type": "sanity",
                "incumbent": winner,
                "challenger": winner,
                "distractors": other_suspects,
                "incumbent_option": suspect_to_option[winner],
                "challenger_option": suspect_to_option[winner],
                "expected_option": suspect_to_option[winner],
                "option_to_suspect": option_to_suspect,
                "suspect_to_option": suspect_to_option,
                "visible_profiles": _build_visible_profiles(world, option_to_suspect),
                "prefix_clues": prefix_clues,
                "prefix_slot_targets": [target for target, _ in prefix_slots],
                "prefix_slot_categories": [category for _, category in prefix_slots],
                "prefix_counts": _support_counts(prefix_slots, []),
                "late_categories": [],
                "late_slot_targets_by_dose": {},
                "late_clues_by_dose": {},
                "late_distractor_cycle": [],
                "pair_checks": {},
                "generation_attempt": attempt,
            }
            return [example], {}

        incumbent, challenger = rng.sample(SUSPECT_IDS, 2)
        distractors = [suspect_id for suspect_id in SUSPECT_IDS if suspect_id not in {incumbent, challenger}]
        late_distractor_cycle = _build_late_distractor_cycle(suspect_to_option, distractors)

        incumbent_categories = rng.sample(CATEGORY_ORDER, m + k)
        challenger_categories = rng.sample(CATEGORY_ORDER, k)
        distractor_categories = rng.sample(CATEGORY_ORDER, len(distractors))

        prefix_distractor_slots = list(zip(distractors, distractor_categories, strict=True))
        fresh_prefix_slots = [(challenger, category) for category in challenger_categories] + [
            (incumbent, category) for category in incumbent_categories
        ] + prefix_distractor_slots
        committed_prefix_slots = [(incumbent, category) for category in incumbent_categories] + [
            (challenger, category) for category in challenger_categories
        ] + prefix_distractor_slots

        fresh_prefix_clues, fresh_prefix_counts = _render_clue_sequence(world, fresh_prefix_slots, suspect_to_option)
        committed_prefix_clues, committed_prefix_counts = _render_clue_sequence(
            world,
            committed_prefix_slots,
            suspect_to_option,
        )

        prefix_counts = _support_counts(fresh_prefix_slots, [])
        max_dose_counts = _support_counts(fresh_prefix_slots, [challenger] * len(LATE_CATEGORY_ORDER))
        if any(prefix_counts[distractor] > prefix_counts[incumbent] for distractor in distractors):
            continue
        if max_dose_counts[challenger] <= max(
            max_dose_counts[incumbent],
            *(max_dose_counts[distractor] for distractor in distractors),
        ):
            continue

        visible_profiles = _build_visible_profiles(world, option_to_suspect)
        shared_base = {
            "task_family": "mhst",
            "task_variant": "pair",
            "world_id": f"mhst_world__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
            "pair_id": f"mhst_pair__{split}__m{m}__k{k}__{world_index:04d}__seed{seed}",
            "split": split,
            "m": m,
            "k": k,
            "incumbent": incumbent,
            "challenger": challenger,
            "distractors": distractors,
            "incumbent_option": suspect_to_option[incumbent],
            "challenger_option": suspect_to_option[challenger],
            "option_to_suspect": option_to_suspect,
            "suspect_to_option": suspect_to_option,
            "visible_profiles": visible_profiles,
            "late_categories": deepcopy(LATE_CATEGORY_ORDER),
            "late_distractor_cycle": late_distractor_cycle,
            "generation_attempt": attempt,
        }

        records: list[dict[str, Any]] = []
        for history_type, prefix_slots, prefix_clues, prefix_counter in [
            ("fresh", fresh_prefix_slots, fresh_prefix_clues, fresh_prefix_counts),
            ("committed", committed_prefix_slots, committed_prefix_clues, committed_prefix_counts),
        ]:
            late_slot_targets_by_dose: dict[str, list[str]] = {}
            late_clues_by_dose: dict[str, list[dict[str, Any]]] = {}
            support_counts_by_dose: dict[str, dict[str, int]] = {}
            for dose in range(len(LATE_CATEGORY_ORDER) + 1):
                late_targets = _late_targets(challenger, late_distractor_cycle, dose)
                late_slots = list(zip(late_targets, LATE_CATEGORY_ORDER, strict=True))
                late_clues, _ = _render_clue_sequence(
                    world,
                    late_slots,
                    suspect_to_option,
                    encounter_counts=prefix_counter,
                )
                late_slot_targets_by_dose[str(dose)] = late_targets
                late_clues_by_dose[str(dose)] = late_clues
                support_counts_by_dose[str(dose)] = _support_counts(prefix_slots, late_targets)

            records.append(
                {
                    **shared_base,
                    "example_id": f"{shared_base['pair_id']}__{history_type}",
                    "history_type": history_type,
                    "prefix_clues": prefix_clues,
                    "prefix_slot_targets": [target for target, _ in prefix_slots],
                    "prefix_slot_categories": [category for _, category in prefix_slots],
                    "prefix_counts": _support_counts(prefix_slots, []),
                    "late_slot_targets_by_dose": late_slot_targets_by_dose,
                    "late_clues_by_dose": late_clues_by_dose,
                    "support_counts_by_dose": support_counts_by_dose,
                }
            )

        pair_checks = _pair_checks(records[0], records[1], late_distractor_cycle)
        for record in records:
            record["pair_checks"] = pair_checks
        return records, pair_checks

    raise RuntimeError(f"Failed to generate MHST record after {max_attempts} attempts.")


def generate_mhst_pairs(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def generate_mhst_sanity(config: dict[str, Any]) -> list[dict[str, Any]]:
    sanity = config["sanity"]
    split = sanity.get("split", "dev")
    count_per_label = int(sanity["count_per_label"])
    m = int(sanity.get("m", 3))
    k = int(sanity.get("k", 0))
    seed = int(config["seed"])

    records: list[dict[str, Any]] = []
    for label_index, expected_option in enumerate(OPTION_LABELS):
        for example_index in range(count_per_label):
            world_index = label_index * count_per_label + example_index
            examples, _ = _build_pair_examples(
                seed,
                split,
                m,
                k,
                world_index,
                sanity_mode=True,
                expected_option=expected_option,
            )
            records.extend(examples)
    return records
