from __future__ import annotations

from typing import Any

from src.data.mhst_worlds import OPTION_LABELS

MHST_CANONICAL_ANSWERS = [f" {option_label}" for option_label in OPTION_LABELS]

PROMPT_PREAMBLE = "\n".join(
    [
        "You are tracking which suspect is currently most likely.",
        "",
        "Each clue points to exactly one suspect profile.",
        "Use all clues.",
        "Respond with exactly one of: Option A, Option B, Option C, Option D.",
    ]
)


def _render_suspects_block(record: dict[str, Any]) -> str:
    lines = ["Suspects"]
    for option_label in OPTION_LABELS:
        profile = record["visible_profiles"][option_label]
        lines.append(
            (
                f"{option_label}: "
                f"vehicle={profile['vehicle']}; "
                f"clothing={profile['clothing']}; "
                f"job={profile['job']}; "
                f"schedule={profile['schedule']}; "
                f"hobby={profile['hobby']}"
            )
        )
    return "\n".join(lines)


def _render_clues_block(header: str, clues: list[dict[str, Any]]) -> str:
    lines = [header]
    for index, clue in enumerate(clues, start=1):
        lines.append(f"{index}. {clue['text']}")
    return "\n".join(lines)


def render_mhst_full_prompt_prefix(record: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            PROMPT_PREAMBLE,
            _render_suspects_block(record),
            _render_clues_block("Clues so far", record["prefix_clues"]),
        ]
    )


def render_mhst_intervention_prefix(record: dict[str, Any]) -> str:
    # Keep the extraction-time boundary token compatible with the full scoring prompt,
    # while still excluding the `Later clues` header itself.
    return render_mhst_full_prompt_prefix(record) + "\n\n"


def render_mhst_prefix_prompt(record: dict[str, Any]) -> str:
    return render_mhst_full_prompt_prefix(record) + "\n\nFinal answer:"


def render_mhst_full_prompt(record: dict[str, Any], dose: int) -> str:
    later_clues = record["late_clues_by_dose"][str(dose)]
    return (
        render_mhst_full_prompt_prefix(record)
        + "\n\n"
        + _render_clues_block("Later clues", later_clues)
        + "\n\nFinal answer:"
    )


def render_mhst_prompt_samples(record: dict[str, Any]) -> dict[str, str]:
    samples = {
        "prefix_prompt": render_mhst_prefix_prompt(record),
        "prefix_boundary_partial": render_mhst_full_prompt_prefix(record),
        "intervention_prefix_boundary": render_mhst_intervention_prefix(record),
    }
    if record.get("late_clues_by_dose"):
        samples["full_prompt_dose_0"] = render_mhst_full_prompt(record, 0)
        samples["full_prompt_dose_3"] = render_mhst_full_prompt(record, 3)
        samples["full_prompt_dose_6"] = render_mhst_full_prompt(record, 6)
    return samples
