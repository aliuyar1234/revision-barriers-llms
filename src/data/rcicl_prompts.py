from __future__ import annotations

from typing import Any

RCICL_CANONICAL_ANSWERS = [" dax", " wug"]

PROMPT_PREAMBLE = "\n".join(
    [
        "You are tracking which simple labeling rule is currently best supported.",
        "",
        "There are six candidate rules:",
        "1. dax iff color = red",
        "2. dax iff color = blue",
        "3. dax iff shape = circle",
        "4. dax iff shape = triangle",
        "5. dax iff size = small",
        "6. dax iff size = large",
        "Texture never matters.",
        "",
        "Each example is evidence for one of these rules.",
        "The evidence may be conflicting, so do not assume that one rule fits every example perfectly.",
        "Use all examples and answer for the query object using the rule that is currently best supported.",
        "Respond with exactly one of: dax, wug.",
    ]
)


def _render_object(features: dict[str, str]) -> str:
    return (
        f"color={features['color']}; "
        f"shape={features['shape']}; "
        f"size={features['size']}; "
        f"texture={features['texture']}"
    )


def _render_examples_block(header: str, examples: list[dict[str, Any]]) -> str:
    lines = [header]
    for index, example in enumerate(examples, start=1):
        lines.append(f"{index}. {_render_object(example['features'])} -> {example['label']}")
    return "\n".join(lines)


def _render_query_block(query_object: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Query object",
            _render_object(query_object["features"]),
        ]
    )


def render_rcicl_full_prompt_prefix(record: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            PROMPT_PREAMBLE,
            _render_examples_block("Examples so far", record["prefix_examples"]),
            _render_query_block(record["query_object"]),
        ]
    )


def render_rcicl_prefix_prompt(record: dict[str, Any]) -> str:
    return render_rcicl_full_prompt_prefix(record) + "\n\nFinal answer:"


def render_rcicl_full_prompt(record: dict[str, Any], dose: int) -> str:
    later_examples = record["late_examples_by_dose"][str(dose)]
    return (
        render_rcicl_full_prompt_prefix(record)
        + "\n\n"
        + _render_examples_block("Later examples", later_examples)
        + "\n\nFinal answer:"
    )


def render_rcicl_prompt_samples(record: dict[str, Any]) -> dict[str, str]:
    samples = {
        "prefix_prompt": render_rcicl_prefix_prompt(record),
        "prefix_boundary_partial": render_rcicl_full_prompt_prefix(record),
    }
    if record.get("late_examples_by_dose"):
        samples["full_prompt_dose_0"] = render_rcicl_full_prompt(record, 0)
        samples["full_prompt_dose_3"] = render_rcicl_full_prompt(record, 3)
        samples["full_prompt_dose_6"] = render_rcicl_full_prompt(record, 6)
    return samples
