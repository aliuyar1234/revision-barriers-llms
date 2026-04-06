from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_for_csv(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: sanitize_for_csv(row.get(name)) for name in fieldnames})


def main() -> None:
    qwen_summary = read_json(ROOT / "outputs/behavior/m3_qwen/E2__qwen25_7b__mhst__summary__seed0.json")
    companion_summary = read_json(ROOT / "outputs/behavior/m7_qwen/E2__qwen25_7b__mhst__companion_mean_flip_summary__seed0.json")
    llama_summary = read_json(ROOT / "outputs/behavior/m3_llama/E3__llama31_8b__mhst__summary__seed0.json")
    rcicl_summary = read_json(ROOT / "outputs/behavior/m6_qwen_rcicl/E6__qwen25_7b__rcicl__summary__seed0.json")
    rcicl_sanity = read_json(ROOT / "outputs/behavior/m6_qwen_rcicl/E6__qwen25_7b__rcicl__sanity_summary__seed0.json")
    probe_original = read_json(ROOT / "outputs/probes/E4__qwen25_7b__mhst__selected_probe__seed0.json")
    probe_rerun = read_json(ROOT / "outputs/probes/E4r1__qwen25_7b__mhst__selected_probe__seed0.json")
    baseline_rerun = read_json(ROOT / "outputs/probes/E4r1__qwen25_7b__mhst__baseline_metrics__seed0.json")
    m5_original = read_json(ROOT / "outputs/interventions/E5__qwen25_7b__mhst__summary__seed0.json")
    m5_rerun = read_json(ROOT / "outputs/interventions/E5r2__qwen25_7b__mhst__summary__seed0.json")

    test_overall = qwen_summary["by_split"]["test"]["overall"]
    output_margin = qwen_summary["output_margin_control"]

    main_behavior_rows = [
        {
            "row": "MHST test paired barrier",
            "pair_count": test_overall["valid_pair_count"],
            "mean_deltaB_pair_cap": test_overall["pair_delta_summary"]["mean_deltaB_pair_cap"],
            "delta_ci_low": test_overall["pair_delta_bootstrap"]["ci"]["low"],
            "delta_ci_high": test_overall["pair_delta_bootstrap"]["ci"]["high"],
            "notes": "primary behavioral headline",
        },
        {
            "row": "MHST test S_lead paired barrier",
            "pair_count": test_overall["S_lead_valid_pair_count"],
            "mean_deltaB_pair_cap": test_overall["S_lead_delta_summary"]["mean_deltaB_pair_cap"],
            "delta_ci_low": test_overall["S_lead_delta_bootstrap"]["ci"]["low"],
            "delta_ci_high": test_overall["S_lead_delta_bootstrap"]["ci"]["high"],
            "notes": "leader-consistent subset only",
        },
        {
            "row": "MHST test companion mean flipability",
            "pair_count": companion_summary["overall"]["pair_count"],
            "mean_deltaB_pair_cap": companion_summary["overall"]["mean_deltaMeanFlip_pair"],
            "delta_ci_low": companion_summary["overall"]["delta_bootstrap"]["ci"]["low"],
            "delta_ci_high": companion_summary["overall"]["delta_bootstrap"]["ci"]["high"],
            "notes": "fresh minus committed mean flipability over doses",
        },
        {
            "row": "MHST test companion mean flipability S_lead",
            "pair_count": companion_summary["S_lead"]["pair_count"],
            "mean_deltaB_pair_cap": companion_summary["S_lead"]["mean_deltaMeanFlip_pair"],
            "delta_ci_low": companion_summary["S_lead"]["delta_bootstrap"]["ci"]["low"],
            "delta_ci_high": companion_summary["S_lead"]["delta_bootstrap"]["ci"]["high"],
            "notes": "leader-consistent subset under companion metric",
        },
        {
            "row": "Output-margin control",
            "pair_count": "",
            "mean_deltaB_pair_cap": "",
            "delta_ci_low": "",
            "delta_ci_high": "",
            "notes": f"scheme={output_margin['chosen_scheme']} satisfied={output_margin['satisfies_mandatory_control']}",
        },
    ]

    latent_causal_rows = [
        {
            "row": "Original M4 probe",
            "selected_layer": probe_original["selected_layer"],
            "selected_C": probe_original["selected_C"],
            "test_auroc": probe_original["test_auroc"],
            "baseline_o": baseline_rerun["baselines"]["o"]["test_auroc"],
            "baseline_m_o": baseline_rerun["baselines"]["m_o"]["test_auroc"],
            "delta_vs_o": probe_original["delta_auroc_vs_o"]["mean_delta"],
            "delta_vs_m_o": probe_original["delta_auroc_vs_m_o"]["mean_delta"],
            "selected_alpha": "",
            "c4_assessment": "",
            "notes": "probe loses to output-margin baselines overall",
        },
        {
            "row": "Rerun M4 probe",
            "selected_layer": probe_rerun["selected_layer"],
            "selected_C": probe_rerun["selected_C"],
            "test_auroc": probe_rerun["test_auroc"],
            "baseline_o": baseline_rerun["baselines"]["o"]["test_auroc"],
            "baseline_m_o": baseline_rerun["baselines"]["m_o"]["test_auroc"],
            "delta_vs_o": probe_rerun["delta_auroc_vs_o"]["mean_delta"],
            "delta_vs_m_o": probe_rerun["delta_auroc_vs_m_o"]["mean_delta"],
            "selected_alpha": "",
            "c4_assessment": "",
            "notes": "boundary-aligned rerun changes layer but not the overall conclusion",
        },
        {
            "row": "Original M5 intervention",
            "selected_layer": m5_original["selected_layer"],
            "selected_C": "",
            "test_auroc": "",
            "baseline_o": "",
            "baseline_m_o": "",
            "delta_vs_o": "",
            "delta_vs_m_o": "",
            "selected_alpha": m5_original["selected_alpha"],
            "c4_assessment": m5_original["c4_assessment"],
            "notes": "locked one-site steering stays null",
        },
        {
            "row": "Rerun M5 intervention",
            "selected_layer": m5_rerun["selected_layer"],
            "selected_C": "",
            "test_auroc": "",
            "baseline_o": "",
            "baseline_m_o": "",
            "delta_vs_o": "",
            "delta_vs_m_o": "",
            "selected_alpha": m5_rerun["selected_alpha"],
            "c4_assessment": m5_rerun["c4_assessment"],
            "notes": "boundary-alignment rerun remains null",
        },
    ]

    scope_rows = [
        {
            "setting": "MHST / Qwen2.5-7B",
            "mean_deltaB_pair_cap": qwen_summary["by_split"]["test"]["overall"]["pair_delta_summary"]["mean_deltaB_pair_cap"],
            "ci_low": qwen_summary["by_split"]["test"]["overall"]["pair_delta_bootstrap"]["ci"]["low"],
            "ci_high": qwen_summary["by_split"]["test"]["overall"]["pair_delta_bootstrap"]["ci"]["high"],
            "valid_pair_count": qwen_summary["by_split"]["test"]["overall"]["valid_pair_count"],
            "cleanliness_note": "primary setting; no-conflict sanity passes",
        },
        {
            "setting": "MHST / Llama-3.1-8B",
            "mean_deltaB_pair_cap": llama_summary["by_split"]["test"]["overall"]["pair_delta_summary"]["mean_deltaB_pair_cap"],
            "ci_low": llama_summary["by_split"]["test"]["overall"]["pair_delta_bootstrap"]["ci"]["low"],
            "ci_high": llama_summary["by_split"]["test"]["overall"]["pair_delta_bootstrap"]["ci"]["high"],
            "valid_pair_count": llama_summary["by_split"]["test"]["overall"]["valid_pair_count"],
            "cleanliness_note": "same-sign but lower coverage",
        },
        {
            "setting": "RC-ICL / Qwen2.5-7B",
            "mean_deltaB_pair_cap": rcicl_summary["mean_deltaB_pair_cap"],
            "ci_low": rcicl_summary["by_split"]["test"]["overall"]["pair_delta_bootstrap"]["ci"]["low"],
            "ci_high": rcicl_summary["by_split"]["test"]["overall"]["pair_delta_bootstrap"]["ci"]["high"],
            "valid_pair_count": rcicl_summary["valid_pair_count"],
            "cleanliness_note": f"same-sign but no-conflict sanity fails ({rcicl_sanity['overall_accuracy']:.2f} < 0.95)",
        },
    ]

    output_dir = ROOT / "paper" / "tables"
    write_csv(
        output_dir / "table_1_main_behavior_summary.csv",
        main_behavior_rows,
        ["row", "pair_count", "mean_deltaB_pair_cap", "delta_ci_low", "delta_ci_high", "notes"],
    )
    write_csv(
        output_dir / "table_2_latent_causal_summary.csv",
        latent_causal_rows,
        [
            "row",
            "selected_layer",
            "selected_C",
            "test_auroc",
            "baseline_o",
            "baseline_m_o",
            "delta_vs_o",
            "delta_vs_m_o",
            "selected_alpha",
            "c4_assessment",
            "notes",
        ],
    )
    write_csv(
        output_dir / "table_3_scope_summary.csv",
        scope_rows,
        ["setting", "mean_deltaB_pair_cap", "ci_low", "ci_high", "valid_pair_count", "cleanliness_note"],
    )


if __name__ == "__main__":
    main()
