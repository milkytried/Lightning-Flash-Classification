"""Create the Version 2 Phase 3 preflight gate report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torchvision

if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v2_phase3_common import environment_record, now_iso, read_yaml, sha256_file, write_json


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/v2_full.yaml"))
    parser.add_argument("--validation-json", type=Path, default=Path("report/V2_FULL_DATASET_VALIDATION.json"))
    parser.add_argument("--decision-json", type=Path, default=Path("report/V2_PHASE2_DECISION.json"))
    args = parser.parse_args()

    config = read_yaml(args.config)
    validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
    decision = json.loads(args.decision_json.read_text(encoding="utf-8"))
    full = config["full_build"]
    manifest_path = Path(full["outputs"]["manifest"])
    ledger_path = Path(full["outputs"]["frame_ledger"])
    natural_path = Path(full["outputs"]["natural_manifest"])
    manifest = pd.read_csv(manifest_path)
    natural = pd.read_csv(natural_path)

    checks = {
        "primary_contamination_zero": validation["contamination"]["primary_minus20_plus30"] == 0,
        "cross_split_overlaps_zero": all(value == 0 for key, value in validation["overlaps"].items() if key != "within_split_duplicate_crop"),
        "invalid_patch_counts_zero": all(value == 0 for value in validation["patch_audit"].values()),
        "deterministic_manifest_regeneration_passed": decision["criteria"]["manifest_regeneration_deterministic"],
        "validation_and_test_support_passed": decision["criteria"]["sufficient_holdout_samples"],
        "natural_prevalence_set_exists": natural_path.exists() and len(natural) > 0 and decision["criteria"]["natural_prevalence_created"],
        "latlon_rf_roc_auc_below_0_75": decision["geographic_rf_test_roc_auc"] < 0.75,
        "geography_at_least_0_05_below_best_image": decision["geographic_auc_gap_image_minus_geo"] >= 0.05,
        "clustered_intervals_no_severe_instability": decision["criteria"]["cluster_intervals_stable"],
        "all_phase2_tests_passed": decision["criteria"]["all_tests_pass"],
    }
    passed = all(checks.values())
    env = environment_record()
    env["torchvision"] = torchvision.__version__
    hashes = {
        "configuration_hash": full["configuration_hash"],
        "v2_full_config_sha256": sha256_file(args.config),
        "mmd_inventory_hash": sha256_file(config["outputs"]["inventory_csv"]),
        "frame_ledger_hash": sha256_file(ledger_path),
        "controlled_manifest_hash": sha256_file(manifest_path),
        "natural_prevalence_manifest_hash": sha256_file(natural_path),
        "source_code_commit": git_commit(),
    }
    summary = {
        "created_at_utc": now_iso(),
        "passed": passed,
        "checks": checks,
        "hashes": hashes,
        "environment": env,
        "controlled_manifest_rows": int(len(manifest)),
        "natural_prevalence_rows": int(len(natural)),
        "phase2_decision": decision["decision"],
        "phase2_key_values": {
            "geographic_rf_test_roc_auc": decision["geographic_rf_test_roc_auc"],
            "best_image_test_roc_auc": decision["best_image_test_roc_auc"],
            "geographic_auc_gap_image_minus_geo": decision["geographic_auc_gap_image_minus_geo"],
            "primary_contamination": validation["contamination"]["primary_minus20_plus30"],
            "patch_audit": validation["patch_audit"],
            "overlaps": validation["overlaps"],
        },
    }
    write_json("report/V2_PHASE3_PREFLIGHT.json", summary)

    rows = [{"condition": key, "pass": value} for key, value in checks.items()]
    md = [
        "# V2 Phase 3 Preflight Gate",
        "",
        f"Created: `{summary['created_at_utc']}`",
        "",
        f"Gate result: `{'PASS' if passed else 'FAIL'}`",
        "",
        "## Go/No-Go Conditions",
        "",
        markdown_table(rows),
        "",
        "## Frozen Input Hashes",
        "",
        markdown_table([{"item": key, "sha256_or_value": value} for key, value in hashes.items()]),
        "",
        "## Environment",
        "",
        markdown_table([{"item": key, "value": value} for key, value in env.items()]),
        "",
        "No Version 1, Phase 1A, Phase 1B, or Phase 2 artifacts were modified by this preflight check.",
    ]
    Path("report/V2_PHASE3_PREFLIGHT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Phase 3 preflight gate failed; training is not allowed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

