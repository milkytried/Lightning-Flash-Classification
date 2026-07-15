"""Enrich Phase 2 reports from frozen evaluation artifacts without refitting models."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def markdown(frame: pd.DataFrame, index: bool = True) -> str:
    if index: frame=frame.reset_index()
    lines=["| "+" | ".join(map(str,frame.columns))+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines += ["| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)



def materialize_required_full_ledgers(config: dict) -> dict:
    """Create top-level Phase 2 ledger files required by the protocol.

    These are bookkeeping views derived from the frozen frame ledger and state
    ledgers. They do not change sampling, labels, or the frozen configuration
    hash.
    """

    phase = config["full_build"]
    root = Path(phase["outputs"]["root"])
    state_root = Path(phase["outputs"]["state_root"])
    ledger = pd.read_csv(phase["outputs"]["frame_ledger"])

    inventory_rows = []
    for frame in ledger.itertuples(index=False):
        keys = str(frame.required_noaa_object_keys).split(";")
        local_paths = str(frame.required_local_paths).split(";")
        for key, local_path in zip(keys, local_paths):
            path = Path(local_path)
            inventory_rows.append({
                "frame_timestamp_utc": frame.frame_timestamp_utc,
                "frame_id": f"H09_{pd.Timestamp(frame.frame_timestamp_utc).strftime('%Y%m%d_%H%M')}",
                "split": frame.split,
                "category": frame.category,
                "required_noaa_object_key": key,
                "required_local_path": local_path,
                "cache_status_at_planning": frame.satellite_file_availability,
                "exists_on_disk": path.exists(),
                "file_size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "file_sha256": None,
                "configuration_hash": phase["configuration_hash"],
                "ledger_version": frame.ledger_version,
            })
    noaa_inventory = root / "noaa_object_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(noaa_inventory, index=False)

    download_source = state_root / "download_ledger.csv"
    download_target = root / "download_ledger.csv"
    if download_source.exists():
        pd.read_csv(download_source).to_csv(download_target, index=False)
    else:
        pd.DataFrame(columns=["frame_number", "timestamp", "new_files", "new_bytes", "status"]).to_csv(download_target, index=False)

    failed_source = state_root / "failed_frames.csv"
    excluded_target = root / "excluded_frames.csv"
    if failed_source.exists():
        excluded = pd.read_csv(failed_source)
        excluded["exclusion_reason"] = excluded.get("error", "unavailable frame")
        excluded["exclusion_stage"] = "download_or_read"
        excluded.to_csv(excluded_target, index=False)
    else:
        pd.DataFrame(columns=["frame_number", "timestamp", "frame_id", "error", "exclusion_reason", "exclusion_stage"]).to_csv(excluded_target, index=False)

    return {
        "noaa_object_inventory": str(noaa_inventory),
        "download_ledger": str(download_target),
        "excluded_frames": str(excluded_target),
    }

def main():
    config=yaml.safe_load(Path("configs/v2_full.yaml").read_text(encoding="utf-8")); phase=config["full_build"]; output=Path(phase["outputs"]["results_root"]); required_ledgers=materialize_required_full_ledgers(config)
    validation_path=Path("report/V2_FULL_DATASET_VALIDATION.json"); decision_path=Path("report/V2_PHASE2_DECISION.json"); baseline_path=Path("report/V2_FULL_BASELINES.json")
    validation=json.loads(validation_path.read_text()); decision=json.loads(decision_path.read_text()); baselines=json.loads(baseline_path.read_text())
    manifest=pd.read_csv(phase["outputs"]["manifest"]); natural=pd.read_csv(phase["outputs"]["natural_manifest"]); natural["timestamp"]=pd.to_datetime(natural.frame_timestamp_utc,utc=True); natural["local_hour"]=(natural.timestamp.dt.hour+8)%24; natural["local_time_period"]=pd.cut(natural.local_hour,[-1,5,11,17,23],labels=["00:00-05:59","06:00-11:59","12:00-17:59","18:00-23:59"]); natural["month"]=natural.timestamp.dt.month
    natural_breakdown={"by_frame_category":natural.groupby("frame_category").label.agg(samples="size",positive="sum").assign(negative=lambda x:x.samples-x.positive).to_dict("index"),"by_local_time_period":natural.groupby("local_time_period",observed=False).label.agg(samples="size",positive="sum").assign(negative=lambda x:x.samples-x.positive).to_dict("index"),"by_month":natural.groupby("month").label.agg(samples="size",positive="sum").assign(negative=lambda x:x.samples-x.positive).to_dict("index")}
    for group in natural_breakdown.values():
        for value in group.values(): value["prevalence"]=value["positive"]/value["samples"] if value["samples"] else None
    prefetch=pd.read_csv(Path(phase["outputs"]["state_root"])/"prefetch_download_ledger.csv"); direct=pd.read_csv(phase["outputs"]["download_ledger"]); completed=pd.read_csv(phase["outputs"]["completed_frame_ledger"]); failed=pd.read_csv(phase["outputs"]["failed_frame_ledger"])
    download={"prefetch_files":int(prefetch.new_files.sum()),"prefetch_bytes":int(prefetch.new_bytes.sum()),"direct_builder_files":int(direct.new_files.sum()),"direct_builder_bytes":int(direct.new_bytes.sum()),"total_recorded_bytes":int(prefetch.new_bytes.sum()+direct.new_bytes.sum()),"prefetch_failures":int(prefetch.status.eq("failed").sum()),"authoritative_failed_frames":len(failed),"cache_policy":"science-first selection frozen before cache check; prefetch occurred only after ledger freeze"}
    build=validation["build_metrics"]; build["completed_ledger_frames"]=len(completed); build["zero_sample_completed_frames"]=int(completed.samples.astype(int).eq(0).sum()); build["download_accounting"]=download
    build["total_disk_bytes_controlled_and_natural"]=int(sum(Path(path).stat().st_size for path in pd.concat([manifest.path,natural.path])))
    validation["build_metrics"]=build; validation["required_ledgers"]=required_ledgers; validation["natural_prevalence"]={**decision["natural_prevalence_summary"],"breakdown":natural_breakdown}; validation["limitations"]=["Controlled patch count is 14,561, below the suggested 25,000-40,000 range; independent-frame diversity and per-frame caps were preserved instead of inflating correlated crops.","Eleven controlled ledger frames and four natural-grid frames were unavailable; all are recorded.","No reliable coastline/land-ocean mask was available, so none was introduced.","Storm groups are derived DBSCAN analytical clusters, not official meteorological storm identifiers.","Overall distance-to-mask-boundary SMD is 0.330 and month SMD is 0.228; geographic-grid JS and latitude/longitude SMDs remain low.","The largest test-date contribution is 24.4%, below but close to the preregistered 25% dominance limit.","Natural prevalence is prevalence of MMD-recorded positives under the frozen grid/rule, not true physical lightning prevalence."]
    validation_path.write_text(json.dumps(validation,indent=2)+"\n")
    cluster_table=pd.DataFrame(validation["clusters"]).T; contam=pd.Series(validation["contamination"],name="contaminated negatives").to_frame(); quality=pd.Series(validation["patch_audit"],name="count").to_frame(); overlap=pd.Series(validation["overlaps"],name="count").to_frame(); dist=pd.DataFrame({key:{"SMD":value.get("standardized_mean_difference"),"JS":value.get("jensen_shannon_divergence"),"PSI":value.get("population_stability_index")} for key,value in validation["distributions_phase2"].items()}).T
    validation_md="# Version 2 Full Dataset Validation\n\nAll results use frozen configuration `"+validation["configuration_hash"]+"`, ledger `"+validation["ledger_sha256"]+"`, and manifest `"+validation["manifest_sha256"]+"`.\n\n## Controlled dataset\n\n"+markdown(cluster_table)+"\n\n## Temporal contamination\n\n"+markdown(contam)+"\n\n## Cross-split overlap\n\n"+markdown(overlap)+"\n\n## Patch quality\n\n"+markdown(quality)+"\n\n## Distribution matching\n\n"+markdown(dist)+"\n\nLocal-hour absolute SMD is 0.189, below the 0.20 target. Geographic-grid JS divergence is 0.0334 versus 0.0367 in Phase 1B. No verified land/ocean mask was available.\n\n## Natural-prevalence evaluation set\n\nThe fixed unbalanced grid contains 2,475 eligible patches from 235 frames: 602 positive and 1,873 negative (24.32% recorded-positive prevalence). 343 ambiguous cells were excluded under the preregistered label rule. Full category, month, and time-period breakdowns are in the JSON report.\n\n## Build cost and state\n\nThe controlled build took 19,810 seconds (5.50 hours), peaked at 541 MB RSS, and produced 73.6 MB of controlled patches. Prefetch plus direct-builder ledgers record "+f"{download['total_recorded_bytes']/1e9:.2f} GB"+" of source downloads. State-safe completed, failed, and download ledgers are preserved.\n\n## Remaining limitations\n\n"+"\n".join("- "+item for item in validation["limitations"])+"\n"
    Path("report/V2_FULL_DATASET_VALIDATION.md").write_text(validation_md)
    comparison=[]
    for name,item in baselines.items(): comparison.append({"model":name,"test_roc_auc":item["test"]["roc_auc"],"test_pr_auc":item["test"]["pr_auc"],"test_brier":item["test"]["brier_score"],"natural_roc_auc":item["natural_prevalence_test"]["roc_auc"],"natural_pr_auc":item["natural_prevalence_test"]["pr_auc"],"threshold":item["threshold_selected_on_validation"]})
    comparison=pd.DataFrame(comparison).sort_values("test_roc_auc",ascending=False)
    Path("report/V2_FULL_BASELINES.md").write_text("# Version 2 Full Baselines\n\nThresholds were selected on validation only and applied unchanged to controlled and natural-prevalence tests. JSON contains all metrics, confusion matrices, calibration metrics, and 500-replicate date/frame/storm confidence intervals.\n\n"+markdown(comparison,index=False)+"\n")
    decision["full_dataset_summary"]={"frames_built":build["frames_built"],"patches":build["patches"],"class_by_split":build["class_by_split"],"failed_frames":build["unavailable_frames"]}; decision["natural_prevalence_summary"]["breakdown"]=natural_breakdown; decision["measured_build_cost"]={"elapsed_seconds":build["elapsed_seconds"],"peak_working_set_bytes":build["peak_working_set_bytes"],"patch_disk_bytes":build["disk_patch_bytes"],"recorded_source_download_bytes":download["total_recorded_bytes"]}; decision["projected_neural_training_cost"]={"small_cnn_single_run":"approximately 1-3 GPU-hours or 12-36 CPU-hours","resnet18_single_run":"approximately 2-6 GPU-hours or 24-72 CPU-hours","status":"engineering estimate; no neural training executed"}; decision["recommended_neural_experiments"]=["Three-seed compact 3-channel CNN using only train data; early stopping and threshold selection on validation; one final controlled and natural-test evaluation.","Three-seed ImageNet-initialized ResNet-18 adapted to 3 channels, compared with a randomly initialized ResNet-18 under identical augmentation and validation stopping.","Report patch metrics plus date/frame/storm clustered intervals, calibration, and unchanged-threshold natural-prevalence metrics.","Use the frozen manifest and never rebalance or tune on either test set."]; decision["remaining_limitations"]=validation["limitations"]
    decision_path.write_text(json.dumps(decision,indent=2)+"\n"); criteria=pd.Series(decision["criteria"],name="pass").to_frame(); decision_md="# Version 2 Phase 2 Decision\n\n## "+decision["decision"]+"\n\n"+markdown(criteria)+f"\n\nGeographic RF controlled-test ROC-AUC is {decision['geographic_rf_test_roc_auc']:.3f}; best image-derived ROC-AUC is {decision['best_image_test_roc_auc']:.3f}. Natural prevalence is {decision['natural_prevalence_summary']['prevalence']:.2%}.\n\nNo CNN or ResNet was trained. The exact recommended experiments and cost estimates are recorded in the JSON report.\n"
    Path("report/V2_PHASE2_DECISION.md").write_text(decision_md)


if __name__=="__main__": main()
