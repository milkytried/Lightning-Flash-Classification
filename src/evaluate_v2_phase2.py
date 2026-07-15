"""Validate and evaluate the frozen V2 Phase 2 pilot (no neural-network training)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex


def markdown(value, index=True):
    frame = value.to_frame() if isinstance(value, pd.Series) else value.copy()
    if index:
        frame = frame.reset_index()
    columns = [str(item) for item in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def metrics(y, score, threshold):
    pred = np.asarray(score) >= threshold
    y = np.asarray(y)
    result = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
              "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
              "f1": f1_score(y, pred, zero_division=0), "mcc": matthews_corrcoef(y, pred),
              "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    result["brier_score"] = float(brier_score_loss(y, score))
    bins=np.linspace(0,1,11); ids=np.clip(np.digitize(score,bins)-1,0,9); result["calibration_error"] = float(sum(np.mean(ids==item)*abs(np.mean(np.asarray(score)[ids==item])-np.mean(y[ids==item])) for item in range(10) if np.any(ids==item)))
    result["roc_auc"] = roc_auc_score(y, score) if np.unique(y).size == 2 else None
    result["pr_auc"] = average_precision_score(y, score) if np.unique(y).size == 2 else None
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in result.items()}


def threshold_by_f1(y, score):
    candidates = np.unique(np.r_[0, score, 1])
    values = np.array([f1_score(y, score >= item, zero_division=0) for item in candidates])
    return float(candidates[np.flatnonzero(values == values.max())[-1]])


def bootstrap(frame, score, threshold, cluster, repeats, seed):
    rng = np.random.default_rng(seed); groups = frame[cluster].astype(str).unique(); draws = []
    for _ in range(repeats):
        chosen = rng.choice(groups, len(groups), replace=True)
        indices = np.concatenate([np.flatnonzero(frame[cluster].astype(str).to_numpy() == item) for item in chosen])
        draws.append(metrics(frame.label.to_numpy()[indices], np.asarray(score)[indices], threshold))
    output = {}
    for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "brier_score", "calibration_error"]:
        values = [item[key] for item in draws if item[key] is not None]
        output[key] = {"lower": float(np.quantile(values, .025)), "median": float(np.quantile(values, .5)),
                       "upper": float(np.quantile(values, .975)), "valid_replicates": len(values)}
    return output


def distribution_stats(frame):
    pos, neg = frame[frame.label.eq(1)], frame[frame.label.eq(0)]; result = {}
    for column in ["centre_lat", "centre_lon", "local_hour", "month", "distance_to_study_mask_boundary_km"]:
        a, b = pos[column].to_numpy(float), neg[column].to_numpy(float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); bins = np.linspace(lo, hi + 1e-9, 11)
        pa, _ = np.histogram(a, bins=bins); pb, _ = np.histogram(b, bins=bins)
        pa = (pa + 1e-6) / (pa.sum() + 1e-5); pb = (pb + 1e-6) / (pb.sum() + 1e-5)
        result[column] = {"positive_mean": float(a.mean()), "negative_mean": float(b.mean()),
                          "standardized_mean_difference": float((a.mean()-b.mean())/pooled) if pooled else 0.0,
                          "jensen_shannon_divergence": float(jensenshannon(pa, pb) ** 2),
                          "population_stability_index": float(np.sum((pa-pb)*np.log(pa/pb))),
                          "ks_statistic": float(ks_2samp(a, b).statistic), "ks_pvalue": float(ks_2samp(a, b).pvalue)}
    grid = pd.crosstab(frame.geographic_grid_cell, frame.label).reindex(columns=[0,1], fill_value=0)
    p = (grid[1].to_numpy()+1e-6); q = (grid[0].to_numpy()+1e-6); p /= p.sum(); q /= q.sum()
    result["geographic_grid_2d"] = {"jensen_shannon_divergence": float(jensenshannon(p, q)**2),
                                    "occupied_cells_positive": int((grid[1]>0).sum()), "occupied_cells_negative": int((grid[0]>0).sum())}
    return result


def overlap_checks(frame):
    checks = {}
    for column in ["date", "frame_id", "storm_id", "path", "sha256"]:
        sets = {split: set(part[column].dropna().astype(str)) for split, part in frame.groupby("split")}
        checks[column] = sum(len(sets[a] & sets[b]) for a,b in [("train","val"),("train","test"),("val","test")])
    source = {split: set(";".join(part.source_himawari_files).split(";")) for split, part in frame.groupby("split")}
    checks["source_file"] = sum(len(source[a]&source[b]) for a,b in [("train","val"),("train","test"),("val","test")])
    crop_key = frame.frame_id.astype(str)+"|"+frame.crop_x0.astype(str)+"|"+frame.crop_y0.astype(str)+"|"+frame.crop_x1.astype(str)+"|"+frame.crop_y1.astype(str)
    crop_sets = {split: set(crop_key.loc[part.index]) for split,part in frame.groupby("split")}
    checks["crop"] = sum(len(crop_sets[a]&crop_sets[b]) for a,b in [("train","val"),("train","test"),("val","test")])
    checks["within_split_duplicate_crop"] = int(sum(crop_key.loc[part.index].duplicated().sum() for _,part in frame.groupby("split")))
    return checks


def patch_audit(frame):
    counts={key:0 for key in ["missing","corrupt","wrong_dimensions","wrong_channels","constant","black","nonfinite","hash_mismatch","duplicate_file_hash","duplicate_crop","manifest_file_mismatch","incorrect_split_path","incorrect_label_metadata"]}
    counts["duplicate_file_hash"]=int(frame.sha256.duplicated().sum())
    crop_columns=["frame_id","crop_x0","crop_y0","crop_x1","crop_y1"]; counts["duplicate_crop"]=int(frame.duplicated(crop_columns).sum())
    for row in frame.itertuples():
        path=Path(row.path)
        if not path.exists(): counts["missing"]+=1; continue
        if sha256_file(path)!=row.sha256: counts["hash_mismatch"]+=1
        try:
            opened=Image.open(path); counts["wrong_dimensions"]+=int(opened.size!=(64,64)); counts["wrong_channels"]+=int(opened.mode!="RGB"); image=np.asarray(opened)
        except Exception: counts["corrupt"]+=1; continue
        counts["constant"]+=int(np.ptp(image)==0); counts["black"]+=int(np.mean(np.all(image==0,axis=2))>.02); counts["nonfinite"]+=int(not np.isfinite(image).all())
        expected=Path("data/processed/v2/full/patches")/row.split/("positive" if row.label==1 else "negative")
        counts["incorrect_split_path"]+=int(expected not in path.parents); counts["incorrect_label_metadata"]+=int((row.label==1)!=("positive" in path.parts))
        counts["manifest_file_mismatch"]+=int(path.name not in str(row.path))
    return counts

def distribution_plot(frame, path):
    canvas=Image.new("RGB",(1000,760),"white"); draw=ImageDraw.Draw(canvas)
    for panel,(column,title) in enumerate([("centre_lat","Latitude"),("centre_lon","Longitude"),("local_hour","Local hour"),("month","Month")]):
        left=40+(panel%2)*500; top=35+(panel//2)*370; width=420; height=290
        draw.rectangle((left,top,left+width,top+height),outline="black"); draw.text((left,top-22),title,fill="black")
        values=frame[column].to_numpy(float); bins=np.linspace(values.min(),values.max()+1e-9,13); hist=[]
        for label in [1,0]:
            count,_=np.histogram(frame.loc[frame.label.eq(label),column],bins=bins); hist.append(count/max(count.sum(),1))
        maximum=max(np.max(hist[0]),np.max(hist[1]),1e-9)
        for number in range(12):
            x0=left+number*width/12; x1=left+(number+1)*width/12-2
            for offset,(series,colour) in enumerate(zip(hist,[(30,100,210),(220,90,40)])):
                bar=series[number]/maximum*height*.9; draw.rectangle((x0+offset*3,top+height-bar,x1-offset*3,top+height),outline=colour,width=2)
        draw.text((left+8,top+8),"blue: positive   orange: negative",fill="black")
    path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/v2_full.yaml")); parser.add_argument("--bootstrap",type=int,default=500); args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8")); phase=config["full_build"]; manifest_path=Path(phase["outputs"]["manifest"])
    frame=pd.read_csv(manifest_path); frame["timestamp"]=pd.to_datetime(frame.frame_timestamp_utc,utc=True); frame["local_hour"]=(frame.timestamp.dt.hour+8)%24; frame["month"]=frame.timestamp.dt.month
    output=Path(phase["outputs"]["results_root"]); output.mkdir(parents=True,exist_ok=True); report=Path("report")
    distributions=distribution_stats(frame)
    phase1a_path=Path("data/processed/v2/pilot_v2/manifest.csv"); phase1a=None
    if phase1a_path.exists():
        old=pd.read_csv(phase1a_path); old["timestamp"]=pd.to_datetime(old.frame_timestamp_utc,utc=True); old["local_hour"]=(old.timestamp.dt.hour+8)%24; old["month"]=old.timestamp.dt.month
        if "distance_to_study_mask_boundary_km" not in old:
            m=config["study_mask"]; old["distance_to_study_mask_boundary_km"]=np.minimum.reduce([old.centre_lat-m["latitude_min"],m["latitude_max"]-old.centre_lat,old.centre_lon-m["longitude_min"],m["longitude_max"]-old.centre_lon])*111
        old["geographic_grid_cell"]=((old.centre_lat-config["study_mask"]["latitude_min"])//.5).astype(int).astype(str)+"_"+((old.centre_lon-config["study_mask"]["longitude_min"])//.5).astype(int).astype(str)
        phase1a=distribution_stats(old)
    index=MMDSpatiotemporalIndex.from_inventory(Path(config["outputs"]["inventory_csv"]),config["study_mask"])
    windows={"same_frame":[0,10],"minus10_plus20":[-10,20],"primary_minus20_plus30":[-20,30],"minus30_plus40":[-30,40]}; contamination={k:0 for k in windows}
    negatives=frame[frame.label.eq(0)]
    for row in negatives.itertuples():
        for name,(start,end) in windows.items():
            clear=index.patch_query(row.timestamp,row.centre_lat,row.centre_lon,64,float(config["satellite"]["degrees_per_pixel"]),10,start,end)["clear"]
            contamination[name]+=int(not clear)
    overlaps=overlap_checks(frame); patches=patch_audit(frame)
    clusters={}
    for split,part in frame.groupby("split"):
        positive=part[part.label.eq(1)]; by_frame=positive.groupby("frame_id").size()
        clusters[split]={"patches":len(part),"positive":int(part.label.sum()),"negative":int((part.label==0).sum()),"dates":int(part.date.nunique()),
                         "frames":int(part.frame_id.nunique()),"positive_frames":int(positive.frame_id.nunique()),"storms":int(part.storm_id.nunique()),
                         "active_dates":int(positive.date.nunique()),"positive_per_frame_median":float(by_frame.median()),"positive_per_frame_max":int(by_frame.max()),
                         "largest_frame_fraction":float(part.groupby("frame_id").size().max()/len(part)),"largest_date_fraction":float(part.groupby("date").size().max()/len(part)),
                         "largest_storm_fraction":float(part.groupby("storm_id").size().max()/len(part))}
    build_metrics=json.loads((Path(phase["outputs"]["root"])/"build_metrics.json").read_text()); ledger_path=Path(phase["outputs"]["frame_ledger"])
    validation={"manifest_sha256":sha256_file(manifest_path),"ledger_sha256":sha256_file(ledger_path),"configuration_hash":phase["configuration_hash"],"build_metrics":build_metrics,"clusters":clusters,"contamination":contamination,
                "overlaps":overlaps,"patch_audit":patches,"distributions_phase2":distributions,"distributions_phase1b":phase1a,"distributions_by_split":{split:distribution_stats(part) for split,part in frame.groupby("split")}}
    (output/"geographic_distribution.json").write_text(json.dumps({"phase2":distributions,"phase1b":phase1a,"by_split":{split:distribution_stats(part) for split,part in frame.groupby("split")}},indent=2)+"\n")
    pd.DataFrame(distributions).T.to_csv(output/"geographic_distribution.csv")
    distribution_plot(frame,output/"geographic_distributions.png")
    feature_sets={"latlon":["centre_lat","centre_lon"],"time_month":["hour_sin","hour_cos","month_sin","month_cos"],
                  "latlon_time_month_rf":["centre_lat","centre_lon","hour_sin","hour_cos","month_sin","month_cos"],
                  "mean_channel":["mean_B08","mean_B13","mean_B15"],"b13_min":["min_B13"]}
    frame["hour_sin"]=np.sin(2*np.pi*frame.local_hour/24); frame["hour_cos"]=np.cos(2*np.pi*frame.local_hour/24); frame["month_sin"]=np.sin(2*np.pi*(frame.month-1)/12); frame["month_cos"]=np.cos(2*np.pi*(frame.month-1)/12)
    specs=[]
    for key in ["latlon","time_month","mean_channel","b13_min"]:
        specs += [(key+"_logistic",feature_sets[key],make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))),
                  (key+"_random_forest",feature_sets[key],RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1))]
    specs.append(("latlon_time_month_random_forest",feature_sets["latlon_time_month_rf"],RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1)))
    train,val,test=[frame[frame.split.eq(x)].copy() for x in ["train","val","test"]]; natural_path=Path(phase["outputs"]["natural_manifest"]); natural=pd.read_csv(natural_path); natural["timestamp"]=pd.to_datetime(natural.frame_timestamp_utc,utc=True); natural["local_hour"]=(natural.timestamp.dt.hour+8)%24; natural["month"]=natural.timestamp.dt.month; natural["hour_sin"]=np.sin(2*np.pi*natural.local_hour/24); natural["hour_cos"]=np.cos(2*np.pi*natural.local_hour/24); natural["month_sin"]=np.sin(2*np.pi*(natural.month-1)/12); natural["month_cos"]=np.cos(2*np.pi*(natural.month-1)/12); predictions=[]; natural_predictions=[]; results={}
    for number,(name,features,model) in enumerate(specs):
        model.fit(train[features],train.label); vs=model.predict_proba(val[features])[:,1]; ts=model.predict_proba(test[features])[:,1]; threshold=threshold_by_f1(val.label.to_numpy(),vs)
        ns=model.predict_proba(natural[features])[:,1]; results[name]={"features":features,"threshold_selected_on_validation":threshold,"validation":metrics(val.label,vs,threshold),"test":metrics(test.label,ts,threshold),"natural_prevalence_test":metrics(natural.label,ns,threshold),"clustered_bootstrap":{}}
        for cluster in ["date","frame_id","storm_id"]: results[name]["clustered_bootstrap"][cluster]=bootstrap(test,ts,threshold,cluster,args.bootstrap,42+number)
        for row,score in zip(natural.itertuples(),ns): natural_predictions.append({"model":name,"path":row.path,"label":row.label,"date":row.date,"frame_id":row.frame_id,"storm_id":row.storm_id,"score":score,"threshold":threshold,"prediction":int(score>=threshold)})
        for row,score in zip(test.itertuples(),ts): predictions.append({"model":name,"path":row.path,"label":row.label,"date":row.date,"frame_id":row.frame_id,"storm_id":row.storm_id,"score":score,"threshold":threshold,"prediction":int(score>=threshold)})
    pd.DataFrame(predictions).to_csv(output/"baseline_predictions.csv",index=False); pd.DataFrame(natural_predictions).to_csv(output/"natural_prevalence_predictions.csv",index=False); (output/"clustered_bootstrap.json").write_text(json.dumps({k:v["clustered_bootstrap"] for k,v in results.items()},indent=2)+"\n")
    (report/"V2_FULL_DATASET_VALIDATION.json").write_text(json.dumps(validation,indent=2)+"\n"); (report/"V2_FULL_BASELINES.json").write_text(json.dumps(results,indent=2)+"\n")
    validation_md="# V2 Phase 2 Pilot Validation\n\n"+f"Manifest SHA-256: `{validation['manifest_sha256']}`\n\n## Split counts\n\n"+markdown(pd.DataFrame(clusters).T)+"\n\n## Contamination\n\n"+markdown(pd.Series(contamination,name="contaminated negatives").to_frame())+"\n\n## Integrity\n\n"+markdown(pd.DataFrame({"cross_split_overlap":overlaps,"patch_issue":pd.Series(patches)}).fillna(0))+"\n"
    (report/"V2_FULL_DATASET_VALIDATION.md").write_text(validation_md)
    table=[]
    for name,item in results.items(): table.append({"model":name,**item["test"]})
    baseline_md="# V2 Phase 2 Baselines\n\nThresholds were selected on validation F1 only. Confidence intervals are 95% cluster bootstraps (500 replicates) on the test split.\n\n"+markdown(pd.DataFrame(table).drop(columns="confusion_matrix"),index=False)+"\n"
    (report/"V2_FULL_BASELINES.md").write_text(baseline_md)
    geo_auc=results["latlon_random_forest"]["test"]["roc_auc"]; image_names=[n for n in results if n.startswith("mean_channel") or n.startswith("b13_min")]; best_image=max(results[n]["test"]["roc_auc"] for n in image_names)
    full_build_geo=distributions["geographic_grid_2d"]["jensen_shannon_divergence"]; phase1a_geo=phase1a["geographic_grid_2d"]["jensen_shannon_divergence"] if phase1a else None
    local_hour_smd=abs(distributions["local_hour"]["standardized_mean_difference"]); natural_metrics=json.loads((Path(phase["outputs"]["natural_root"])/"build_metrics.json").read_text()); tests_file=output/"test_results.json"; tests_pass=tests_file.exists() and json.loads(tests_file.read_text()).get("passed",False)
    criteria={"zero_primary_contamination":contamination["primary_minus20_plus30"]==0,
              "sufficient_holdout_samples":all(clusters[s]["positive"]>=500 and clusters[s]["negative"]>=500 and clusters[s]["active_dates"]>=25 and clusters[s]["positive_frames"]>=75 and clusters[s]["storms"]>=30 for s in ["val","test"]),
              "zero_cross_split_overlap":all(value==0 for key,value in overlaps.items() if key!="within_split_duplicate_crop"),
              "zero_duplicate_crops":overlaps["within_split_duplicate_crop"]==0 and patches["duplicate_crop"]==0,
              "zero_invalid_patches":all(value==0 for key,value in patches.items() if key not in {"duplicate_file_hash"}),
              "manifest_regeneration_deterministic":json.loads((output/"determinism.json").read_text(encoding="utf-8-sig")).get("match",False) if (output/"determinism.json").exists() else False,
              "geographic_rf_auc_below_0_75":geo_auc<.75,"geography_not_competitive_with_image":geo_auc<best_image-.05,
              "cluster_intervals_stable":all(results["b13_min_logistic"]["clustered_bootstrap"][cluster]["roc_auc"]["lower"]>results["latlon_random_forest"]["clustered_bootstrap"][cluster]["roc_auc"]["upper"]+.05 for cluster in ["date","frame_id","storm_id"]),
              "no_dominant_cluster":all(max(clusters[s]["largest_frame_fraction"],clusters[s]["largest_date_fraction"],clusters[s]["largest_storm_fraction"])<.25 for s in ["val","test"]),
              "natural_prevalence_created":natural_metrics["positive"]>0 and natural_metrics["negative"]>0,"all_tests_pass":tests_pass}
    decision="Proceed to neural-network training" if all(criteria.values()) else "Revise the full dataset design"
    decision_data={"decision":decision,"criteria":criteria,"geographic_rf_test_roc_auc":geo_auc,"best_image_test_roc_auc":best_image,"geographic_auc_gap_image_minus_geo":best_image-geo_auc,"full_build_grid_js":full_build_geo,"phase1a_grid_js":phase1a_geo,"natural_prevalence_summary":natural_metrics,"local_hour_absolute_smd":local_hour_smd,"note":"Decision follows the twelve pre-registered Phase 2 stopping rules; thresholds are validation-selected and never retuned on either test set."}
    (report/"V2_PHASE2_DECISION.json").write_text(json.dumps(decision_data,indent=2)+"\n"); (report/"V2_PHASE2_DECISION.md").write_text("# V2 Phase 2 Decision\n\n## "+decision+"\n\n"+markdown(pd.Series(criteria,name="pass").to_frame())+f"\n\nLatitude/longitude RF test ROC-AUC: {geo_auc:.3f}; best image-derived ROC-AUC: {best_image:.3f}.\n")
    print(json.dumps({"validation":validation,"baselines":results,"decision":decision_data},indent=2))


if __name__=="__main__": main()

