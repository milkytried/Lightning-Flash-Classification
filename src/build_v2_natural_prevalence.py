"""Build the frozen natural-prevalence grid sample from Phase 2 test frames."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.build_satellite_dataset import FrameSlot, build_hsd_key, download_frame, patch_black_fraction
from src.build_v2_full import _atomic_json, crop, load_frame, pixel_to_lonlat, row_record, save_patch
from src.create_mmd_inventory import sha256_file
from src.mmd_spatiotemporal_index import MMDSpatiotemporalIndex

LOGGER = logging.getLogger(__name__)


def build(config_path: Path, resume: bool = True) -> tuple[pd.DataFrame, dict]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")); phase = config["full_build"]
    ledger = pd.read_csv(phase["outputs"]["frame_ledger"]); ledger = ledger[ledger.split.eq("test")].copy()
    ledger["frame_timestamp_utc"] = pd.to_datetime(ledger.frame_timestamp_utc, utc=True)
    index = MMDSpatiotemporalIndex.from_inventory(Path(config["outputs"]["inventory_csv"]), config["study_mask"])
    root = Path(phase["outputs"]["natural_root"]); root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "build_checkpoint.json"; rows=[]; excluded=[]; failed=[]; start=1
    if resume and checkpoint_path.exists():
        state=json.loads(checkpoint_path.read_text(encoding="utf-8")); rows=state["rows"]; excluded=state["excluded"]; failed=state["failed"]; start=state["next_frame_number"]
    size=int(config["satellite"]["patch_size"]); step=int(phase["natural_prevalence"]["grid_step_pixels"]); margin=float(config["labels"]["safety_margin_km"])
    for number,frame in enumerate(ledger.itertuples(),1):
        if number < start: continue
        timestamp=pd.Timestamp(frame.frame_timestamp_utc).tz_convert("UTC"); slot=FrameSlot(timestamp)
        try:
            expected=[]
            for band in config["satellite"]["bands"]:
                for segment in config["satellite"]["segments"]:
                    path=Path(config["inputs"]["himawari_cache"])/slot.bucket/build_hsd_key(slot,band,int(segment))
                    if not path.exists(): raise FileNotFoundError(f"controlled-build cache missing {path}")
                    expected.append(path)
            files=download_frame(slot,config["satellite"]["bands"],config["satellite"]["segments"],config["inputs"]["himawari_cache"]); image=load_frame(files,config)
        except Exception as exc:
            failed.append({"frame_id":f"H09_{slot.date}_{slot.hhmm}","error":str(exc)}); _atomic_json(checkpoint_path,{"next_frame_number":number+1,"rows":rows,"excluded":excluded,"failed":failed}); continue
        frame_rows=0
        for y in range(size//2,image.shape[0]-size//2+1,step):
            for x in range(size//2,image.shape[1]-size//2+1,step):
                patch,bounds=crop(image,x,y,size)
                if patch is None or patch_black_fraction(patch)>.02: continue
                lat,lon=pixel_to_lonlat(x,y,image.shape[:2],config["study_mask"])
                positive=index.patch_query(timestamp,lat,lon,size,float(config["satellite"]["degrees_per_pixel"]),0.0,0,10)
                if not positive["clear"]: label=1; reason="natural-grid patch contains recorded strike in [t,t+10m)"
                else:
                    primary=index.patch_query(timestamp,lat,lon,size,float(config["satellite"]["degrees_per_pixel"]),margin,-20,30)
                    if not primary["clear"]:
                        excluded.append({"frame_id":f"H09_{slot.date}_{slot.hhmm}","x":x,"y":y,"reason":"ambiguous: clear positive window but not frozen negative window"}); continue
                    label=0; reason="natural-grid patch clear under frozen [t-20m,t+30m) rule"
                path=root/"patches"/("positive" if label else "negative")/f"{slot.date}_{slot.hhmm}_grid_{x}_{y}.png"; digest=save_patch(path,patch)
                record=row_record(frame,files,timestamp,label,reason,lat,lon,x,y,bounds,patch,digest,path,config,
                                  nearest_distance=(positive if label else primary)["nearest_distance_km"],nearest_delta=(positive if label else primary)["nearest_time_difference_minutes"])
                record["split"]="natural_prevalence_test"; record["grid_sampling_probability"]=float(phase["natural_prevalence"]["sampling_probability"]); rows.append(record); frame_rows+=1
        _atomic_json(checkpoint_path,{"next_frame_number":number+1,"rows":rows,"excluded":excluded,"failed":failed}); LOGGER.info("Natural frame %d/%d samples=%d",number,len(ledger),frame_rows)
    manifest=pd.DataFrame(rows); target=Path(phase["outputs"]["natural_manifest"]); temporary=target.with_suffix(".csv.tmp"); manifest.to_csv(temporary,index=False); temporary.replace(target)
    metrics={"frames_planned":len(ledger),"frames_built":int(manifest.frame_id.nunique()),"grid_cells_eligible":len(manifest),"positive":int(manifest.label.sum()),"negative":int((manifest.label==0).sum()),"prevalence":float(manifest.label.mean()),"ambiguous_excluded":len(excluded),"failed_frames":len(failed),"manifest_sha256":sha256_file(target),"configuration_hash":phase["configuration_hash"]}
    (root/"build_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); (root/"ambiguous_excluded.json").write_text(json.dumps(excluded,indent=2)+"\n"); (root/"failed_frames.json").write_text(json.dumps(failed,indent=2)+"\n"); checkpoint_path.unlink(missing_ok=True)
    return manifest,metrics


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/v2_full.yaml")); parser.add_argument("--no-resume",action="store_true"); args=parser.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s"); _,metrics=build(args.config,not args.no_resume); print(json.dumps(metrics,indent=2))


if __name__=="__main__": main()
