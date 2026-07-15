"""Bounded concurrent prefetch for the already-frozen Phase 2 frame ledger."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from src.build_satellite_dataset import FrameSlot, build_hsd_key, download_frame


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/v2_full.yaml")); parser.add_argument("--start-frame",type=int,default=100); parser.add_argument("--workers",type=int,default=4); args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8")); phase=config["full_build"]; ledger=pd.read_csv(phase["outputs"]["frame_ledger"]); ledger=ledger.iloc[args.start_frame-1:].copy(); cache=Path(config["inputs"]["himawari_cache"])
    def fetch(item):
        number,row=item; timestamp=pd.Timestamp(row.frame_timestamp_utc); slot=FrameSlot(timestamp); before={}
        for band in config["satellite"]["bands"]:
            for segment in config["satellite"]["segments"]:
                path=cache/slot.bucket/build_hsd_key(slot,band,int(segment)); before[str(path)]=path.stat().st_size if path.exists() else None
        try:
            files=download_frame(slot,config["satellite"]["bands"],config["satellite"]["segments"],str(cache)); new=[path for path in files if before[str(path)] is None]
            return {"frame_number":number,"timestamp":timestamp.isoformat(),"status":"completed","new_files":len(new),"new_bytes":sum(path.stat().st_size for path in new),"error":""}
        except Exception as exc: return {"frame_number":number,"timestamp":timestamp.isoformat(),"status":"failed","new_files":0,"new_bytes":0,"error":str(exc)}
    records=[]; target=Path(phase["outputs"]["state_root"])/"prefetch_download_ledger.csv"; target.parent.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch,(number,row)):number for number,row in zip(range(args.start_frame,len(pd.read_csv(phase["outputs"]["frame_ledger"]))+1),ledger.itertuples())}
        for future in as_completed(futures):
            records.append(future.result()); pd.DataFrame(records).sort_values("frame_number").to_csv(target,index=False)
    print(pd.DataFrame(records).groupby("status").agg(frames=("frame_number","size"),files=("new_files","sum"),bytes=("new_bytes","sum")))


if __name__=="__main__": main()
