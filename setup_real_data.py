#!/usr/bin/env python3
"""
One-command setup for real data ingestion.
Run this to verify and test your daily PNG pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, cwd=".")
        if result.returncode == 0:
            print(f"\n✅ Success")
            return True
        else:
            print(f"\n❌ Failed (exit code: {result.returncode})")
            return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    """Setup and verify real data pipeline."""
    print("\n" + "="*60)
    print("🚀 LIGHTNING FLASH CLASSIFICATION")
    print("   Real Data Pipeline Setup")
    print("="*60)
    
    steps = [
        ("Installing new packages", 
         'cd "c:\\Projects\\Project Capstone" && .\\venv\\Scripts\\pip install schedule Pillow opencv-python -q'),
        
        ("Creating directories",
         'cd "c:\\Projects\\Project Capstone" && mkdir -Force data\\raw\\himawari8_pngs, logs\\daily_summaries'),
        
        ("Running test suite",
         'cd "c:\\Projects\\Project Capstone" && python -m pytest tests/test_daily_ingestion.py -v'),
        
        ("Processing first PNG (if exists)",
         'cd "c:\\Projects\\Project Capstone" && python src/daily_scheduler.py once'),
    ]
    
    results = {}
    for desc, cmd in steps:
        results[desc] = run_command(cmd, desc)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SETUP SUMMARY")
    print("="*60)
    
    for step, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {step}")
    
    if all(results.values()):
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("""
Your daily data ingestion pipeline is ready!

NEXT STEPS:

1. Save your Himawari-8 PNG here:
   c:\\Projects\\Project Capstone\\data\\raw\\himawari8_pngs\\

2. Each day at 6 AM, the system will:
   ✅ Scan for new PNG files
   ✅ Extract satellite channels (IR, WV, VIS)
   ✅ Create training patches
   ✅ Update HDF5 dataset

3. When you get lightning labels (CSV):
   ✅ Save to: data/raw/mmd_lightning.csv
   ✅ Run labeling script
   ✅ Start training: python src/train.py

See REAL_DATA_SETUP.md for detailed instructions!
        """)
        return 0
    else:
        print("\n❌ Setup incomplete. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
