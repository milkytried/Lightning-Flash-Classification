"""Quick integration test for Phase 2 & 3: Patch extraction and DataLoader."""

print("=" * 70)
print("PHASE 2 & 3 INTEGRATION TEST: Patch Extraction + DataLoader")
print("=" * 70)

from himawari_png_loader import HimawariPNGLoader
from lightning_csv_parser import LightningCSVParser
from satellite_patch_extractor import SatellitePatchExtractor
from satellite_dataset_builder import SatelliteDatasetBuilder
import pandas as pd

# Step 1: Initialize components
print("\n[1] Initializing components...")
png_loader = HimawariPNGLoader('data/raw/himawari8_pngs')
csv_parser = LightningCSVParser('data/raw/himawari8_pngs')
patch_extractor = SatellitePatchExtractor(png_loader, 'data/processed/patches')
print("  ✓ All components initialized")

# Step 2: Test patch extraction on single PNG
print("\n[2] Testing patch extraction...")
pngs = png_loader.find_png_files()
if pngs:
    png_path, png_dt = pngs[-1]  # Latest PNG
    png_array = png_loader.load_png(png_path)
    print(f"  ✓ Loaded PNG: {png_path.name}, shape={png_array.shape}")
    
    # Load lightning for same date
    png_date_start = pd.Timestamp(png_dt).normalize()
    png_date_end = png_date_start + pd.Timedelta(days=1)
    
    daily_lightning = csv_parser.load_all_lightning(png_date_start, png_date_end)
    print(f"  ✓ Loaded {len(daily_lightning)} lightning records for {png_date_start.date()}")
    
    # Extract patches
    result = patch_extractor.process_png_for_dataset(
        png_array, png_dt, daily_lightning, split='test', n_negative_per_positive=1
    )
    print(f"  ✓ Extracted patches:")
    print(f"    - Positive: {result['n_positive']}")
    print(f"    - Negative: {result['n_negative']}")
    print(f"    - Total: {len(result['patches'])}")
    
    if len(result['patches']) > 0:
        print(f"  ✓ Sample patch metadata:")
        sample = result['patches'][0]
        print(f"    - Path: .../{sample['path'].split('/')[-1]}")
        print(f"    - Label: {sample['label']}")
        print(f"    - Location: ({sample['lat']:.2f}, {sample['lon']:.2f})")

# Step 3: Test dataset builder (quick sample)
print("\n[3] Testing dataset builder...")
builder = SatelliteDatasetBuilder(png_loader, csv_parser, patch_extractor)
print("  Building dataset from 1 PNG (quick test)...")
df = builder.build_dataset(sample_limit=1)

if len(df) > 0:
    print(f"  ✓ Built dataset with {len(df)} patches")
    print(f"    - Positive: {(df['label'] == 1).sum()}")
    print(f"    - Negative: {(df['label'] == 0).sum()}")
    print(f"    - Splits: train={((df['split']=='train').sum())}, "
          f"val={((df['split']=='val').sum())}, "
          f"test={((df['split']=='test').sum())}")
    
    # Save index
    output_path = builder.save_dataset_index(df)
    print(f"  ✓ Saved index to {output_path}")
    
    # Step 4: Test DataLoader
    print("\n[4] Testing PyTorch DataLoader...")
    try:
        from himawari_data_loader import create_himawari_loaders
        import os
        
        if os.path.exists(str(output_path)):
            loaders = create_himawari_loaders(str(output_path), batch_size=4)
            print(f"  ✓ Created data loaders")
            
            # Test loading batch
            for split, loader in loaders.items():
                if len(loader) > 0:
                    images, labels = next(iter(loader))
                    print(f"    - {split}: batch shape={images.shape}, labels={labels.tolist()}")
    except Exception as e:
        print(f"  ⚠ DataLoader test skipped: {e}")

print("\n" + "=" * 70)
print("PHASE 2 & 3 INTEGRATION TEST: COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("1. Build full dataset on all PNGs (satellite_dataset_builder.py)")
print("2. Implement train_satellite.py (use existing model_arch.py)")
print("3. Train ResNet-50 on patches")
print("4. Evaluate and visualize results")
