"""Quick test of PNG loader and CSV parser on sample data."""

from himawari_png_loader import HimawariPNGLoader
from lightning_csv_parser import LightningCSVParser
from datetime import datetime, timedelta

print("=" * 60)
print("PHASE 1 VALIDATION TEST")
print("=" * 60)

# Test 1: PNG Loader
print("\n[1] Testing PNG Loader...")
png_loader = HimawariPNGLoader('data/raw/himawari8_pngs')
pngs = png_loader.find_png_files()
print(f"  ✓ Found {len(pngs)} PNG files")

if pngs:
    png_path, png_dt = pngs[-1]  # Load latest PNG
    image = png_loader.load_png(png_path)
    print(f"  ✓ Loaded PNG: {png_path.name}, shape={image.shape}")
    
    # Test coordinate mapping
    test_coords = [
        (3.1, 101.7),  # Kuala Lumpur
        (2.2, 102.2),  # Pahang
        (5.4, 100.3),  # Perak
    ]
    
    for lat, lon in test_coords:
        if png_loader.validate_coordinates(lat, lon):
            x, y = png_loader.latlon_to_pixel(lat, lon)
            print(f"  ✓ ({lat}, {lon}) → pixel ({x}, {y})")
        else:
            print(f"  ✗ ({lat}, {lon}) out of bounds")

# Test 2: Lightning CSV Parser (quick sample)
print("\n[2] Testing Lightning CSV Parser (sample)...")
parser = LightningCSVParser('data/raw/himawari8_pngs')
print(f"  ✓ Found {len(parser.csv_files)} CSV files")

# Load a single day (should be fast)
try:
    print("  Loading 2023-01-01...")
    df = parser.load_all_lightning(
        datetime(2023, 1, 1),
        datetime(2023, 1, 2)
    )
    print(f"  ✓ Loaded {len(df)} lightning records")
    
    if len(df) > 0:
        print(f"    Sample record:")
        sample = df.iloc[0]
        print(f"      Timestamp: {sample['timestamp']}")
        print(f"      Location: ({sample['latitude']:.4f}, {sample['longitude']:.4f})")
        print(f"      Amplitude: {sample['amplitude']}")
        print(f"      Type: {sample['strike_type']}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 60)
print("PHASE 1 VALIDATION: COMPLETE")
print("=" * 60)
print("\nNext steps:")
print("1. Implement satellite_patch_extractor.py")
print("2. Implement satellite_dataset_builder.py")
print("3. Create dataset index CSV")
