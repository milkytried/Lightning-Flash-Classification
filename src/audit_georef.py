"""Audit PNG and NetCDF georeferencing information."""
import os
from PIL import Image

# Check PNG dimensions
png_path = 'data/raw/himawari8_pngs/12_May_Himawari.png'
if os.path.exists(png_path):
    img = Image.open(png_path)
    print(f'PNG: {png_path}')
    print(f'  Size: {img.size} (width x height)')
    print(f'  Format: {img.format}')
    print(f'  Mode: {img.mode}')
    print(f'  Info keys: {list(img.info.keys())}')
    if 'exif' in img.info:
        print(f'  EXIF data available')
else:
    print(f'File not found: {png_path}')

# Check netCDF file
try:
    import netCDF4
    nc_path = 'data/raw/himawari8_pngs/lightning.nc'
    if os.path.exists(nc_path):
        print(f'\nNetCDF: {nc_path}')
        ds = netCDF4.Dataset(nc_path)
        print(f'  Dimensions: {dict(ds.dimensions)}')
        print(f'  Variables: {list(ds.variables.keys())}')
        
        # Print variable details
        for var_name in list(ds.variables.keys())[:10]:
            var = ds.variables[var_name]
            print(f'    {var_name}: shape={var.shape}, dtype={var.dtype}')
            if hasattr(var, 'units'):
                print(f'      units: {var.units}')
            if hasattr(var, 'long_name'):
                print(f'      long_name: {var.long_name}')
        
        ds.close()
    else:
        print(f'File not found: {nc_path}')
except ImportError:
    print('\nnetCDF4 not installed')
except Exception as e:
    print(f'Error reading NetCDF: {e}')
    import traceback
    traceback.print_exc()

# List PNG files in directory structure
print('\n\nSample PNG files found:')
for root, dirs, files in os.walk('data/raw/himawari8_pngs'):
    for f in files:
        if f.endswith('.png'):
            print(f'  {os.path.join(root, f)}')
            if len([x for x in os.walk('data/raw/himawari8_pngs') for file in x[2] if file.endswith('.png')]) > 10:
                print('  ... (more files)')
                break
    else:
        continue
    break
