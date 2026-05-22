"""Extract geographic bounds from NetCDF."""
import netCDF4
import numpy as np

nc_path = 'data/raw/himawari8_pngs/lightning.nc'
ds = netCDF4.Dataset(nc_path)

lat = np.array(ds.variables['lat'][:])
lon = np.array(ds.variables['lon'][:])

print(f'Latitude range: {lat.min()} to {lat.max()}')
print(f'Longitude range: {lon.min()} to {lon.max()}')
print(f'Latitude resolution: {lat[1] - lat[0]:.6f} degrees/sample')
print(f'Longitude resolution: {lon[1] - lon[0]:.6f} degrees/sample')
print(f'NetCDF grid shape (lon x lat): {len(lon)} x {len(lat)}')
print(f'Data shape (lon x lat x months): {ds.variables["thunder_hours"].shape}')

# Malaysia bounds (approximate)
print(f'\nMalaysia region (approx):')
print(f'  Latitude: 1.0 to 6.5 degrees N')
print(f'  Longitude: 100.0 to 120.0 degrees E')

# Find indices for Malaysia
lat_diff = np.abs(lat - 1.0)
lat_idx_min = lat_diff.argmin()
lat_diff = np.abs(lat - 6.5)
lat_idx_max = lat_diff.argmin()

lon_diff = np.abs(lon - 100.0)
lon_idx_min = lon_diff.argmin()
lon_diff = np.abs(lon - 120.0)
lon_idx_max = lon_diff.argmin()

print(f'\nIndex bounds for Malaysia in NetCDF grid:')
print(f'  Latitude indices: {lat_idx_min} to {lat_idx_max} (rows)')
print(f'  Longitude indices: {lon_idx_min} to {lon_idx_max} (cols)')
print(f'  Grid coverage: {lat_idx_max - lat_idx_min} rows x {lon_idx_max - lon_idx_min} cols')
print(f'  Actual lat/lon bounds: lat=[{lat[lat_idx_min]:.2f}, {lat[lat_idx_max]:.2f}], lon=[{lon[lon_idx_min]:.2f}, {lon[lon_idx_max]:.2f}]')

ds.close()

print(f'\n\nCONCLUSION:')
print(f'✓ NetCDF has complete lat/lon georeferencing')
print(f'✓ Can map lightning lat/lon → grid indices')
print(f'✓ PNG (950x800) is likely a downsampled/cropped view')
print(f'✓ Need to determine PNG crop bounds relative to full grid')
