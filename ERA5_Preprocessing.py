import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import sys
import xarray as xr
import xesmf as xe
from matplotlib import ticker,_cm
from datetime import datetime, timedelta
from dateutil import tz
import warnings
warnings.filterwarnings("ignore")


def regrid_era5_to_veros(
    ERA5_path,
    veros_grid_path,
    output_path=None,
    variables=None,
    variables_dict=None,
    method="bilinear",
    weights_path='/Odyssey/private/e25cheve/data/Weights/weight_ERA5_1deg.nc',
    verbose=True,
):
    if verbose: print("Loading Veros grid…")
    ds_veros = xr.open_dataset(veros_grid_path)

    if "yt" in ds_veros: ds_veros = ds_veros.rename({"yt": "lat"})
    if "xt" in ds_veros: ds_veros = ds_veros.rename({"xt": "lon"})

    if verbose: print("Opening ERA5 dataset for coordinate extraction…")
    ds_full = xr.open_dataset(ERA5_path, chunks={"time": 1})

    if "latitude" in ds_full: ds_full = ds_full.rename({"latitude": "lat"})
    if "longitude" in ds_full: ds_full = ds_full.rename({"longitude": "lon"})

    if ds_full.lon.min() < 0:
        if verbose: print("Adjusting ERA5 longitude to 0–360…")
        ds_full = ds_full.assign_coords(lon=(ds_full.lon % 360))
        ds_full = ds_full.sortby("lon")

    if variables is None:
        variables = list(ds_full.data_vars.keys())

    if verbose:
        print("Variables to regrid:", variables)
    if verbose: print("Preparing regridder…")

    grid_src = ds_full[["lon", "lat"]]
    grid_tgt = ds_veros[["lon", "lat"]]
    if os.path.exists(weights_path):
        if verbose: print("Reusing existing weights:", weights_path)
        regridder = xe.Regridder(grid_src, grid_tgt, method, reuse_weights=True, filename=weights_path)
    else:
        if verbose: print("Computing weights (one-time cost)…")
        regridder = xe.Regridder(grid_src, grid_tgt, method, reuse_weights=False, filename=weights_path)

    for var in variables:
        if os.path.exists(output_path):
            with xr.open_dataset(output_path) as ds_out:
                if var in ds_out.variables:
                    print(f"Skipping '{var}' (already in {output_path})")
                    continue        
 
        print(f"\n=== Regridding ERA5 variable '{var}' ===")
    
        var_da = ds_full[var]
    
        regridded_list = []
        for t in var_da.time:
            if verbose:
                print(f"  - Regridding timestep {t.values}")
    
            frame = var_da.sel(time=t)
            frame_rg = regridder(frame)
            regridded_list.append(frame_rg)
    
        var_out = xr.concat(regridded_list, dim="time")
        var_out.attrs["units"] = var_da.attrs.get("units", "unknown")
        var_out.attrs["long_name"] = var_da.attrs.get("long_name", var)
        var_out.attrs["regrid_method"] = method
    
        mode = "a" if os.path.exists(output_path) else "w"
         
        print(f"  → Writing '{var}' ({mode})")
        var_out = var_out.astype("float32")
        var_out.to_dataset(name=var).to_netcdf(output_path, mode=mode)

        print("Done.")


import re

path_to_files = '/Odyssey/private/e25cheve/veros/veros_assets/global_1deg'
veros_file = 'forcing_1deg_global.nc'
veros_path = os.path.join(path_to_files, veros_file)

ERA5_files = '/Odyssey/private/e25cheve/data/ERA5_1deg_Flux'
ERA5_outputs = '/Odyssey/private/e25cheve/data/ERA5_1deg_Flux_processed'

variables = ['ewss','nsss','tp','e','ssr','str','sshf','slhf']

for fname in sorted(os.listdir(ERA5_files)):

    if not fname.endswith(".nc"):
        continue

    match = re.search(r'(\d{4})', fname)
    if match is None:
        continue

    year = int(match.group(1))

    if year <= 1993:
        continue

    ERA5_path = os.path.join(ERA5_files, fname)

    ERA5_output_file = fname[:-3] + '_processed.nc'
    ERA5_output_path = os.path.join(ERA5_outputs, ERA5_output_file)

    print(f"Regridding {fname} → {ERA5_output_file}")

    regrid_era5_to_veros(
        ERA5_path,
        veros_path,
        output_path=ERA5_output_path,
        variables=variables,
        method='bilinear'
    )

print("All files processed.")

