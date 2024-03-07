import xarray as xr
import xesmf as xe
import cf_xarray as cfxr
import glob
from pathlib import Path
import numpy as np
from project_utils import utils

### -----------------------------------------------------------------------
### pre-processing functions

def concat_regrid_files(ds, out_grid, VAR_NAME, MODEL, OCEAN = False): 
    """ regrid
       add encoding attributes to regridded data 
       """
    
    fill_val = ds[VAR_NAME].encoding['_FillValue']
    var_dtype = ds[VAR_NAME].encoding['dtype'].type
    
    if OCEAN and MODEL in ["MPI-ESM1-2-LR"]: 
        ds = ds.rename({"longitude": "lon", "latitude": "lat"}).drop(["i","j"], dim = None)[[VAR_NAME]]
    elif OCEAN and MODEL == "MIROC6": 
        ds = ds.rename({"longitude": "lon", "latitude": "lat"}).drop(["x","y"], dim = None)[[VAR_NAME]]
    elif OCEAN and MODEL in ["NorCPM1"]:
        lat_corners = cfxr.bounds_to_vertices(ds.isel(time = 0).vertices_latitude, "vertices", order=None)
        lon_corners = cfxr.bounds_to_vertices(ds.isel(time = 0).vertices_longitude, "vertices", order=None)
        ds = ds.assign(lon_b=lon_corners, lat_b=lat_corners).drop(["vertices_latitude", "vertices_longitude", "time_bnds"])
    elif OCEAN and MODEL == "IPSL-CM6A-LR": 
        ds = ds.assign_coords({"y": ds.y, "x": ds.x}).sel(x = slice(1, 361)) ## remove np.nan slice from raw data
        lat_corners = cfxr.bounds_to_vertices(ds.isel(time = 0).bounds_nav_lat, "nvertex", order=None)
        lon_corners = cfxr.bounds_to_vertices(ds.isel(time = 0).bounds_nav_lon, "nvertex", order=None)
        ds = ds.assign(lon_b=lon_corners, lat_b=lat_corners).rename({"nav_lon": "lon", "nav_lat": "lat"}).drop(
            ["bounds_nav_lat", "bounds_nav_lon", "time_bounds", "area"])
    elif OCEAN and MODEL == "CNRM-CM6-1": 
        ds = ds.assign_coords({"y": ds.y, "x": ds.x}).sel(x = slice(1, 361)) ## remove np.nan slice from raw data
        lat_corners = cfxr.bounds_to_vertices(ds.isel(time = 0).bounds_lat, "nvertex", order=None)
        lon_corners = cfxr.bounds_to_vertices(ds.isel(time = 0).bounds_lon, "nvertex", order=None)
        ds = ds.assign(lon_b=lon_corners, lat_b=lat_corners).drop(
            ["bounds_lat", "bounds_lon", "time_bounds"])
    elif OCEAN and MODEL in ["ACCESS-ESM1-5", "CanESM5"]:
        ds = ds.rename({"longitude": "lon", "latitude": "lat"}).drop(["vertices_latitude","vertices_longitude"], 
                                                                     dim = None)[[VAR_NAME]]
    elif OCEAN:
        ds = ds[[VAR_NAME]]
        
    regridder = xe.Regridder(ds, out_grid, method = "bilinear", periodic = True, 
                            ignore_degenerate=True)
    ds_regrid = regridder(ds, keep_attrs=True)
    
    ds_regrid[VAR_NAME].encoding = {VAR_NAME: {'zlib': True, 'complevel': 1,
                                               'chunksizes': (1, len(out_grid.lat), len(out_grid.lon)),
                                               'dtype': var_dtype,
                                               '_FillValue': fill_val}}
    return(ds_regrid)

def regrid_save_files(out_grid, VARIANTS, VAR_NAME, FREQ, MODEL, EXP, GRID, OVERWRITE = False, OCEAN = False): 
    """combine, regrid, and save files (separately for each variant)
       skips if file already exists, unless overwrite set to True
    """
    out_res = (out_grid.lat[1] - out_grid.lat[0]).values.astype('int')
    for v in VARIANTS:

    #create filename
        out_fname = "../processed_data/CMIP6/"+VAR_NAME+"_"+FREQ+"_"+MODEL+"_"+\
                     EXP+"_"+v+"_18500101_20141231_"+str(out_res)+"x"+str(out_res)+".nc"
    
        ## skip to next variant if file already exists
        if not OVERWRITE and Path(out_fname).exists():
            print(v, "not regridding", end =" ")
            continue
    
        print(v, "regridding...", end =" ")   
        variant_files = sorted(glob.glob("../input_data/CMIP6_"+EXP+"/"+FREQ+"/"+VAR_NAME+"/*"+MODEL+"_*"+v+"_"+GRID+"*.nc"))
        ds = xr.open_mfdataset(variant_files).sel(time = slice("1850-01-01", "2014-12-31"))

        ## change dates for models with weird calendars to create consistency across models
        if MODEL in ["CanESM5", "GISS-E2-1-G", "NorCPM1"]:
            dates = utils.load_dates()
            ds.coords['time'] = dates
            
        ds_regrid = concat_regrid_files(ds, out_grid, VAR_NAME, MODEL, OCEAN)
    
        # write netcdf
        ds_regrid.to_netcdf(out_fname)
    
def calc_save_moving_average(VARIANTS, VAR_NAME, FREQ, MODEL, EXP, WINDOW, OVERWRITE = False, out_res = 5):
    for v in VARIANTS:
        #create filename
        out_fname = "../processed_data/CMIP6/"+VAR_NAME+"_"+FREQ+"_"+MODEL+"_"+\
          EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month.nc"
    
        ## skip to next variant if file already exists
        if not OVERWRITE and Path(out_fname).exists():
            print(v, "not recalculating moving average", end =" ")
            continue
            
        print(v, "calculating moving average...", end =" ")
        ds = xr.open_mfdataset("../processed_data/CMIP6/"+VAR_NAME+"_"+FREQ+"_"+MODEL+"_"+\
             EXP+"_"+v+"_18500101_20141231_"+str(out_res)+"x"+str(out_res)+".nc")
        ds = ds.rolling(time = WINDOW).mean(dim = "time")
    
        # write netcdf
        ds.to_netcdf(out_fname)
    
def calc_save_forced_response(VARIANTS, VAR_NAME, FREQ, MODEL, EXP, WINDOW, OVERWRITE = False, out_res = 5):
    out_fname = "../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+str(out_res)+"x"+str(out_res)+"_" + str(WINDOW) + "month_forced_response.nc"
        
    ## skip if file already exists
    if not OVERWRITE and Path(out_fname).exists():
        print("not recalculating forced response", end =" ")
    
    else: 
        print("calculating forced_response...", end =" ")
    
        dat = []
        for v in VARIANTS:
            ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+\
                             "_monthly_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month.nc", chunks = "auto")
            ds = ds.assign_coords({"variant": v})
            dat.append(ds)
    
        dat = xr.concat(dat, dim = "variant")
    
        dat = dat.groupby("time.year").mean(dim = ['time', 'variant'])
        dat_daily = xr.ones_like(ds)
        dat_daily = (dat_daily.groupby("time.year")*dat).drop("year")
        dat_daily.to_netcdf(out_fname)
        
def calc_save_mean_sd(VAR_NAME, VARIANTS, FREQ, MODEL, EXP, WINDOW, OVERWRITE=False, out_res = 5):    

    dat_fr = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_forced_response.nc", chunks = "auto")
    
    for v in VARIANTS:
        mean_file = "../processed_data/CMIP6/"+VAR_NAME+"_"+FREQ+"_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_mean.nc"
        sd_file = "../processed_data/CMIP6/"+VAR_NAME+"_"+FREQ+"_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_sd.nc"
      
        ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month.nc", chunks = "auto")
        ds = ds - dat_fr
        if Path(mean_file).exists() and not OVERWRITE:
            print("not recalculating mean")
        else: 
            print("calculating mean...")
            ds.mean(dim = ['time']).to_netcdf(mean_file)
        
        if Path(sd_file).exists() and not OVERWRITE:
            print("not recalculating standard deviation")
        else:   
            print("calculating standard deviation...")
            ds.std(dim = ['time']).to_netcdf(sd_file)
        
def calc_save_quantiles(VAR_NAME, VARIANTS, FREQ, MODEL, EXP, WINDOW, q, OVERWRITE=False, out_res = 5):
    
    dat_fr = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_forced_response.nc", chunks = "auto")
    
    for v in VARIANTS:
        out_file = "../processed_data/CMIP6/"+VAR_NAME+"_"+FREQ+"_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_quantiles.nc"
        if Path(out_file).exists() and not OVERWRITE:
            print(v, "not recalculating quantiles")
        else: 
            print(v, "calculating quantiles...")
            ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month.nc", chunks = "auto")
            ds_mean = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_mean.nc", chunks = "auto")
            ds_sd = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+"_"+v+"_"+str(out_res)+"x"+str(out_res)+"_"+str(WINDOW)+"month_sd.nc", chunks = "auto")
            ds = ds - dat_fr
            ds = (ds - ds_mean)/ds_sd
            ds.quantile(q = q, dim = ['time']).to_netcdf(out_file)
