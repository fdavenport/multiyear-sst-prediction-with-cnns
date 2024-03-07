import xarray as xr
import numpy as np
from project_utils import utils

### -----------------------------------------------------------------------
### reading/loading processed data 

def read_data(VAR_NAME, MODEL, WINDOW, dates = None, VARIANTS = None, stat = None, 
              EXP = "historical", as_xarray = False):

    dat = []
    for v in VARIANTS:
        if stat == "forced_response":
            ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+\
                                 "_5x5_"+str(WINDOW)+"month_"+stat+".nc").sel(time = dates)
            ds_val = ds[VAR_NAME].values
        elif stat == "quantiles": 
            ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+\
                                 "_"+v+"_5x5_"+str(WINDOW)+"month_"+stat+".nc")
            ds_val = np.expand_dims(ds[VAR_NAME].values, 0)
        elif stat in ["mean", "sd"]:
            ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+\
                                 "_"+v+"_5x5_"+str(WINDOW)+"month_"+stat+".nc")
            ds_val = np.expand_dims(ds[VAR_NAME].values, 0)
        else:
            ds = xr.open_dataset("../processed_data/CMIP6/"+VAR_NAME+"_monthly_"+MODEL+"_"+EXP+\
                                 "_"+v+"_5x5_"+str(WINDOW)+"month.nc").sel(time = dates)
            ds_val = ds[VAR_NAME].values
        dat.append(ds_val)
    dat = np.concatenate(dat, axis = 0)
        
    if as_xarray:
        lats = ds.lat
        lons = ds.lon
        if stat in ["mean", "sd"]: 
            dat = xr.DataArray(data = dat.reshape(len(VARIANTS), len(lats), len(lons)),
                               coords = dict(variant = VARIANTS, lat = lats, lon = lons))
        elif stat == "quantiles":
            dat = xr.DataArray(data = dat.reshape(len(VARIANTS), 2, len(lats), len(lons)), 
                               coords = dict(variant = VARIANTS, q = np.arange(2), 
                                             lat = lats, lon = lons))
        else:
            dat = xr.DataArray(data = dat.reshape(len(VARIANTS), len(dates), len(lats), len(lons)), 
                               coords=dict(variant = VARIANTS, time = dates, lat = lats, lon = lons), 
                               name = VAR_NAME)
    return(dat)

def load_xy_data(orig_dates, input_length, lead, prediction_length, X_VAR_NAME, Y_VAR_NAME, VARIANTS, MODEL):
    """## returns detrended and normalized x and y data"""
    
    input_dates, output_dates, _ = utils.get_prediction_dates(orig_dates, input_length, lead, prediction_length)
    
    ## input_data
    input_dat = []
    for i, window in enumerate(input_length):
        dat = read_data(X_VAR_NAME, MODEL, window, dates = input_dates[:,i], VARIANTS = VARIANTS, 
                                   EXP = "historical")
        dat_fr = read_data(X_VAR_NAME, MODEL, window, dates = input_dates[:,i], VARIANTS = VARIANTS, 
                                   EXP = "historical", stat = "forced_response")
        dat_mean = read_data(X_VAR_NAME, MODEL, window, EXP = "historical", VARIANTS = VARIANTS, stat = "mean")
        dat_mean = np.broadcast_to(np.expand_dims(dat_mean, 1), 
                                   (dat_mean.shape[0], len(input_dates[:,i]), dat_mean.shape[1], dat_mean.shape[2])).reshape(-1, dat_mean.shape[1], dat_mean.shape[2])
        dat_sd = read_data(X_VAR_NAME, MODEL, window, EXP = "historical", VARIANTS = VARIANTS, stat = "sd")
        dat_sd = np.broadcast_to(np.expand_dims(dat_sd, 1), 
                                   (dat_sd.shape[0], len(input_dates[:,i]), dat_sd.shape[1], dat_sd.shape[2])).reshape(-1, dat_sd.shape[1], dat_sd.shape[2])
        input_dat.append((dat - dat_fr - dat_mean)/dat_sd)
    input_dat = np.stack(input_dat, axis = -1)

    ## output data
    dat = read_data(Y_VAR_NAME, MODEL, prediction_length, dates = output_dates, VARIANTS = VARIANTS, 
                                   EXP = "historical")
    dat_fr = read_data(Y_VAR_NAME, MODEL, prediction_length, dates = output_dates, VARIANTS = VARIANTS, 
                                   EXP = "historical", stat = "forced_response")
    dat_mean = read_data(Y_VAR_NAME, MODEL, prediction_length, EXP = "historical", VARIANTS = VARIANTS, stat = "mean")
    dat_mean = np.broadcast_to(np.expand_dims(dat_mean, 1), 
                                   (dat_mean.shape[0], len(output_dates), dat_mean.shape[1], dat_mean.shape[2])).reshape(-1, dat_mean.shape[1], dat_mean.shape[2])
    dat_sd = read_data(Y_VAR_NAME, MODEL, prediction_length, EXP = "historical", VARIANTS = VARIANTS, stat = "sd")
    dat_sd = np.broadcast_to(np.expand_dims(dat_sd, 1), 
                            (dat_sd.shape[0], len(output_dates), dat_sd.shape[1], dat_sd.shape[2])).reshape(-1, dat_sd.shape[1], dat_sd.shape[2])
    output_dat = (dat - dat_fr - dat_mean)/dat_sd
    
    return(input_dat, output_dat)
    
def load_persistance_data(orig_dates, input_length, lead, prediction_length, Y_VAR_NAME, VARIANTS, MODEL):
    
    input_dates, _, _ = utils.get_prediction_dates(orig_dates, input_length, lead, prediction_length)
    
    dat = read_data(Y_VAR_NAME, MODEL, prediction_length, dates = input_dates[:,0], 
                    VARIANTS = VARIANTS, EXP = "historical")
    dat_fr = read_data(Y_VAR_NAME, MODEL, prediction_length, dates = input_dates[:,0], 
                          VARIANTS = VARIANTS, EXP = "historical", stat = "forced_response")
    dat_mean = read_data(Y_VAR_NAME, MODEL, prediction_length, VARIANTS = VARIANTS, EXP = "historical", stat = "mean")
    dat_mean = np.broadcast_to(np.expand_dims(dat_mean, 1), 
                                   (dat_mean.shape[0], len(input_dates), dat_mean.shape[1], dat_mean.shape[2])).reshape(-1, dat_mean.shape[1], dat_mean.shape[2])
    
    dat_sd = read_data(Y_VAR_NAME, MODEL, prediction_length, VARIANTS = VARIANTS, EXP = "historical", stat = "sd")
    dat_sd = np.broadcast_to(np.expand_dims(dat_sd, 1), 
                            (dat_sd.shape[0], len(input_dates), dat_sd.shape[1], dat_sd.shape[2])).reshape(-1, dat_sd.shape[1], dat_sd.shape[2])
    dat = (dat - dat_fr - dat_mean)/dat_sd
    
    return(dat)

def load_xy_obs(orig_dates, input_length, lead, prediction_length, X_VAR_NAME, Y_VAR_NAME, DATASET):
    """## returns detrended and normalized x and y data"""
    
    input_dates, output_dates, _ = utils.get_prediction_dates(orig_dates, input_length, lead, prediction_length)
    
    ## input_data
    input_dat = []
    for i, window in enumerate(input_length):
        dat = xr.open_dataset("../processed_data/"+DATASET+"/"+X_VAR_NAME+"_5x5_"+\
                              str(window)+"month.nc").sel(time = input_dates[:,i])[X_VAR_NAME].values
        dat_trend = xr.open_dataset("../processed_data/"+DATASET+"/"+X_VAR_NAME+"_5x5_"+\
                              str(window)+"month_trend_prediction.nc").sel(time = input_dates[:,i])[X_VAR_NAME].values
        dat_mean = xr.open_dataset("../processed_data/"+DATASET+"/"+X_VAR_NAME+"_5x5_"+\
                              str(window)+"month_mean.nc")[X_VAR_NAME].values
        dat_sd = xr.open_dataset("../processed_data/"+DATASET+"/"+X_VAR_NAME+"_5x5_"+\
                              str(window)+"month_sd.nc")[X_VAR_NAME].values
        input_dat.append((dat - dat_trend - dat_mean)/dat_sd)
    input_dat = np.stack(input_dat, axis = -1)

    ## output data
    dat = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+\
                              str(prediction_length)+"month.nc").sel(time = output_dates)[Y_VAR_NAME].values
    dat_trend = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+\
                              str(prediction_length)+"month_trend_prediction.nc").sel(time = output_dates)[Y_VAR_NAME].values
    dat_mean = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+\
                              str(prediction_length)+"month_mean.nc")[Y_VAR_NAME].values
    dat_sd = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+\
                              str(prediction_length)+"month_sd.nc")[Y_VAR_NAME].values
    output_dat = (dat - dat_trend - dat_mean)/dat_sd
    
    return(input_dat, output_dat)
    
def load_persistance_obs(orig_dates, input_length, lead, prediction_length, Y_VAR_NAME, DATASET):
    
    input_dates, _, _ = utils.get_prediction_dates(orig_dates, input_length, lead, prediction_length)
    
    dat = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+ str(prediction_length)+"month.nc").sel(time = input_dates[:,0])[Y_VAR_NAME].values
    
    dat_trend = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+ str(prediction_length)+"month_trend_prediction.nc").sel(time = input_dates[:,0])[Y_VAR_NAME].values
    
    dat_mean = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+\
                              str(prediction_length)+"month_mean.nc")[Y_VAR_NAME].values
    
    dat_sd = xr.open_dataset("../processed_data/"+DATASET+"/"+Y_VAR_NAME+"_5x5_"+\
                              str(prediction_length)+"month_sd.nc")[Y_VAR_NAME].values
    dat = (dat - dat_trend - dat_mean)/dat_sd
    
    return(dat)
