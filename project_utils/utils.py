import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf 

### -----------------------------------------------------------------------
### miscellaneous functions

def load_dates(dataset="CMIP"):
    """returns monthly date time series for historical simulations"""
    if dataset=="CMIP": 
        dates = pd.to_datetime(pd.read_csv("../processed_data/cmip_date_timeseries.csv", index_col = 0).index)
    elif dataset=="ERSSTv5":
        dates = pd.to_datetime(pd.read_csv("../processed_data/ersstv5_date_timeseries.csv", index_col = 0).index)
    return(dates)

def load_lat_lon():
    """returns lat and lon values for 5x5 global grid"""
    lats = pd.read_csv("../processed_data/lat_5x5.csv")['lat'].to_numpy()
    lons = pd.read_csv("../processed_data/lon_5x5.csv")['lon'].to_numpy()
    return lats, lons

def load_land_mask(MODEL, tensor = False):
    """returns land mask for 5x5 global grid
    ocean grids = 1
    land grids = 0 """
    land_mask = xr.open_dataset("../processed_data/land_mask_5x5_"+MODEL+".nc").mask.values
    if tensor:
        return tf.convert_to_tensor(np.expand_dims(land_mask, axis = -1))
    else: 
        return land_mask
    
def get_prediction_dates(orig_dates, input_length, lead, prediction_length):
    """takes initial date timeseries and returns dates for all input maps and
       dates for output predictions
       all dates correspond to last month of average period"""
    
    ## prediction dates correspond to the day the prediction is made
    first_date = sum(input_length) + 1
    last_date = len(orig_dates) - (lead + prediction_length-1)
    prediction_dates_ind = np.arange(first_date - 1, last_date) #subtract one for base-zero indexing 
    
    input_dates = []
    placeholder = 1 ## months before prediction date for given input layer
    for months in input_length:
        input_dates.append(orig_dates[prediction_dates_ind - (placeholder)])
        placeholder += months
        
    input_dates = np.stack(input_dates, axis = -1)
    output_dates = orig_dates[prediction_dates_ind + lead + prediction_length-1]
    
    prediction_dates = orig_dates[prediction_dates_ind]
    
    return(input_dates, output_dates, prediction_dates)

def shift_input_maps(x, lons, edge_lon = 32.5, pole_padding = 5):
    """shift input maps to have different edge longitude
       also removes latitudes at poles based on pole_padding 
       pole padding is in NUMBER OF GRID ROWS not degrees
       assumes 4D input maps [time, lat, lon, layers]"""
    
    edge_lon_ind = list(lons).index(edge_lon)
    shift_ind = np.concatenate([np.arange(edge_lon_ind,len(lons)), 
                                np.arange(0, edge_lon_ind+1)])
    
    x_shift = x[:,pole_padding:-pole_padding,shift_ind,:]
    return(x_shift)

def to_quantile_categories(y_dat, q):
    if(len(q.shape) == 2): 
        N = q.shape[0]
        y_dat = y_dat.reshape(N, -1)
        y_digitize = []
        for k in range(N):
            y_digitize.append(np.digitize(y_dat[k], q[k]))
        y_digitize = np.concatenate(y_digitize, axis = 0)
    else: 
        y_digitize = np.digitize(y_dat, q)
    
    return(tf.keras.utils.to_categorical(y_digitize))

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
def _calc_accuracy_func(y_predict, y_true, y_quant, by_class = False, metric = "accuracy"):
    """
    inner function called with xr.apply_ufunc
    """
    y_predict = y_predict.reshape(-1)
    y_true = y_true.reshape(-1)
    
    if np.isnan(y_true).all() or np.isnan(y_predict).all():
        if by_class and metric in ["cm_normall", "cm_normpred"]:
            score = np.array(np.nan).repeat(9)
        elif by_class:
            score = np.array(np.nan).repeat(3)
        else:
            score = np.nan
    else: 
        ## remove nans
        idx = np.where(np.logical_not(np.isnan(y_predict)))[0]
        y_predict = y_predict[idx]
        y_true = y_true[idx]
        y_true = to_quantile_categories(y_true, y_quant).argmax(axis = 1)
    
        if metric=="cm_normall":
            score = confusion_matrix(y_true, y_predict, normalize = "all").reshape(-1)
        elif metric=="cm_normpred":
            score = confusion_matrix(y_true, y_predict, normalize = "pred").reshape(-1)
        elif by_class and metric=="precision":
            score = precision_score(y_true, y_predict, average = None)
        elif by_class and metric == "accuracy":
            cm = confusion_matrix(y_true, y_predict)
            score = cm.diagonal()/cm.sum(axis=1)
        else:
            score = accuracy_score(y_true, y_predict)
    
    return(score)

def calc_accuracy(ds_predict, ds_true, ds_quant, conf_q = None, 
                 by_variant = False, by_class = False, metric = "accuracy"):
    """calculate accuracy at each grid cell
    if conf_q is specified, accuracy is calculated for samples above a confidence threshold
    example: conf_q = 0.8, accuracy is calculated for the 20% most confident predictions
    """
    
    if conf_q is not None:
        conf_values = ds_predict.to_array("variable").max("variable")
        conf_thr = conf_values.quantile(q = conf_q, dim = ["time"])
        ds_predict = xr.where(conf_values >= conf_thr, ds_predict, np.nan)
        ds_true = xr.where(conf_values >= conf_thr, ds_true, np.nan)
    
    varname1 = list(ds_predict.keys())[0]
    ## create mask based on nans and predictions below confidence threshold
    na_mask = xr.where(np.isnan(ds_predict[varname1]), np.nan, 1)
    
    ## convert probabilities to predicted class
    ds_predict = ds_predict.fillna(0).to_array("variable").argmax("variable")
    ds_predict = xr.where(np.isnan(na_mask), np.nan, ds_predict)
        
    if by_variant:
        accuracy_ds = xr.apply_ufunc(_calc_accuracy_func, ds_predict, ds_true, ds_quant, 
                            input_core_dims = (["time"], ["time"], ["q"]), 
                             vectorize = "true", ## required when function can only take 1D array
                                        ) 
    elif by_variant and by_class:
        accuracy_ds = xr.apply_ufunc(_calc_accuracy_func, ds_predict, ds_true, ds_quant, 
                                     kwargs={"by_class": by_class, "metric": metric},
                            input_core_dims = (["time"], ["time"], ["q"]), 
                            output_core_dims = [["predicted_class"]],
                             vectorize = "true")
    elif by_class:
        ## calculate metric for each variant separately, then average
        accuracy_ds = xr.apply_ufunc(_calc_accuracy_func, ds_predict, ds_true, ds_quant, 
                                     kwargs={"by_class": by_class, "metric": metric},
                            input_core_dims = (["time"], ["time"], ["q"]), 
                            output_core_dims = [["predicted_class"]],
                             vectorize = "true")
        accuracy_ds = accuracy_ds.mean(dim = "variant")
    else:
        ## calculate metric for each variant separately, then average
        accuracy_ds = xr.apply_ufunc(_calc_accuracy_func, ds_predict, ds_true, ds_quant, 
                            input_core_dims = (["time"], ["time"], ["q"]), 
                             vectorize = "true", ## required when function can only take 1D array
                                        ) 
        accuracy_ds = accuracy_ds.mean(dim = "variant")
                                 
    return(accuracy_ds)

def _calc_persistance_accuracy(y, q, y_persist):
    if np.isnan(y).all():
        score = np.nan
    else: 
        y = to_quantile_categories(y, q)
        y_persist = to_quantile_categories(y_persist, q)
        score = accuracy_score(y, y_persist)
    return(score)

def calc_persistance_accuracy(ds_y, ds_quant, ds_persist):
    
    accuracy_ds = xr.apply_ufunc(_calc_persistance_accuracy, ds_y, ds_quant, ds_persist, 
                             input_core_dims = (["time"], ["q"], ["time"]), 
                             vectorize = "true", ## required when function can only take 1D array
                                        ) 
    accuracy_ds = accuracy_ds.mean(dim = "variant")
    return(accuracy_ds)

def get_cell_weights(land_mask):
    weights = np.cos(np.deg2rad(land_mask.lat))*land_mask
    weights = weights/(weights.sum()/land_mask.sum())
    return(weights)


def _calc_correct_conf(y_predict, y_true, y_quant):
    """
    inner function called with xr.apply_ufunc
    """
    y_predict = y_predict.reshape(-1)
    y_true = y_true.reshape(-1)
    
    if np.isnan(y_true).all() or np.isnan(y_predict).all():
        correct_conf_preds = np.repeat(np.nan, len(y_true))
    else: 
        na_vector_mask = np.where(np.isnan(y_true), np.nan, 1)
        y_true = to_quantile_categories(y_true, y_quant).argmax(axis = 1)
        y_true = np.where(np.isnan(na_vector_mask), np.nan, y_true)
        correct_conf_preds = np.where(y_true == y_predict, 1, 0)

    return(correct_conf_preds)


def find_correct_conf(ds_predict, ds_true, ds_quant, conf_q = None):

    if conf_q is not None:
        conf_values = ds_predict.to_array("variable").max("variable")
        conf_thr = conf_values.quantile(q = conf_q, dim = ["time"])
        ds_predict = xr.where(conf_values >= conf_thr, ds_predict, np.nan)
        ds_true = xr.where(conf_values >= conf_thr, ds_true, np.nan)
    
    varname1 = list(ds_predict.keys())[0]
    ## create mask based on nans and predictions below confidence threshold
    na_mask = xr.where(np.isnan(ds_predict[varname1]), np.nan, 1)
    
    ## convert probabilities to predicted class
    ds_predict = ds_predict.fillna(0).to_array("variable").argmax("variable")
    ds_predict = xr.where(np.isnan(na_mask), np.nan, ds_predict)
        
    correct_conf_ds = xr.apply_ufunc(_calc_correct_conf, ds_predict, ds_true, ds_quant, 
                            input_core_dims = ([], [], ["q"]), 
                             vectorize = "true", ## required when function can only take 1D array
                                        ) 
                                 
    return(correct_conf_ds)