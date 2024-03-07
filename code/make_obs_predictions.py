import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import random
import datetime
from pathlib import Path
import argparse
from project_utils import utils
from project_utils import read_utils as read
from project_utils import model_utils as mu

## ----------------------------------------
## INITIAL PARAMETERS ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=101)
parser.add_argument('--prediction_length', type=int, default=60)
parser.add_argument('--lead', type=int, default=0)
parser.add_argument('--obs_dataset', type=str)
parser.add_argument('--model_NN', type=str)
parser.add_argument('--y_var', type=str, default = "tos")
args = parser.parse_args()

if args.y_var in ["tas"]:
    DOMAIN = "LAND"
else:
    DOMAIN = "OCEAN"

## prediction parameters
input_length = [12, 12, 12, 60] ## input tos variables, (number of averaged months for each), in REVERSE chronological order

orig_dates = utils.load_dates(dataset=args.obs_dataset)
lats, lons = utils.load_lat_lon()

x_test, y_test = read.load_xy_obs(orig_dates, input_length, args.lead,
                                  args.prediction_length, 
                              X_VAR_NAME = "sst", Y_VAR_NAME = "sst", 
                                 DATASET=args.obs_dataset)

x_test = utils.shift_input_maps(x_test, lons, edge_lon = 32.5, pole_padding = 5)
land_mask = utils.load_land_mask(args.model_NN)
_, _, prediction_dates = utils.get_prediction_dates(orig_dates, input_length, args.lead, args.prediction_length)
N_dates = len(prediction_dates)

if DOMAIN == "LAND":
    d_lats, d_lons = np.where(np.equal(land_mask[5:-5,:], 0)) ## land domain lat/lon indices
elif DOMAIN == "OCEAN":
    d_lats, d_lons = np.where(np.equal(land_mask[5:-5,:], 1))
    
## create xarray object to save predictions
empty_array = np.full((N_dates, len(lats), len(lons)), np.nan)
test_predictions = xr.Dataset(
    data_vars=dict(
        var1=(["time", "lat", "lon"], empty_array.copy()), 
        var2=(["time", "lat", "lon"], empty_array.copy()), 
        var3=(["time", "lat", "lon"], empty_array.copy())), 
    coords=dict(
        time = prediction_dates, 
        lat = lats, 
        lon = lons))

print("building model...")
model = mu.build_model(input_shape = (26, 73, 4), output_size = 3,
                           n_conv_blocks = 3, n_filters = 16, kernels = [(3,3), (3,3), (3,3)],
                           n_dense_layers=1, n_neurons = 32,
                           activity_reg = 0, dropout_rate = 0.2,
                           output_activation = "softmax")

print("loading models...")
for i, (i_lat, i_lon) in enumerate(zip(d_lats, d_lons)): 
    
    if i % 25 == 0:
        print(i)
    
    ## check that i_lat/i_lon works for this model
    if np.isnan(y_test[:,i_lat+5, i_lon]).all():
        continue
    
    weight_file = "../processed_data/training/"+args.y_var+"_"+str(args.prediction_length)+\
                  "mo_"+str(args.lead)+"lead_"+args.model_NN+"_trained_models/weights_"+str(lats[i_lat+5])+"_"+\
                         str(lons[i_lon])+"_"+str(args.seed)+".h5"
    if not Path(weight_file).exists():
        continue
    
    model.load_weights(weight_file)
    
    pred = model.predict(x_test, verbose=0)
    test_predictions.var1.loc[dict(lat = lats[i_lat+5], lon = lons[i_lon])] = pred[:,0]
    test_predictions.var2.loc[dict(lat = lats[i_lat+5], lon = lons[i_lon])] = pred[:,1]
    test_predictions.var3.loc[dict(lat = lats[i_lat+5], lon = lons[i_lon])] = pred[:,2]
        
test_predictions.to_netcdf("../processed_data/training/"+args.y_var+"_"+str(args.prediction_length)+\
                  "mo_"+str(args.lead)+"lead_"+args.model_NN+"-trained_"+args.obs_dataset+"-predictions_"+str(args.seed)+".nc")