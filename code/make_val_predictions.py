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
parser.add_argument('--model_NN', type=str)
parser.add_argument('--y_var', type=str, default = "tos")
args = parser.parse_args()

if args.y_var in ["tas"]:
    DOMAIN = "LAND"
else:
    DOMAIN = "OCEAN"

## prediction parameters
input_length = [12, 12, 12, 60] ## input tos variables, (number of averaged months for each), in REVERSE chronological order

from project_utils.variant_dict import VARIANT_DICT, val_index
val_variants = VARIANT_DICT[args.model_NN][val_index]

orig_dates = utils.load_dates()
lats, lons = utils.load_lat_lon()

x_val, y_val = read.load_xy_data(orig_dates, input_length, args.lead, args.prediction_length,
                              X_VAR_NAME = "tos", Y_VAR_NAME = args.y_var, VARIANTS = val_variants, 
                              MODEL = args.model_NN)
y_quantiles_val = read.read_data("tos", args.model_NN, 
                             args.prediction_length, VARIANTS = val_variants, stat = "quantiles")

## remove pole latitudes
x_val = utils.shift_input_maps(x_val, lons, edge_lon = 32.5, pole_padding = 5)
y_val = y_val[:,5:-5,:]
y_quantiles_val = y_quantiles_val[:,:,5:-5,:]

land_mask = utils.load_land_mask(args.model_NN)
_, _, prediction_dates = utils.get_prediction_dates(orig_dates, input_length, args.lead, args.prediction_length)
N_dates = len(prediction_dates)

if DOMAIN == "LAND":
    d_lats, d_lons = np.where(np.equal(land_mask[5:-5,:], 0)) ## land domain lat/lon indices
elif DOMAIN == "OCEAN":
    d_lats, d_lons = np.where(np.equal(land_mask[5:-5,:], 1))
    
## create xarray object to save predictions
empty_array = np.full((len(val_variants), N_dates, len(lats), len(lons)), np.nan)
val_predictions = xr.Dataset(
    data_vars=dict(
        var1=(["variant", "time", "lat", "lon"], empty_array.copy()), 
        var2=(["variant", "time", "lat", "lon"], empty_array.copy()), 
        var3=(["variant", "time", "lat", "lon"], empty_array.copy())), 
    coords=dict(
        variant = val_variants, 
        time = prediction_dates, 
        lat = lats, 
        lon = lons))

empty_array = np.full((len(val_variants), len(lats), len(lons)), np.nan)
val_loss = xr.Dataset(
    data_vars=dict(
        val_loss =(["variant", "lat", "lon"], empty_array.copy())), 
    coords=dict(
        variant = val_variants, 
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
    if np.isnan(y_val[:,i_lat, i_lon]).all():
        continue
    
    weight_file = "../processed_data/training/"+args.y_var+"_"+str(args.prediction_length)+\
                  "mo_"+str(args.lead)+"lead_"+args.model_NN+"_trained_models/weights_"+str(lats[i_lat+5])+"_"+\
                         str(lons[i_lon])+"_"+str(args.seed)+".h5"
    if not Path(weight_file).exists():
        continue
    
    model.load_weights(weight_file)
    model.compile(optimizer=
                  tf.keras.optimizers.Adam(learning_rate = 0.0004), 
                  loss = 'categorical_crossentropy')
    q_val = y_quantiles_val[:,:, i_lat, i_lon]
    
    for j, v in enumerate(val_variants): 
        x = x_val[(j*N_dates):((j+1)*N_dates)]
        y = utils.to_quantile_categories(y_val[:,i_lat, i_lon], q_val)[(j*N_dates):((j+1)*N_dates)]

        loss = model.evaluate(x, y, verbose = 0)
        pred = model.predict(x, verbose=0)
            
        val_predictions.var1.loc[dict(variant = v, lat = lats[i_lat+5], lon = lons[i_lon])] = pred[:,0]
        val_predictions.var2.loc[dict(variant = v, lat = lats[i_lat+5], lon = lons[i_lon])] = pred[:,1]
        val_predictions.var3.loc[dict(variant = v, lat = lats[i_lat+5], lon = lons[i_lon])] = pred[:,2]
        val_loss.val_loss.loc[dict(variant = v, lat = lats[i_lat+5], lon = lons[i_lon])] = loss
        
val_predictions.to_netcdf("../processed_data/training/"+args.y_var+"_"+str(args.prediction_length)+\
                  "mo_"+str(args.lead)+"lead_"+args.model_NN+"-trained_"+args.model_NN+"-val_predictions_"+str(args.seed)+".nc")
val_loss.to_netcdf("../processed_data/training/"+args.y_var+"_"+str(args.prediction_length)+\
                  "mo_"+str(args.lead)+"lead_"+args.model_NN+"-trained_"+args.model_NN+"-val_loss_"+str(args.seed)+".nc")
