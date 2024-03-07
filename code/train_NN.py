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
parser.add_argument('--model', type=str)
parser.add_argument('--y_var', type=str, default = "tos")
parser.add_argument('--overwrite', action='store_true') #don't overwrite unless overwrite specified
args = parser.parse_args()

if args.y_var in ["tas", "pr"]:
    DOMAIN = "LAND"
else:
    DOMAIN = "OCEAN"

input_length = [12, 12, 12, 60] ## input tos variables, (number of averaged months for each), in REVERSE chronological order

from project_utils.variant_dict import VARIANT_DICT, train_index, val_index
train_variants = VARIANT_DICT[args.model][train_index]
val_variants = VARIANT_DICT[args.model][val_index]

orig_dates = utils.load_dates()
lats, lons = utils.load_lat_lon()

# # ----------------------------------------
# # LOAD TRAINING DATA ---------------------

x_train, y_train = read.load_xy_data(orig_dates, input_length, args.lead, args.prediction_length,
                              X_VAR_NAME = "tos", Y_VAR_NAME = args.y_var, VARIANTS = train_variants, 
                              MODEL = args.model)

x_val, y_val = read.load_xy_data(orig_dates, input_length, args.lead, args.prediction_length,
                              X_VAR_NAME = "tos", Y_VAR_NAME = args.y_var, VARIANTS = val_variants, 
                              MODEL = args.model)

# # ----------------------------------------
# # CONVERT TRAINING DATA ------------------

## shift input data so "edge" of input maps doesn't bisect ocean basins
## remove 5 grid rows at each pole (where SSTs have essentially no variability to learn or predict)
x_train = utils.shift_input_maps(x_train, lons, edge_lon = 32.5, pole_padding = 5)
x_val = utils.shift_input_maps(x_val, lons, edge_lon = 32.5, pole_padding = 5)

# remove pole latitudes
y_train = y_train[:,5:-5,:] 
y_val = y_val[:,5:-5,:]

y_quantiles_train = read.read_data(args.y_var, args.model, 
                             args.prediction_length, VARIANTS = train_variants, stat = "quantiles")
y_quantiles_val = read.read_data(args.y_var, args.model, 
                             args.prediction_length, VARIANTS = val_variants, stat = "quantiles")

y_quantiles_train = y_quantiles_train[:,:,5:-5,:] 
y_quantiles_val = y_quantiles_val[:,:,5:-5,:]

land_mask = utils.load_land_mask(args.model)
## ----------------------------------------
## TRAIN MODELS ---------------------------

if DOMAIN == "LAND":
    d_lats, d_lons = np.where(np.equal(land_mask[5:-5,:], 0)) ## land domain lat/lon indices
elif DOMAIN == "OCEAN":
    d_lats, d_lons = np.where(np.equal(land_mask[5:-5,:], 1))

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                           patience=5, restore_best_weights = True)
lr_callback = tf.keras.callbacks.LearningRateScheduler(mu.lr_scheduler)

## create output directory if necessary
output_dir = "../processed_data/training/"+args.y_var+"_"+str(args.prediction_length)+\
                  "mo_"+str(args.lead)+"lead_"+args.model+"_trained_models/"
if not Path(output_dir).exists():
    Path(output_dir).mkdir(parents=True)

## loop through lat/lon values to train individual models
for i_lat, i_lon in zip(d_lats, d_lons): 

    ## subset y data
    y_sub_train = y_train[:,i_lat, i_lon]
    if np.isnan(y_sub_train).all():
        continue
    q_train = y_quantiles_train[:,:, i_lat, i_lon]
    q_val = y_quantiles_val[:,:, i_lat, i_lon]

    y_sub_train = utils.to_quantile_categories(y_train[:,i_lat, i_lon], q_train)
    y_sub_val = utils.to_quantile_categories(y_val[:,i_lat, i_lon], q_val)


    weight_file = output_dir+"weights_"+str(lats[i_lat+5])+"_"+\
                         str(lons[i_lon])+"_"+str(args.seed)+".h5"
    print(i_lat, i_lon)
    if not args.overwrite and Path(weight_file).exists(): 
        continue
        
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    ## train model
    model = mu.build_model(SEED = args.seed, 
                           input_shape = (26, 73, 4), output_size = 3, 
                           n_conv_blocks = 3, n_filters = 16, kernels = [(3,3), (3,3), (3,3)], 
                           n_dense_layers=1, n_neurons = 32, 
                           activity_reg = 0, dropout_rate = 0.2, 
                           output_activation = "softmax")

    model.compile(optimizer=
                  tf.keras.optimizers.Adam(learning_rate = 0.0004), 
                  loss = 'categorical_crossentropy')

    history = model.fit(x_train, y_sub_train, 
                        steps_per_epoch = 100, epochs = 250,
                        batch_size = 32, verbose = 1, 
                        validation_data = [x_val, y_sub_val], 
                        callbacks = [early_stopping_callback, lr_callback])

    ## save history
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = output_dir+"history_"+str(lats[i_lat+5])+"_"+\
                         str(lons[i_lon])+"_"+str(args.seed)+".csv"
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    ## save model
    model.save_weights(weight_file)
