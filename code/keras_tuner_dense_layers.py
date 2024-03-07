import numpy as np
import tensorflow as tf
import keras_tuner as kt
import random
import argparse

from project_utils import utils
from project_utils import model_utils as mu
from project_utils import read_utils as read

BATCH_SIZE = 32

parser = argparse.ArgumentParser()
parser.add_argument('--lat', type=float)
parser.add_argument('--lon', type=float)
parser.add_argument('--prediction_length', type=int, default=60)
parser.add_argument('--model', type=str)
parser.add_argument('--y_var', type=str, default = "tos")
parser.add_argument('--overwrite', action='store_true') #don't overwrite unless overwrite specified
args = parser.parse_args()

lead = 0 #months 
input_length = [12, 12, 12, 60] ## input tos variables, (number of averaged months for each), in REVERSE chronological order

from project_utils.variant_dict import VARIANT_DICT, train_index, val_index
train_variants = VARIANT_DICT[args.model][train_index]
val_variants = VARIANT_DICT[args.model][val_index]

orig_dates = utils.load_dates()
lats, lons = utils.load_lat_lon()

# # --- load data -----

x_train, y_train = read.load_xy_data(orig_dates, input_length, lead, 
                                     args.prediction_length, 
                              X_VAR_NAME = "tos", Y_VAR_NAME = args.y_var, 
                                     VARIANTS = train_variants, 
                              MODEL = args.model)

x_val, y_val = read.load_xy_data(orig_dates, input_length, lead, 
                                 args.prediction_length, 
                              X_VAR_NAME = "tos", Y_VAR_NAME = args.y_var,
                                 VARIANTS = val_variants, 
                              MODEL = args.model)

x_train = utils.shift_input_maps(x_train, lons, edge_lon = 32.5, pole_padding = 5)
x_val = utils.shift_input_maps(x_val, lons, edge_lon = 32.5, pole_padding = 5)

# format y_data
i = np.where(lats == args.lat)[0][0]
j = np.where(lons == args.lon)[0][0]
y_quantiles_train = read.read_data(args.y_var, args.model, 
                             args.prediction_length, VARIANTS = train_variants, stat = "quantiles")
y_quantiles_val = read.read_data(args.y_var, args.model, 
                             args.prediction_length, VARIANTS = val_variants, stat = "quantiles")
q_train = y_quantiles_train[:,:, i, j]
q_val = y_quantiles_val[:,:, i, j]

y_train = utils.to_quantile_categories(y_train[:,i,j], q_train)
y_val = utils.to_quantile_categories(y_val[:,i,j], q_val)

# # -------

def model_builder(hp):
    
    n_dense_layers = hp.Choice("n_dense_layers", [1, 2, 3])
    n_neurons = hp.Choice("n_neurons", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    
    np.random.seed(101) 
    random.seed(101)
    tf.random.set_seed(101)

    model = mu.build_model(n_conv_blocks = 3, 
                           n_filters = 16, 
                           kernels = [(3,3), (3,3), (3,3)], 
                           n_dense_layers = n_dense_layers, 
                           n_neurons = n_neurons, 
                           SEED = 101, 
                           dropout_rate = 0.3, 
                           activity_reg = 0)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0004), 
                  loss = 'categorical_crossentropy')
    
    return model

tuner = kt.GridSearch(
    hypermodel=model_builder,
    objective="val_loss",
    max_trials=50,
    seed=101,
    executions_per_trial=1,
    overwrite=True, 
    directory="../processed_data/kt_results/kt_dense",
    project_name="kt_60mo_"+args.model+'_'+str(args.lat)+'_'+str(args.lon)) 

tuner.search_space_summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

lr_callback = tf.keras.callbacks.LearningRateScheduler(mu.lr_scheduler)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='../processed_data/kt_logs/kt_dense/kt_60mo_'+args.model+'_'+str(args.lat)+'_'+str(args.lon))

tuner.search(x_train, y_train, epochs=250, verbose=1, 
             batch_size = BATCH_SIZE, steps_per_epoch = 100,  
             validation_data=(x_val, y_val),
             callbacks=[tensorboard_callback, 
            early_stopping_callback, lr_callback])
