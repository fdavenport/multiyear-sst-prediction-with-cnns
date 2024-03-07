import numpy as np
import xarray as xr
import tensorflow as tf

### -----------------------------------------------------------------------
### tensorflow model building and training functions

def mask_input(inputs):
    """ convert any nan inputs to zeros """
    ds = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
    return(ds)

def build_model(input_shape = (26, 73, 4), SEED = 101, 
                             n_conv_blocks = 3, n_filters = 16, 
                             n_dense_layers = 3, n_neurons = 32, 
                             dropout_rate = 0.2, activity_reg = 0,  
                             output_size = 3,  output_activation = "softmax", 
                             kernels = [(3,3), (3,3), (3,3)]):
    """ """
    tf.keras.utils.set_random_seed(SEED)
    
    ## define input shape
    input_layer = tf.keras.layers.Input(shape=input_shape, name = "input") 
        
    layers = tf.keras.layers.Lambda(mask_input, name = "input_mask")(input_layer) 
            
    ## CONV BLOCKS ###
    for num_lay in range(n_conv_blocks):
        layers = tf.keras.layers.Conv2D(n_filters, kernels[num_lay], 
                                            padding='same', 
                                            activation = 'relu', 
                                            name = "Conv2D_"+str(num_lay),
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                        kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED))(layers)
        layers = tf.keras.layers.AveragePooling2D((2,2), 
                                              padding='same', 
                                              name = "Pool_"+str(num_lay))(layers)

    layers = tf.keras.layers.Flatten()(layers)
    
    ## DENSE LAYERS ###
    for num_lay in range(n_dense_layers):
        layers = tf.keras.layers.Dense(n_neurons, activation = 'relu', 
                                       name = "Dense_"+str(num_lay),
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                        kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED), 
                                      kernel_regularizer=tf.keras.regularizers.L2(activity_reg))(layers)
        layers = tf.keras.layers.Dropout(dropout_rate)(layers)
        
    output_layer = tf.keras.layers.Dense(output_size, activation=output_activation, 
                                    name = "prediction_layer", 
                        bias_initializer=tf.keras.initializers.RandomNormal(seed=SEED),
                        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=SEED))(layers)
   
    model = tf.keras.models.Model(input_layer, output_layer)
    
    return(model)
    
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.05)

def lr_scheduler_transfer(epoch, lr):
    return lr * tf.math.exp(-0.01)
