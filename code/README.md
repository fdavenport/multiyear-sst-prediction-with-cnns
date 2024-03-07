## Description of files: 

### data pre-processing: 
* **process_CMIP6_data.ipynb**: regrid CMIP6 data, calculate GCM ensemble means, mean, sd, terciles
* **process_obs_data.ipynb**: regrid ERSSTv5 data, calculate grid-cell trends, mean, sd, terciles

### training CNNs and making predictions: 
* **keras_tuner_learning_rate.py**, **keras_tuner_dense_layers.py**, **keras_tuner_conv_layers.py**, **keras_tuner_reg.py**: perform hyperparameter tuning using keras tuner
* **train_NN.py**: train convolutional neural network model for every ocean grid point for one GCM and one prediction time period
* **train_NN_GPU.py**: same as train_NN.py, but set up to train using a GPU
* **make_val_predictions.py**, **make_test_predictions.py**, **make_obs_predictions.py**: load trained CNNs and make predictions for validation data, test data, and ERSSTv5 data

Python scripts to train CNNs and make predictions are set up to run from the command line with input arguments specified. Possible arguments (and defaults) are listed at the top of each python script. Example command line code to run a script: 
```bash
$ python train_NN.py --seed=101 --prediction_length=60 --model="MPI-ESM1-2-LR" --y_var="tos"

$ python make_test_predictions.py --seed=101 --lead=0 --prediction_length=60 --model_test="MPI-ESM1-2-LR" --model_NN="MPI-ESM1-2-LR" --y_var="tos"
```

### analysis results and figures: 
* **find_best_CNN.ipynb**: find which CNN random initialization has lowest validation loss at each grid cell for each GCM and lead time
* **calculate_accuracy.ipynb**: calculate accuracy for GCM test data and ERSSTv5 data
* **define_regions.ipynb**: define region boundaries used in analysis
* **Fig_accuracy_maps.ipynb**: create Figure 2, Figure 3, Figures S2-S9 and Figures S12-S13
* **Fig_GCM_ERSSTv5_accuracy_comparison.ipynb**: create Figure 4
* **Fig_hyperparameter_tuning.ipynb**: create Figure S1
* **Fig_persistance_accuracy.ipynb**: create Figures S10, S11 and S14
* **Fig_concurrent_windows_of_opportunity.ipynb**: create Figure S15

