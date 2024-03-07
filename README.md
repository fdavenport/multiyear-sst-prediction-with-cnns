# multiyear SST prediction with CNNs

data and code for multiyear prediction

## Repository Organization
* **input_data**: placeholder directory for raw data (not included in repository, but publicly available. see details below)
* **code**: jupyter notebooks and python scripts to read and pre-process data, train models, analyze results, and make figures
* **processed_data**: processed data from analysis
* **project_utils**: python functions used in code
* **figures**: placeholder directory for figure pdfs 
* environment.yml - specifies python packages needed to run notebooks

## Data
Data used in the analysis is publicly available from the following sources: 

* **CMIP6 historical simulations:** available through the Earth System Grid Federation ([https://aims2.llnl.gov/search/cmip6/](https://aims2.llnl.gov/search/cmip6/))
* **NOAA Extended Reconstructed SST v5 dataset:** available through NOAA ([https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html))

## Steps to set up the environment and install python functions
1. install the required python modules using conda. The environment.yml provides information on the required modules. 
2. install project_utils in conda environment with the following command (should be run from within the main project directory): 
```bash
pip install -e . --user
```
