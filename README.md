# Foursquare Location Matching

#### Authors : Youri Matiounine, Vincent Schuler, Théo Viel

4th place solution for the [Foursquare Location Matching competition](https://www.kaggle.com/c/foursquare-location-matching)



## Solution Overview

![](img.png)

The solution write-up is available [here](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335810)

The solution shared here contain's @theo's pipeline. It is slightly more complicated than the actual pipeline, but somehow runs out of memory during inference. 
The submitted model does not use features from `fe_theo.py`, and uses less candidates.

## Data

- Competition data is available on the [competition page](https://www.kaggle.com/c/foursquare-location-matching/data)

- Ressources & dictionaries are available [on Kaggle](https://www.kaggle.com/datasets/theoviel/foursquare-data)


## Using the repository

### Notebooks

Notebooks contain our training pipeline, they need to be run in the following order :

- `Cleaning.ipynb` : Processes the data
- `Matching.ipynb` : Creates pairs
- `Level 1.ipynb`  : Creates features for the level 1 model
- `Classification.ipynb` : Using `LEVEL = 1`, trains the prefiltering level 1 model
- `Level 2.ipynb`  : Creates features for the level 2 model
- `Classification.ipynb` : Using `LEVEL = 2`, trains the final model

### Code structure

```
src
├── inference
│   └── main.py             # Boosting inference functions
├── model_zoo
│   ├── catboost.py         # To train a Catboost model
│   ├── lgbm.py             # To train a LightGBM model
│   └── xgb.py              # To train a XGBoost model
├── training           
│   └── main_boosting.py    # Boosting training functions
├── utils 
│   ├── logger.py           # Logging utils
│   └── plot.py             # Plotting utils
├── cleaning.py             # Functions for data cleaning           
├── dtypes.py               # Handling pandas dataframe dtypes
├── fe_theo.py              # Theo's features
├── fe.py                   # Youri & Vincent's features
├── matching.py             # Functions for pairs matching
├── params.py               # Parameters
├── pp.py                   # Post-processing utils
└── ressources.py           # Ressources used for cleaning / matching
```
