#!/usr/bin/env python3

import pandas as pd
import os
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

# read in the data
dd = pd.read_csv(
    r"/capstor/scratch/cscs/smeier/eff_wind_regression/effective_wind_const.csv"
)

# format the columns
dd["time"] = pd.to_datetime(dd["time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
dd["CH4_mass"] = dd["integrated_CH4_mass"] / dd["L_max"]

# select rows and features
dd_rm_index = dd[dd["index"] >= 10]

filtered_subset = dd_rm_index.dropna(
    subset=[
        "wind_speed",
        "effective_wind_speed",
        "emission_height",
        "surface_roughness",
        "monin_obukhov_length",
        "CH4_mass",
    ]
)

# Prepare the independent variables (X) and the dependent variable (y)
X = filtered_subset[
    [
        "wind_speed",
        "emission_height",
        "surface_roughness",
        "monin_obukhov_length",
        "CH4_mass",
    ]
]
y = filtered_subset["effective_wind_speed"]

print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Calling the TPOT Classifier
tpot = TPOTRegressor(
    generations=100,
    population_size=100,
    cv=5,
    verbosity=2,
    n_jobs=-1,
)
# Run TPOT
tpot.fit(X_train, y_train)
# Get the score
tpot.score(X_test, y_test)

# Define the base filename
base_filename = "/capstor/scratch/cscs/smeier/eff_wind_regression/tpot_ueff_pipeline"
file_extension = ".py"
filename = base_filename + file_extension

# Check if the file already exists and create a new filename if necessary
counter = 1
while os.path.exists(filename):
    filename = f"{base_filename}_{counter}{file_extension}"
    counter += 1

# Export the best pipeline
tpot.export(filename)
