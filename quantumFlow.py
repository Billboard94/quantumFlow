import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

def replace_i_with_j(matlab_complex):
    return complex(str(matlab_complex).replace('i', 'j'))

df = pd.read_csv('trainingdata_probe.csv', sep=",", decimal=".")            # Read the training data set
rho_index = df.columns.get_loc('rho')
rho = df.iloc[:, rho_index :]
rho = np.vectorize(replace_i_with_j)(rho)
input = df.iloc[:, : rho_index]
real_rho = [num.real for num in rho]
imag_rho = [num.imag for num in rho]
rho = np.column_stack((real_rho,imag_rho))

## Split the data into training data and test data
input_train, input_test, rho_train, rho_test = train_test_split(input, rho, train_size=0.8, random_state=100)

## Build the Model
 # Linear Regression
lr = LinearRegression()
lr.fit(input_train, rho_train)

## Do Predictions
rho_lr_train_pred = lr.predict(input_train)
rho_lr_test_pred = lr.predict(input_test)

## Evaluate Model Performance
lr_train_mse = mean_squared_error(rho_train, rho_lr_train_pred)
lr_train_r2 = r2_score(rho_train, rho_lr_train_pred)

lr_test_mse = mean_squared_error(rho_test, rho_lr_test_pred)
lr_test_r2 = r2_score(rho_test, rho_lr_test_pred)

# Write results in table
lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test_R2']


## Build Random Forest Model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(input_train, rho_train) 

rho_rf_train_pred = rf.predict(input_train)
rho_rf_test_pred = rf.predict(input_test)

rf_train_mse = mean_squared_error(rho_train, rho_rf_train_pred)
rf_train_r2 = r2_score(rho_train, rho_rf_train_pred)

rf_test_mse = mean_squared_error(rho_test, rho_rf_test_pred)
rf_test_r2 = r2_score(rho_test, rho_rf_test_pred)

rf_results = pd.DataFrame(['RandomForest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test_R2']

## Combine the result tables
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

## Plot evaluation results
plt.scatter(x=rho_train, y=rho_lr_train_pred, c="#7CAE00",alpha=0.3)

# Flatten the multidimensional arrays to 1D vectors
rho_train_flat = np.ravel(rho_train)
rho_lr_train_pred_flat = np.ravel(rho_lr_train_pred)

# Perform polynomial fitting on the flattened vectors
z = np.polyfit(rho_train_flat, rho_lr_train_pred_flat, 1)
p = np.poly1d(z)
print(p)
# Plot using the flattened arrays
plt.scatter(x=rho_train_flat, y=rho_lr_train_pred_flat, c="#7CAE00", alpha=0.3)
plt.plot(rho_train_flat, p(rho_train_flat), "#F8766D")
plt.ylabel('Predicted rho')
plt.xlabel('DG rho')