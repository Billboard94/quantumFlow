import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('trainingdata_small.csv')            # Read the training data set
# print(df)
# rho = df.iloc[:,-10:].to_numpy()
input = df.iloc[:,:3].to_numpy()
print(input)
# print(input)

## Split the data into training data and test data
# input_train, input_test, rho_train, rho_test = train_test_split(input, rho, train_size=0.2, random_state=100)

# ## Build the Model
#  # Linear Regression
# lr = LinearRegression()
# lr.fit(input_train, rho_train)

# ## Do Predictions
# rho_lr_train_pred = lr.predict(input_train)
# rho_lr_test_pred = lr.predict(input_test)

# ## Evaluate Model Performance
# lr_train_mse = mean_squared_error(rho_train, rho_lr_train_pred)
# lr_train_r2 = r2_score(rho_train, rho_lr_train_pred)

# lr_test_mse = mean_squared_error(rho_test, rho_lr_test_pred)
# lr_test_r2 = r2_score(rho_test, rho_lr_test_pred)

# # Write results in table
# lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
# lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test_R2']


# ## Build Random Forest Model
# rf = RandomForestRegressor(max_depth=2, random_state=100)
# rf.fit(input_train, rho_train) 

# rho_rf_train_pred = rf.predict(input_train)
# rho_rf_test_pred = rf.predict(input_test)

# rf_train_mse = mean_squared_error(rho_train, rho_rf_train_pred)
# rf_train_r2 = r2_score(rho_train, rho_rf_train_pred)

# rf_test_mse = mean_squared_error(rho_test, rho_rf_test_pred)
# rf_test_r2 = r2_score(rho_test, rho_rf_test_pred)

# rf_results = pd.DataFrame(['RandomForest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
# rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test_R2']

# ## Combine the result tables
# df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

# ## Plot evaluation results
# plt.scatter(x=rho_train, y=rho_lr_train_pred, c="#7CAE00",alpha=0.3)

# z = np.polyfit(rho_train, rho_lr_train_pred, 1)
# p = np.poly1d(z)

# plt.plot(rho_train, p(rho_train), "#F8766D")
# plt.ylabel('Predicted rho')
# plt.xlabel('DG rho')