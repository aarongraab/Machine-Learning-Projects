import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir, 'A2data.tsv')
data = pd.read_csv(data_path, sep = '\t', index_col=0)

X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Baseline with cross validation: Leave-One-Out
model_baseline = LinearRegression()
loo = LeaveOneOut()  
scores_baseline = cross_val_score(model_baseline, X, y, cv = loo, scoring = 'neg_mean_squared_error')
rmse_baseline = np.sqrt(-scores_baseline)

# Model 1: Linear Regression using Standard Scaler
model1 = make_pipeline(StandardScaler(), LinearRegression())
scores_model1 = cross_val_score(model1, X, y, cv = loo, scoring = 'neg_mean_squared_error')
rmse_model1 = np.sqrt(-scores_model1)

# Model 2: Gradient Boosting Regressor
model2 = GradientBoostingRegressor()
scores_model2 = cross_val_score(model2, X, y, cv = loo, scoring = 'neg_mean_squared_error')
rmse_model2 = np.sqrt(-scores_model2)

# Print Statements
print(f"Baseline Model (Linear Regression, LOO-CV): {rmse_baseline.mean():.3f} ± {rmse_baseline.std():.3f}")
print(f"Model 1 (Linear Regression with StandardScaler, LOO-CV): {rmse_model1.mean():.3f} ± {rmse_model1.std():.3f}")
print(f"Model 2 (Gradient Boosting Regressor, LOO-CV): {rmse_model2.mean():.3f} ± {rmse_model2.std():.3f}")

# Plot RMSE score
data_to_plot = [rmse_baseline, rmse_model1, rmse_model2]

# Graphs data
plt.boxplot(data_to_plot, labels = ['Baseline', 'Model 1', 'Model 2'])
plt.ylabel('RMSE')
plt.title('Assignment 2: Exploring Regression Methods')
plt.show()
