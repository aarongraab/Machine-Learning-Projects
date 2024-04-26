##Project Summary:
This project requires analyzing a dataset with 99 numeric input features and one numeric label across 48 instances to generate regression models. 
I was expected to explore different regression methods and preprocessing techniques using scikit-learn, applying cross-validation to select and assess the best performing approach.

##Project Goals:
-To explore and apply preprocessing techniques on a dataset.
-To become familiar with various regression approaches in scikit-learn.
-To implement regression methods and evaluate them using cross-validation.

##Project #2 Directory Outline:
-project2.py
-RMSE Regression Methods Plot.

##Expected Inputs:
-TO PROTECT THE PRIVACY OF THE PROJECT, THE REAL DATA FILE WILL NOT BE INCLUDED.
-A single data file containing 99 features labeled X1-X99, across 48 rows, each row representing a unique instance of data.
-Each row in the datafile includes a Y column indicating the numeric output tasked to predict.

##Libraries Used:
-os: interacting with operating systems.
-pandas: fata loading and manipulation.
-numpy: numerical operations.
-matplotlib.pylab: plotting graphs.
-sklearn.model_selection: cross-validation utilities.
-sklearn.linear_model: linear regression models.
-sklearn.preprocessing: data scaling.
-sklearn.ensemble: used for gradient boosting.

##Machine Learning Techniques Used:
-Linear Regression
-Gradient Boosting Regressor
-Cross Validation using Leave-One-Out

##Graphs Plotted:
-RMSE Regression Methods Plot.

##Console Print Statement Outputs
-Baseline Model (Linear Regression, LOO-CV): 2.165 ± 1.731
-Model 1 (Linear Regression with StandardScaler, LOO-CV): 1.862 ± 1.843
-Model 2 (Gradient Boosting Regressor, LOO-CV): 0.582 ± 0.451
