##Project Summary:
To familiarize oneself with various classification techniques in scikit-learn which involves working with a binary classification problem. 
The dataset provided includes 404 features and a binary class label for each of 3000 instances. 
Model performance assessed using 10-fold stratified cross-validation and optimized models using grid-search. 
The final goal was to predict the likelihood of instances belonging to a certain class in a separate test dataset and output these predictions in a specified format.

##Project Goals:
-Gain practical experience with classification methods in scikit-learn.
-Use cross-validation to evaluate and select the best performing model.
-Generate and interpret Precision-Recall and ROC curves to assess model performance.
-Produce a set of predicted likelihoods for a test dataset based on the best-performing model.

##Project #3 Directory Outline:
-project3.py
-Precision-Recall Plot
-ROC Plot

##Expected Inputs:
-TO PROTECT THE PRIVACY OF THE PROJECT, THE REAL DATA FILES WILL NOT BE INCLUDED.
-TrainingData File
-TestData File

##Libraries Used:
-os: interacting with operating systems.
-pandas: fata loading and manipulation.
-numpy: numerical operations.
-matplotlib.pylab: plotting graphs.
-sklearn.model_selection: cross-validation utilities.
-sklearn.linear_model: linear regression models.
-sklearn.preprocessing: data scaling.
-sklearn.ensemble: used for gradient boosting.
-sklearn.metrics: computing various performance metrics.
-sklearn.svm: svm modeling

##Machine Learning Techniques Used:
-Logistic Regression
-Random Forest
-SVM
-10-Fold Stratified Cross Validation

##Graphs Plotted:
-Precision-Recall Curve
-ROC Curve

##Console Print Statement Outputs
Class Proportions:
0    0.924
1    0.076

Average accuracy: 0.9077
Standard deviation: 0.0145
95% confidence interval: [0.8786, 0.9367]

Best parameters for Random Forest: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
Best accuracy for Random Forest: 0.92400
Results for each combination of parameters:
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': None, 'min_samples_split': 10, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 20, 'min_samples_split': 10, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 5, 'n_estimators': 300}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 100}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 200}
0.92400 (+/-0.00267) for {'max_depth': 30, 'min_samples_split': 10, 'n_estimators': 300}


Best parameters for SVM: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
Best accuracy for SVM: 0.92467
Results for each combination of parameters:
0.92400 (+/-0.00267) with: {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
0.91833 (+/-0.01342) with: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
0.92400 (+/-0.00267) with: {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf'}
0.91833 (+/-0.01342) with: {'C': 0.1, 'gamma': 'auto', 'kernel': 'linear'}
0.92400 (+/-0.00267) with: {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
0.90700 (+/-0.03353) with: {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
0.92400 (+/-0.00267) with: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
0.90700 (+/-0.03353) with: {'C': 1, 'gamma': 'auto', 'kernel': 'linear'}
0.92467 (+/-0.00442) with: {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
0.90133 (+/-0.03441) with: {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}
0.92433 (+/-0.00427) with: {'C': 10, 'gamma': 'auto', 'kernel': 'rbf'}
0.90133 (+/-0.03441) with: {'C': 10, 'gamma': 'auto', 'kernel': 'linear'}