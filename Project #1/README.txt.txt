##Project Summary:
The project involves analyzing two datasets to determine if they contain patterns that can be leveraged by machine learning methods for label discrimination. 
The task requires generating visualizations to explore data features, formulating hypotheses about model performance, implementing a Python program for 10-fold 
stratified cross-validation using the KNN classification method, and validating these hypotheses based on the obtained metrics.

##Project Goals:
-To practice data visualization for exploratory analysis.
-To assess the applicability of machine learning on given datasets.
-To develop and implement a simple ML pipeline using KNN and cross-validation in scikit-learn.

##Project #1 Directory Outline:
-Folder for graphs relating to Dataset A.
-Folder for graphs relating to Dataset B.
-knn_dataA.py file.
-knn_dataB.py file.
-Generated plots from code files.

##Expected Inputs:
-TO PROTECT THE PRIVACY OF THE PROJECT, THE REAL DATA FILES WILL NOT BE INCLUDED.
-Dataset A and B contain values between 0-1 for X1, X2, X3, X4, and X5 instances.
-Dataset A and B contain the respected class, either 0 or 1, for each row of five instances.

##Libraries Used:
-pandas: data loading and manipulation.
-numpy: numerical operations.
-matplotlib.pylab: plotting graphs.
-sklearn.metrics: computing various performance metrics.
-sklearn.neighbors: KNN classifier.
-sklearn.model_selection: cross-validation utilities.

##Machine Learning Techniques Used:
-KNN Classification
-10-fold stratified CV

##Graphs Plotted:
-DataSetA Precision-Recall Curve with an average curve across all folds.
-DataSetB Precision-Recall Curve with an average curve across all folds.

##Console Print Statement Outputs
-DatasetA:
--Average Precision: 0.677
--Accuracy: 0.600
--F1-Score: 0.542
--Precision: 0.560
--Recall: 0.560

-DatasetB:
--Average Precision: 0.899
--Accuracy: 0.790
--F1-Score: 0.747
--Precision: 0.893
--Recall: 0.680