import pandas as pd
import os
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split

# Load Data
current_dir = os.path.dirname(os.path.realpath(__file__))  
data_test_path = os.path.join(current_dir, 'A3_TestData.tsv')
data_train_path = os.path.join(current_dir, 'A3_TrainData.tsv')    

data_test = pd.read_csv(data_test_path, sep = '\t')
data_train = pd.read_csv(data_train_path, sep = '\t')

# Class Distribution
class_distribution = data_train['label'].value_counts(normalize = True)
print(class_distribution)

X_train = data_train.drop('label', axis = 1)
y_train = data_train['label']

# Initializ Model
logreg = LogisticRegression(max_iter = 1000) 

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits = 10)

# Evaluate Model
scores = cross_val_score(logreg, X_train, y_train, cv = cv, scoring = 'accuracy')

# Calculate Mean and STD
mean_accuracy = scores.mean()
std_dev_accuracy = scores.std()

print(f"Average accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation: {std_dev_accuracy:.4f}")
print(f"95% confidence interval: [{mean_accuracy - 2 * std_dev_accuracy:.4f}, {mean_accuracy + 2 * std_dev_accuracy:.4f}]")


def predictions_to_text():
    # Initialize RF with best parameter
    rf_best = RandomForestClassifier(max_depth = None, min_samples_split = 2, n_estimators = 100, random_state = 42)
    rf_best.fit(X_train, y_train)

    X_test = data_test

    # Test Data Probability Predictions
    test_probabilities = rf_best.predict_proba(X_test)

    # Probability Extractions to Class 1
    likelihood_class_1 = test_probabilities[:, 1]

    # File Path
    file_path = os.path.join(current_dir, 'predictions.txt') 

    np.savetxt(file_path, likelihood_class_1, fmt = '%f')

    print(f"Predictions saved to {file_path}")


def random_forest():
    param_grid_rf = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split an internal node
    }

    rf = RandomForestClassifier()
    grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, cv = cv, scoring = 'accuracy', n_jobs = -1)

    # Fit GridSearchCV
    grid_search_rf.fit(X_train, y_train)

    print("Best parameters for Random Forest:", grid_search_rf.best_params_)
    print("Best accuracy for Random Forest: {:.5f}".format(grid_search_rf.best_score_))

    # Print results - including mean and standard deviation
    print("\nResults for each combination of parameters:")
    for mean_score, std_score, params in zip(grid_search_rf.cv_results_['mean_test_score'], grid_search_rf.cv_results_['std_test_score'], grid_search_rf.cv_results_['params']):
        print("{:.5f} (+/-{:.5f}) for {}".format(mean_score, std_score * 2, params))


def svm():
    param_grid_svm = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'gamma': ['scale', 'auto'],  # Kernel coefficient
        'kernel': ['rbf', 'linear']  # Kernel type 
    }

    svm = SVC()
    grid_search_svm = GridSearchCV(estimator = svm, param_grid = param_grid_svm, cv = cv, scoring = 'accuracy', n_jobs = -1)

    # Fit GridSearchCV
    grid_search_svm.fit(X_train, y_train)

    print("Best parameters for SVM:", grid_search_svm.best_params_)
    print("Best accuracy for SVM: {:.5f}".format(grid_search_svm.best_score_))

    # Print results - including mean and standard deviation
    print("\nResults for each combination of parameters:")
    for mean_score, std_score, params in zip(grid_search_svm.cv_results_['mean_test_score'], grid_search_svm.cv_results_['std_test_score'], grid_search_svm.cv_results_['params']):
        print("{:.5f} (+/-{:.5f}) with: {}".format(mean_score, std_score * 2, params))


def plots():
    # Best Random Forest Model
    rf_best = RandomForestClassifier(max_depth = None, min_samples_split = 2, n_estimators = 100)
    rf_best.fit(X_train, y_train)

    # Best SVM Model
    svm_best = SVC(C = 10, gamma = 'scale', kernel = 'rbf', probability = True)
    svm_best.fit(X_train, y_train)

    # Baseline Model
    logreg_best = LogisticRegression(max_iter = 1000)
    logreg_best.fit(X_train, y_train)

    models = {
        'Random Forest': rf_best,
        'SVM': svm_best,
        'Baseline (Logistic Regression)': logreg_best
    }

    # Training Data
    no_skill = len(y_train[y_train == 1]) / len(y_train)

    # Precision-Recall Plot
    plt.figure(figsize = (8, 6))
    plt.plot([0, 1], [no_skill, no_skill], linestyle = '--', label = 'No Skill', color = 'black')

    for name, model in models.items():
        # Check if model supports probability estimates
        if hasattr(model, "predict_proba"):
            probas_ = model.predict_proba(X_train)[:, 1]
        else:  # Use decision function for models like SVM without predict_proba
            probas_ = model.decision_function(X_train)
            probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())
        
        precision, recall, _ = precision_recall_curve(y_train, probas_)
        avg_precision = average_precision_score(y_train, probas_)
        plt.plot(recall, precision, marker = '.', label = f'{name} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    # ROC Plot
    plt.figure(figsize = (8, 6))
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'No Skill', color = 'black')

    for name, model in models.items():
        # Check if model supports probability estimates
        if hasattr(model, "predict_proba"):
            probas_ = model.predict_proba(X_train)[:, 1]
        else:  # Use decision function for models like SVM without predict_proba
            probas_ = model.decision_function(X_train)
            probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())
        
        fpr, tpr, _ = roc_curve(y_train, probas_)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, marker = '.', label = f'{name} (AUROC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    random_forest()
    svm()
    plots()
    predictions_to_text()