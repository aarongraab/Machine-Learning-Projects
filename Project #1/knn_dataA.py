import pandas as pd
import numpy as np
import os

import matplotlib.pylab as plt 

from sklearn.metrics import accuracy_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

FOLDS = 10
fold_counter = 0

mean_recall = np.linspace(0, 1, 100)

# Load the dataset
current_dir = os.path.dirname(os.path.realpath(__file__))  
data_path = os.path.join(current_dir, 'A1_dataA.tsv')  

data = pd.read_csv(data_path, sep='\t')

# Set values from data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

average_precision_list = []
accuracy_list = []
f1_score_list = []
precision_list = []
recall_list = []
interp_precision_list = []

# 10 fold stratified cross-validation
cv = StratifiedKFold(n_splits=FOLDS)

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Cross-validation - 10 fold
for train, test in cv.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    
    knn.fit(X_train, y_train)
    y_score = knn.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = auc(recall, precision)
    average_precision_list.append(ap)
    
    y_predicted = knn.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_predicted))
    f1_score_list.append(f1_score(y_test, y_predicted))
    precision_list.append(precision_score(y_test, y_predicted))
    recall_list.append(recall_score(y_test, y_predicted))
    
    interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
    interp_precision_list.append(interp_precision)
    
    # Plot each fold's Precision-Recall curve
    plt.plot(recall, precision, alpha=0.3, label=f'Fold {fold_counter+1} Precision-Recall curve')
    fold_counter += 1

# Mean precision
mean_precision = np.mean(interp_precision_list, axis=0)

# Plot mean curve
plt.plot(mean_recall, mean_precision, label=f'Mean Precision-Recall (AP={np.mean(average_precision_list):.2f})', color='blue', linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best', fontsize='small')
plt.show()

# Printing relevant data
print(f'Average Precision: {np.mean(average_precision_list):.3f}')
print(f'Accuracy: {np.mean(accuracy_list):.3f}')
print(f'F1-Score: {np.mean(f1_score_list):.3f}')
print(f'Precision: {np.mean(precision_list):.3f}')
print(f'Recall: {np.mean(recall_list):.3f}')


# Code from COMP 3202 Resources
'''
f, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(X[y==0,0], X[y==0,1], color='blue', s=2, label='y=0')
axes[0].scatter(X[y!=0,0], X[y!=0,1], color='red', s=2, label='y=1')
axes[0].set_xlabel('X[:,0]')
axes[0].set_ylabel('X[:,1]')
axes[0].legend(loc='lower left', fontsize='small')

k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=12345)
predictor = SVC(kernel='linear', C=1.0, probability=True, random_state=12345)

y_real = []
y_proba = []
for i, (train_index, test_index) in enumerate(k_fold.split(X)):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    predictor.fit(Xtrain, ytrain)
    pred_proba = predictor.predict_proba(Xtest)
    precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
    lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
    axes[1].step(recall, precision, label=lab)
    y_real.append(ytest)
    y_proba.append(pred_proba[:,1])

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Overall AUC=%.4f' % (auc(recall, precision))
axes[1].step(recall, precision, label=lab, lw=2, color='black')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left', fontsize='small')

f.tight_layout()
plt.show()
'''