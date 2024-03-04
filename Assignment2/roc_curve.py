# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: roc_curve.py
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the dataset cheat_data.csv and prepare the data_training numpy array
df = pd.read_csv('DataMiningCodes\Assignment2\cheat_data.csv', sep=',', header=0)
data_training = np.array(df.values)

# Transform the original training features and classes to numbers
X = []
Y = []
marital_encoding = {"Single": [1, 0, 0], "Married": [0, 1, 0], "Divorced": [0, 0, 1]}
refund_encoding = {"Yes": 1, "No": 0}

for row in data_training:
    new_row = []
    for ind, val in enumerate(row):
        if ind == 0:  # Refund
            new_row.append(refund_encoding[val])
        elif ind == 1:  # Marital Status
            new_row.extend(marital_encoding[val])
        elif ind == 2:  # Taxable Income
            new_row.append(float(val.replace("k", "")))
        elif ind == 3:  # Class (target)
            Y.append(refund_encoding[val])
    X.append(new_row)

# Split the dataset into train and test sets
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)

# Fit a decision tree model using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainY)

# Predict probabilities for all test samples
dt_probs = clf.predict_proba(testX)[:, 1]

# Generate a no skill prediction (random classifier - scores should be all zero)
ns_probs = [0 for _ in range(len(testY))]

# Calculate ROC AUC scores
ns_auc = roc_auc_score(testY, ns_probs)
dt_auc = roc_auc_score(testY, dt_probs)

# Print ROC AUC scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# Calculate ROC curves
ns_fpr, ns_tpr, _ = roc_curve(testY, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testY, dt_probs)

# Plot the ROC curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# Set axis labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Show the plot
plt.show()

